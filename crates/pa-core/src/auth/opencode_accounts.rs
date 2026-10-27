//! `OpenCode` Go account stacking: the slot enumeration, the persisted routing
//! state, the cache-strategic account choice, the usage poll, and the
//! inference verdicts that demote an account.
//!
//! The policy is the omp-go `src/routing.ts` port. Prompt caching is per
//! account: a session pins to one account and stays there (its prompt cache
//! stays warm); new sessions join the warm leader (the account that served
//! most recently) while it has headroom; once the leader passes the spill
//! threshold new sessions start on a sibling and the leader drains.

use std::collections::{BTreeMap, HashSet};
use std::io::Write as _;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt as _;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use serde::{Deserialize, Serialize};

use super::provider_oauth::{is_opencode_go_provider, OPENCODE_GO_PROVIDER_ID};
use super::types::AuthCredential;
use super::AuthStorage;
use pa_ai::opencode_go_spec_generated as spec;

/// The persisted routing state file under the agent directory.
pub const ROUTING_FILE: &str = "opencode-go-routing.json";
/// New sessions stop joining an account once its worst window reaches this
/// percent (the primary override).
pub const SPILL_ENV: &str = spec::SPILL_ENV;
/// The omp-go spelling of the same override.
pub const SPILL_ENV_LEGACY: &str = spec::LEGACY_SPILL_ENV;
pub const DEFAULT_SPILL_PCT: f64 = spec::SPILL_DEFAULT_PCT as f64;
/// A hard verdict (funds/workspace/auth) demotes the account for this long.
pub const VERDICT_TTL_MS: i64 = spec::VERDICT_TTL_MS;
/// A session pin lives this long after its last use.
pub const PIN_TTL_MS: i64 = spec::PIN_TTL_MS;
/// The usage poll runs at most this often per account.
pub const USAGE_THROTTLE_MS: i64 = spec::USAGE_THROTTLE_MS;
/// A usage-priming attempt is retried at most this often, so a transient
/// failure never re-polls on every request.
pub const PRIME_RETRY_MS: i64 = spec::USAGE_PRIME_RETRY_MS;
/// Recorded usage older than this is re-polled, so an exhausted slot heals
/// once its rolling/weekly/monthly window resets.
pub const USAGE_REFRESH_MS: i64 = spec::USAGE_REFRESH_MS;
/// The usage poll never blocks an interactive path: it runs on its own thread
/// with a short bound.
pub const USAGE_TIMEOUT_MS: u64 = spec::USAGE_TIMEOUT_MS as u64;
/// A settled 429 demotes the account for this long when the failure carries no
/// retry-after hint.
pub const DEFAULT_RATE_LIMIT_MS: u64 = spec::RATE_LIMIT_DEFAULT_MS as u64;
/// The longest a 429 can demote an account (a bogus retry-after cannot pin the
/// routing state for days).
pub const MAX_RATE_LIMIT_MS: u64 = spec::RATE_LIMIT_MAX_MS as u64;

/// The Go gateway's usage route.
#[must_use]
pub fn usage_url() -> String {
    format!("{}{}", spec::GATEWAY_OPENAI_BASE_URL, spec::USAGE_PATH)
}
const STATE_MAX_ENTRIES: usize = 200;

/// A hard failure the account reported: 401 auth, 402 funds, 403 workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Verdict {
    pub status: u16,
    pub message: String,
    pub at: i64,
}

/// One subscription window's last reported detail (`rolling`, `weekly`,
/// `monthly`). Absent until a poll answers with a well-formed window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WindowUsage {
    /// Used percent in `0..=100` (the payload's own scale).
    pub percent: f64,
    /// The server's window status: `ok` or `rate-limited`.
    pub status: String,
    /// The ISO-8601 instant the window resets.
    pub resets_at: String,
}

/// One slot's routing state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AccountState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub org_id: Option<String>,
    #[serde(default)]
    pub last_served_at_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub used_pct: Option<f64>,
    #[serde(default)]
    pub used_pct_at_ms: i64,
    /// The last background usage-priming attempt (`0` = never). A transient
    /// failure backs off for [`PRIME_RETRY_MS`].
    #[serde(default)]
    pub prime_attempt_at_ms: i64,
    /// A settled 429 demotes the slot until this instant (epoch ms); selection
    /// skips the slot while `now < rate_limited_until_ms`. `0` = not limited.
    #[serde(default)]
    pub rate_limited_until_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<Verdict>,
    /// Per-window usage detail from the last successful poll. Only a poll
    /// writes here: a freshly logged-in slot starts empty (an old owner's
    /// windows must not describe the new credential).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub windows: BTreeMap<String, WindowUsage>,
}

/// A session's pinned slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Pin {
    pub slot: String,
    pub at: i64,
}

/// `opencode-go-routing.json`: per-slot state and per-session pins.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RoutingState {
    #[serde(default)]
    pub accounts: BTreeMap<String, AccountState>,
    #[serde(default)]
    pub pins: BTreeMap<String, Pin>,
    /// The last successful usage poll (epoch ms), whatever slot answered.
    /// `0` = never polled.
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub usage_fetched_at_ms: i64,
}

// serde's `skip_serializing_if` helper signature is `fn(&i64) -> bool`.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_zero_i64(value: &i64) -> bool {
    *value == 0
}

impl RoutingState {
    fn account(&self, slot: &str) -> AccountState {
        self.accounts.get(slot).cloned().unwrap_or_default()
    }

    /// The used fraction, `0.0` while a poll has not answered.
    fn used_fraction(&self, slot: &str) -> f64 {
        self.accounts
            .get(slot)
            .and_then(|account| account.used_pct)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0)
    }

    /// A fresh (unexpired) verdict, if one is stored.
    #[must_use]
    pub fn fresh_verdict(&self, slot: &str, now: i64) -> Option<&Verdict> {
        self.accounts
            .get(slot)
            .and_then(|account| account.verdict.as_ref())
            .filter(|verdict| now.saturating_sub(verdict.at) < VERDICT_TTL_MS)
    }

    /// Whether the slot is demoted by an unexpired 429.
    #[must_use]
    pub fn rate_limited(&self, slot: &str, now: i64) -> bool {
        self.accounts
            .get(slot)
            .is_some_and(|account| now < account.rate_limited_until_ms)
    }
}

/// The routing state path under `agent_dir`.
#[must_use]
pub fn state_path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(ROUTING_FILE)
}

/// Load the state, pruning expired pins. A missing or unreadable file is an
/// empty state (best-effort persistence).
#[must_use]
pub fn load(agent_dir: &Path) -> RoutingState {
    let now = now_ms();
    let mut state = std::fs::read_to_string(state_path(agent_dir))
        .ok()
        .and_then(|content| serde_json::from_str::<RoutingState>(&content).ok())
        .unwrap_or_default();
    state
        .pins
        .retain(|_, pin| now.saturating_sub(pin.at) < PIN_TTL_MS);
    state
}

/// Persist the state atomically (temp file then rename), mode 0600.
pub fn save(agent_dir: &Path, state: &RoutingState) {
    let mut bounded = state.clone();
    if bounded.accounts.len() > STATE_MAX_ENTRIES {
        let mut entries: Vec<(String, AccountState)> = bounded.accounts.into_iter().collect();
        entries.sort_by(|(left_slot, left), (right_slot, right)| {
            right
                .last_served_at_ms
                .cmp(&left.last_served_at_ms)
                .then(left_slot.cmp(right_slot))
        });
        entries.truncate(STATE_MAX_ENTRIES);
        bounded.accounts = entries.into_iter().collect();
    }
    let Ok(content) = serde_json::to_string_pretty(&bounded) else {
        return;
    };
    let path = state_path(agent_dir);
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let temporary = path.with_extension("json.tmp");
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    options.mode(0o600);
    let write = options
        .open(&temporary)
        .and_then(|mut file| file.write_all(format!("{content}\n").as_bytes()));
    if write.is_ok() {
        let _ = std::fs::rename(&temporary, &path);
    }
}

/// Read-modify-write the state under a process lock.
fn update(agent_dir: &Path, mutate: impl FnOnce(&mut RoutingState)) {
    static LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
    let _guard = LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut state = load(agent_dir);
    mutate(&mut state);
    save(agent_dir, &state);
}

/// The configured spill fraction (percent env → 0..=1).
#[must_use]
pub fn spill_fraction() -> f64 {
    let raw = std::env::var(SPILL_ENV)
        .ok()
        .or_else(|| std::env::var(SPILL_ENV_LEGACY).ok());
    raw.and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|percent| percent.is_finite() && *percent >= 0.0)
        .map_or(DEFAULT_SPILL_PCT / 100.0, |percent| {
            (percent / 100.0).min(1.0)
        })
}

/// The stacked slots in auth.json, base first then `#2`, `#3`, … ascending.
#[must_use]
pub fn slots(auth: &AuthStorage) -> Vec<String> {
    let mut slots: Vec<String> = auth
        .list()
        .into_iter()
        .filter(|provider| is_opencode_go_provider(provider))
        .collect();
    slots.sort_by_key(|slot| slot_number(slot));
    slots
}

/// The slot a fresh login stores under: a slot already holding the same
/// account identity (same accountId or orgId) is replaced in place; otherwise
/// the first free slot (`opencode-go`, then `#2`, `#3`, …).
#[must_use]
pub fn choose_login_slot(
    auth: &AuthStorage,
    account_id: Option<&str>,
    org_id: Option<&str>,
) -> String {
    let existing = slots(auth);
    let credentials = auth.get_all();
    for slot in &existing {
        let Some(super::types::AuthCredential::Oauth {
            account_id: stored_account,
            org_id: stored_org,
            ..
        }) = credentials.credential(slot)
        else {
            continue;
        };
        let same_account = account_id.is_some() && stored_account.as_deref() == account_id;
        let same_org = org_id.is_some() && stored_org.as_deref() == org_id;
        if same_account || same_org {
            return slot.clone();
        }
    }
    let mut number = 1_u64;
    loop {
        let candidate = if number == 1 {
            OPENCODE_GO_PROVIDER_ID.to_string()
        } else {
            format!("opencode-go#{number}")
        };
        if !existing.iter().any(|slot| slot == &candidate) {
            return candidate;
        }
        number += 1;
    }
}

fn slot_number(slot: &str) -> (u8, u64) {
    if slot == OPENCODE_GO_PROVIDER_ID {
        return (0, 0);
    }
    let number = slot
        .strip_prefix("opencode-go#")
        .and_then(|number| number.parse::<u64>().ok())
        .unwrap_or(0);
    (1, number)
}

/// Prime every slot's usage for a routed request: the poll runs on a detached
/// thread and this never waits, so a first request against a pre-existing
/// `auth.json` proceeds on unknown percentages (the 429/verdict path rotates
/// on the next turn once the prime lands).
pub fn refresh_usage_before_routing(agent_dir: &Path, auth: &AuthStorage, slots: &[String]) {
    let _ = spawn_usage_prime(agent_dir, auth, slots);
}

/// The `/login` prime: after the credential write the login waits for the
/// poll (bounded by [`USAGE_TIMEOUT_MS`] per account, all accounts polled in
/// parallel), so the first chat starts from real percentages.
pub fn prime_usage_after_login(agent_dir: &Path, auth: &AuthStorage, slots: &[String]) {
    if let Some(handle) = spawn_usage_prime(agent_dir, auth, slots) {
        let _ = handle.join();
    }
}

/// The slots whose usage must be polled: unknown usage, or usage older than
/// [`USAGE_REFRESH_MS`] (an exhausted slot heals after its window resets),
/// subject to the [`PRIME_RETRY_MS`] backoff for a prior attempt.
fn due_slots(state: &RoutingState, slots: &[String], now: i64) -> Vec<String> {
    slots
        .iter()
        .filter(|slot| {
            state.accounts.get(*slot).is_none_or(|account| {
                let usage_due = account.used_pct_at_ms == 0
                    || now.saturating_sub(account.used_pct_at_ms) >= USAGE_REFRESH_MS;
                usage_due && now.saturating_sub(account.prime_attempt_at_ms) >= PRIME_RETRY_MS
            })
        })
        .cloned()
        .collect()
}

/// Spawn the detached per-slot usage poll. `None` when nothing is due or the
/// thread cannot start.
fn spawn_usage_prime(
    agent_dir: &Path,
    auth: &AuthStorage,
    slots: &[String],
) -> Option<std::thread::JoinHandle<()>> {
    if slots.is_empty() {
        return None;
    }
    let state = load(agent_dir);
    let now = now_ms();
    let due = due_slots(&state, slots, now);
    if due.is_empty() {
        return None;
    }
    // Record the attempt before the poll: a transient failure must not
    // re-trigger the prime on every request.
    for slot in &due {
        note_prime_attempt(agent_dir, slot);
    }
    let credentials = usage_candidates(auth, &due);
    if credentials.is_empty() {
        return None;
    }
    let agent_dir = agent_dir.to_path_buf();
    // The caller may already be inside a tokio runtime; the priming poll runs on
    // its own thread (and runtime).
    std::thread::Builder::new()
        .name("opencode-usage-priming".to_string())
        .spawn(move || {
            let Some(runtime) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .ok()
            else {
                return;
            };
            let results = runtime.block_on(async {
                let futures = credentials.iter().map(|(slot, token, org_id)| async move {
                    (
                        slot.clone(),
                        fetch_usage(token, org_id.as_deref(), None).await,
                    )
                });
                futures::future::join_all(futures).await
            });
            for (slot, outcome) in results {
                match outcome {
                    UsageOutcome::Ok(windows) => note_usage_report(&agent_dir, &slot, windows),
                    UsageOutcome::Verdict(status, message) => {
                        record_verdict(&agent_dir, &slot, status, &message);
                    }
                    UsageOutcome::Transient => {}
                }
            }
        })
        .ok()
}

/// The slots a usage poll can run for: an OAuth slot's access token, or a
/// literal API key as the bearer (the legacy pre-stacking `auth.json` shape).
/// A `!`-command key is not a literal bearer and is not polled.
fn usage_candidates(auth: &AuthStorage, slots: &[String]) -> Vec<(String, String, Option<String>)> {
    slots
        .iter()
        .filter_map(|slot| match auth.get_all().credential(slot) {
            Some(AuthCredential::Oauth { access, org_id, .. }) => {
                Some((slot.clone(), access, org_id))
            }
            Some(AuthCredential::ApiKey { key, .. }) => {
                (!key.starts_with('!')).then_some((slot.clone(), key, None))
            }
            _ => None,
        })
        .collect()
}

/// Record that a background usage prime was attempted for `slot`.
fn note_prime_attempt(agent_dir: &Path, slot: &str) {
    let now = now_ms();
    update(agent_dir, |state| {
        state
            .accounts
            .entry(slot.to_string())
            .or_default()
            .prime_attempt_at_ms = now;
    });
}

/// The serving slot's workspace id from its stored credential.
#[must_use]
pub fn slot_org_id(auth: &AuthStorage, slot: &str) -> Option<String> {
    match auth.get_all().credential(slot) {
        Some(AuthCredential::Oauth { org_id, .. }) => org_id,
        _ => None,
    }
}

/// Seed each slot's `org_id` from its stored credential: the state file then
/// carries the workspace id before the first request (a never-served slot).
pub fn seed_org_ids(agent_dir: &Path, auth: &AuthStorage, slots: &[String]) {
    let all = auth.get_all();
    let orgs: Vec<(String, String)> = slots
        .iter()
        .filter_map(|slot| match all.credential(slot) {
            Some(AuthCredential::Oauth {
                org_id: Some(org_id),
                ..
            }) => Some((slot.clone(), org_id)),
            _ => None,
        })
        .collect();
    if orgs.is_empty() {
        return;
    }
    update(agent_dir, |state| {
        for (slot, org_id) in orgs {
            state.accounts.entry(slot).or_default().org_id = Some(org_id);
        }
    });
}

/// Record that `slot` just served a request (the warm-cache leader signal).
pub fn note_served(agent_dir: &Path, slot: &str, org_id: Option<&str>) {
    let org_id = org_id.map(str::to_string);
    let now = now_ms();
    update(agent_dir, |state| {
        let account = state.accounts.entry(slot.to_string()).or_default();
        account.last_served_at_ms = now;
        if let Some(org_id) = org_id {
            account.org_id = Some(org_id);
        }
    });
}

/// Record a settled 429 on `slot`: the account is exhausted until the
/// retry-after window ends, and every pin pointing at it is dropped so the next
/// turn rotates. `note_usage` clears the demotion once a fresh poll reports
/// `usedPct < 1`.
pub fn record_rate_limit(agent_dir: &Path, slot: &str, retry_after_ms: Option<u64>) {
    let now = now_ms();
    let retry_after_ms =
        retry_after_ms.map_or(DEFAULT_RATE_LIMIT_MS, |ms| ms.min(MAX_RATE_LIMIT_MS));
    let retry_after_ms = i64::try_from(retry_after_ms).unwrap_or(spec::RATE_LIMIT_DEFAULT_MS);
    let slot = slot.to_string();
    update(agent_dir, |state| {
        let account = state.accounts.entry(slot.clone()).or_default();
        account.used_pct = Some(1.0);
        account.used_pct_at_ms = now;
        account.rate_limited_until_ms = now.saturating_add(retry_after_ms);
        state.pins.retain(|_, pin| pin.slot != slot);
    });
}

/// Record the worst-window used fraction observed at a usage poll. A poll that
/// reports headroom clears a 429 demotion (the window reset).
pub fn note_usage(agent_dir: &Path, slot: &str, used_pct: f64) {
    let used_pct = used_pct.clamp(0.0, 1.0);
    let now = now_ms();
    update(agent_dir, |state| {
        let account = state.accounts.entry(slot.to_string()).or_default();
        account.used_pct = Some(used_pct);
        account.used_pct_at_ms = now;
        if used_pct < 1.0 {
            account.rate_limited_until_ms = 0;
        }
        state.usage_fetched_at_ms = now;
    });
}

/// Record a poll's per-window detail. The worst window still drives
/// `used_pct` (the routing signal); the map is kept so a client surface can
/// show each window. An empty report changes nothing.
pub fn note_usage_report(agent_dir: &Path, slot: &str, windows: BTreeMap<String, WindowUsage>) {
    if windows.is_empty() {
        return;
    }
    let used_pct = windows
        .values()
        .map(|window| window.percent / 100.0)
        .fold(0.0_f64, f64::max)
        .clamp(0.0, 1.0);
    let now = now_ms();
    update(agent_dir, |state| {
        let account = state.accounts.entry(slot.to_string()).or_default();
        account.windows = windows;
        account.used_pct = Some(used_pct);
        account.used_pct_at_ms = now;
        if used_pct < 1.0 {
            account.rate_limited_until_ms = 0;
        }
        state.usage_fetched_at_ms = now;
    });
}

/// Record a hard verdict and drop every session pin that pointed at the slot
/// (the next turn rotates to a healthy sibling).
pub fn record_verdict(agent_dir: &Path, slot: &str, status: u16, message: &str) {
    let verdict = Verdict {
        status,
        message: message.to_string(),
        at: now_ms(),
    };
    let slot = slot.to_string();
    update(agent_dir, |state| {
        state.accounts.entry(slot.clone()).or_default().verdict = Some(verdict);
        state.pins.retain(|_, pin| pin.slot != slot);
    });
}

/// Forget only the stored hard verdict, so a slot that just re-authenticated
/// is routable again without waiting out [`VERDICT_TTL_MS`]. Usage and pins
/// are untouched.
pub fn clear_verdict(agent_dir: &Path, slot: &str) {
    let slot = slot.to_string();
    update(agent_dir, |state| {
        if let Some(account) = state.accounts.get_mut(&slot) {
            account.verdict = None;
        }
    });
}

/// Drop the slot's routing state entirely — usage, org id, verdict, rate
/// limit — and every pin pointing at it. A fresh login replaces the slot's
/// credential, so the previous owner's 100% usage or verdict must not demote
/// the account that now holds it.
pub fn reset_slot_state(agent_dir: &Path, slot: &str) {
    forget_slot(agent_dir, slot);
}

/// The same removal for `/logout`: the slot's credential is gone, so its
/// routing state and pins must not outlive it.
pub fn forget_slot(agent_dir: &Path, slot: &str) {
    let slot = slot.to_string();
    update(agent_dir, |state| {
        state.accounts.remove(&slot);
        state.pins.retain(|_, pin| pin.slot != slot);
    });
}

/// The session's live pin, if it points at a known, healthy slot.
#[must_use]
pub fn session_pin(state: &RoutingState, session_id: Option<&str>, now: i64) -> Option<String> {
    let session_id = session_id?;
    let pin = state.pins.get(session_id)?;
    if now.saturating_sub(pin.at) >= PIN_TTL_MS {
        return None;
    }
    if state.fresh_verdict(&pin.slot, now).is_some() {
        return None;
    }
    if state.rate_limited(&pin.slot, now) {
        return None;
    }
    Some(pin.slot.clone())
}

/// Pin a session to `slot` (a fresh session records its choice here).
pub fn pin_session(agent_dir: &Path, session_id: &str, slot: &str) {
    let slot = slot.to_string();
    let session_id = session_id.to_string();
    let now = now_ms();
    update(agent_dir, |state| {
        state.pins.insert(session_id, Pin { slot, at: now });
    });
}

/// The routing decision: which slot should a session use?
///
/// An existing pin wins (drain in place); otherwise the warm leader — no fresh
/// verdict, not exhausted; below the spill threshold when any such account
/// exists — sorted by last-served descending, used fraction ascending, slot.
#[must_use]
pub fn choose_slot(
    state: &RoutingState,
    slots: &[String],
    session_id: Option<&str>,
    spill: f64,
    now: i64,
) -> Option<String> {
    if let Some(pin) = session_pin(state, session_id, now) {
        if slots.contains(&pin) {
            return Some(pin);
        }
    }
    let usable: Vec<&String> = slots
        .iter()
        .filter(|slot| state.fresh_verdict(slot, now).is_none())
        .filter(|slot| !state.rate_limited(slot, now))
        .filter(|slot| state.used_fraction(slot) < 1.0)
        .collect();
    if usable.is_empty() {
        return None;
    }
    let mut below: Vec<&String> = usable
        .iter()
        .copied()
        .filter(|slot| state.used_fraction(slot) < spill)
        .collect();
    if below.is_empty() {
        below = usable;
    }
    below.sort_by(|left, right| {
        let left_state = state.account(left);
        let right_state = state.account(right);
        right_state
            .last_served_at_ms
            .cmp(&left_state.last_served_at_ms)
            .then(
                state
                    .used_fraction(left)
                    .total_cmp(&state.used_fraction(right)),
            )
            .then(left.cmp(right))
    });
    below.first().map(|slot| (*slot).clone())
}

/// Resolve the request slot and pin it for a new session.
#[must_use]
pub fn resolve(agent_dir: &Path, slots: &[String], session_id: Option<&str>) -> Option<String> {
    let now = now_ms();
    let state = load(agent_dir);
    let slot = choose_slot(&state, slots, session_id, spill_fraction(), now)?;
    if let Some(session_id) = session_id {
        if session_pin(&state, Some(session_id), now).as_deref() != Some(slot.as_str()) {
            pin_session(agent_dir, session_id, &slot);
        }
    }
    Some(slot)
}

/// Whether a status is a credential verdict (auth, funds, workspace).
#[must_use]
pub fn is_verdict_status(status: u16) -> bool {
    spec::VERDICT_STATUSES.contains(&status)
}

/// Parse the usage payload's per-window detail. Malformed windows are
/// skipped, not fatal.
#[must_use]
pub fn parse_usage_windows(body: &str) -> BTreeMap<String, WindowUsage> {
    let Some(payload) = serde_json::from_str::<serde_json::Value>(body).ok() else {
        return BTreeMap::new();
    };
    let Some(usage) = payload.get("usage").and_then(serde_json::Value::as_object) else {
        return BTreeMap::new();
    };
    let mut windows = BTreeMap::new();
    for &key in spec::USAGE_WINDOWS {
        let Some(window) = usage.get(key).and_then(serde_json::Value::as_object) else {
            continue;
        };
        let Some(percent) = window
            .get("percent")
            .and_then(serde_json::Value::as_f64)
            .filter(|percent| percent.is_finite() && (0.0..=100.0).contains(percent))
        else {
            continue;
        };
        let Some(status) = window
            .get("status")
            .and_then(serde_json::Value::as_str)
            .filter(|status| matches!(*status, "ok" | "rate-limited"))
        else {
            continue;
        };
        let Some(resets_at) = window.get("resetsAt").and_then(serde_json::Value::as_str) else {
            continue;
        };
        windows.insert(
            key.to_string(),
            WindowUsage {
                percent,
                status: status.to_string(),
                resets_at: resets_at.to_string(),
            },
        );
    }
    windows
}

/// Parse the usage payload's worst-window used fraction (max percent / 100).
/// Malformed windows are skipped, not fatal.
#[must_use]
pub fn parse_usage(body: &str) -> Option<f64> {
    parse_usage_windows(body)
        .values()
        .map(|window| window.percent / 100.0)
        .fold(None, |worst: Option<f64>, used| {
            Some(worst.map_or(used, |current| current.max(used)))
        })
}

/// The server's error message from an error payload, if any.
fn error_message(body: &str) -> Option<String> {
    let payload: serde_json::Value = serde_json::from_str(body).ok()?;
    payload
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

/// Poll the account's usage on a detached thread: throttled per slot and
/// single-flight within the process, never on a paint path.
pub fn maybe_poll_usage(
    agent_dir: &Path,
    slot: &str,
    token: &str,
    org_id: Option<&str>,
    session_id: Option<&str>,
) {
    static IN_FLIGHT: LazyLock<Mutex<HashSet<String>>> =
        LazyLock::new(|| Mutex::new(HashSet::new()));
    let now = now_ms();
    let state = load(agent_dir);
    if let Some(account) = state.accounts.get(slot) {
        if account.used_pct_at_ms > 0
            && now.saturating_sub(account.used_pct_at_ms) < USAGE_THROTTLE_MS
        {
            return;
        }
    }
    {
        let mut in_flight = IN_FLIGHT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !in_flight.insert(slot.to_string()) {
            return;
        }
    }
    let agent_dir = agent_dir.to_path_buf();
    let slot = slot.to_string();
    let slot_key = slot.clone();
    let token = token.to_string();
    let org_id = org_id.map(str::to_string);
    let session_id = session_id.map(str::to_string);
    let spawned = std::thread::Builder::new()
        .name("opencode-usage".to_string())
        .spawn(move || {
            let Some(runtime) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .ok()
            else {
                return;
            };
            let result = runtime.block_on(fetch_usage(
                &token,
                org_id.as_deref(),
                session_id.as_deref(),
            ));
            {
                let mut in_flight = IN_FLIGHT
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                in_flight.remove(&slot);
            }
            match result {
                UsageOutcome::Ok(windows) => note_usage_report(&agent_dir, &slot, windows),
                UsageOutcome::Verdict(status, message) => {
                    record_verdict(&agent_dir, &slot, status, &message);
                }
                UsageOutcome::Transient => {}
            }
        });
    if spawned.is_err() {
        let mut in_flight = IN_FLIGHT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        in_flight.remove(&slot_key);
    }
}

enum UsageOutcome {
    Ok(BTreeMap<String, WindowUsage>),
    Verdict(u16, String),
    Transient,
}

async fn fetch_usage(token: &str, org_id: Option<&str>, session_id: Option<&str>) -> UsageOutcome {
    let Ok(client) = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(USAGE_TIMEOUT_MS))
        .build()
    else {
        return UsageOutcome::Transient;
    };
    let session = pa_ai::oauth::harness_session_id(session_id);
    let mut request = client
        .get(usage_url())
        .header("accept", "application/json")
        .header("authorization", format!("Bearer {token}"))
        .header("user-agent", pa_ai::oauth::OPENCODE_USER_AGENT)
        .header(spec::HEADER_CLIENT, "cli")
        .header(spec::HEADER_SESSION, &session)
        .header(spec::HEADER_SESSION_ID, &session)
        .header(spec::HEADER_SESSION_AFFINITY, &session)
        .header(spec::HEADER_SESSION_ID_ALIAS, &session);
    for (name, value) in pa_ai::oauth::trace_headers() {
        request = request.header(name, value);
    }
    if let Some(org_id) = org_id {
        request = request.header(spec::HEADER_ORG, org_id);
    }
    let Ok(response) = request.send().await else {
        return UsageOutcome::Transient;
    };
    let status = response.status().as_u16();
    let body = response.text().await.unwrap_or_default();
    if is_verdict_status(status) {
        let detail = error_message(&body)
            .map(|message| format!(": {message}"))
            .unwrap_or_default();
        return UsageOutcome::Verdict(
            status,
            format!("OpenCode Go usage endpoint returned {status}{detail}"),
        );
    }
    // A rate-limited answer is the settle path's signal, not a usage reading:
    // the poll stays transient instead of recording a verdict.
    if status == spec::RATE_LIMIT_STATUS {
        return UsageOutcome::Transient;
    }
    if !(200..300).contains(&status) {
        return UsageOutcome::Transient;
    }
    let windows = parse_usage_windows(&body);
    if windows.is_empty() {
        UsageOutcome::Transient
    } else {
        UsageOutcome::Ok(windows)
    }
}

fn now_ms() -> i64 {
    #[allow(clippy::cast_possible_truncation)]
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as i64)
}

/// `now_ms` for tests and callers that build pins.
#[must_use]
pub fn epoch_ms() -> i64 {
    now_ms()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::types::AuthStorageData;
    use crate::auth::{AuthCredential, NoOAuth};
    use std::sync::Arc;

    #[test]
    fn usage_url_is_the_spec_gateway_base_plus_usage_path() {
        assert_eq!(usage_url(), "https://opencode.ai/zen/go/v1/usage");
        assert_eq!(
            usage_url(),
            format!("{}{}", spec::GATEWAY_OPENAI_BASE_URL, spec::USAGE_PATH)
        );
    }

    fn credential(account_id: Option<&str>, org_id: Option<&str>) -> AuthCredential {
        AuthCredential::Oauth {
            access: "access".to_string(),
            refresh: None,
            expires: i64::MAX,
            account_id: account_id.map(str::to_string),
            endpoint: None,
            token_endpoint: None,
            client_id: None,
            resource: None,
            issuer: None,
            audience_mode: None,
            org_id: org_id.map(str::to_string),
            api_endpoint: None,
            email: None,
            enterprise_url: None,
        }
    }

    fn account(last_served_at_ms: i64, used_pct: f64) -> AccountState {
        AccountState {
            org_id: None,
            last_served_at_ms,
            used_pct: Some(used_pct),
            used_pct_at_ms: 1,
            prime_attempt_at_ms: 0,
            rate_limited_until_ms: 0,
            verdict: None,
            windows: BTreeMap::new(),
        }
    }

    fn state_with(entries: &[(&str, AccountState)]) -> RoutingState {
        RoutingState {
            accounts: entries
                .iter()
                .map(|(slot, state)| ((*slot).to_string(), state.clone()))
                .collect(),
            pins: BTreeMap::new(),
            usage_fetched_at_ms: 0,
        }
    }

    fn slot_list(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn slots_enumerate_base_then_numeric() {
        let mut data = AuthStorageData::default();
        for slot in [
            "opencode-go#10",
            "anthropic",
            "opencode-go",
            "opencode-go#2",
        ] {
            data.insert(slot, &credential(None, None));
        }
        let auth = AuthStorage::in_memory_without_env(&data, Arc::new(NoOAuth));
        assert_eq!(
            slots(&auth),
            vec!["opencode-go", "opencode-go#2", "opencode-go#10"]
        );
    }

    #[test]
    fn a_login_replaces_the_same_identity_and_appends_otherwise() {
        let mut data = AuthStorageData::default();
        data.insert("opencode-go", &credential(Some("usr_1"), Some("org_a")));
        let auth = AuthStorage::in_memory_without_env(&data, Arc::new(NoOAuth));
        assert_eq!(
            choose_login_slot(&auth, Some("usr_1"), Some("org_a")),
            "opencode-go",
            "the same account replaces its slot"
        );
        assert_eq!(
            choose_login_slot(&auth, Some("usr_2"), Some("org_b")),
            "opencode-go#2",
            "a different account stacks"
        );
        let empty =
            AuthStorage::in_memory_without_env(&AuthStorageData::default(), Arc::new(NoOAuth));
        assert_eq!(
            choose_login_slot(&empty, Some("usr_1"), None),
            "opencode-go"
        );
    }

    #[test]
    fn the_warm_leader_takes_new_sessions() {
        let state = state_with(&[
            ("opencode-go", account(100, 0.2)),
            ("opencode-go#2", account(200, 0.3)),
        ]);
        assert_eq!(
            choose_slot(
                &state,
                &slot_list(&["opencode-go", "opencode-go#2"]),
                None,
                0.9,
                1_000
            )
            .as_deref(),
            Some("opencode-go#2")
        );
    }

    #[test]
    fn a_leader_past_the_spill_threshold_drains() {
        let state = state_with(&[
            ("opencode-go", account(200, 0.95)),
            ("opencode-go#2", account(100, 0.2)),
        ]);
        assert_eq!(
            choose_slot(
                &state,
                &slot_list(&["opencode-go", "opencode-go#2"]),
                None,
                0.9,
                1_000
            )
            .as_deref(),
            Some("opencode-go#2")
        );
    }

    #[test]
    fn when_every_account_is_past_the_spill_threshold_the_leader_still_wins() {
        let state = state_with(&[
            ("opencode-go", account(200, 0.95)),
            ("opencode-go#2", account(100, 0.92)),
        ]);
        assert_eq!(
            choose_slot(
                &state,
                &slot_list(&["opencode-go", "opencode-go#2"]),
                None,
                0.9,
                1_000
            )
            .as_deref(),
            Some("opencode-go"),
            "consolidation is never worse for cache"
        );
    }

    #[test]
    fn an_exhausted_or_verdict_account_is_not_usable() {
        let exhausted = state_with(&[
            ("opencode-go", account(300, 1.0)),
            ("opencode-go#2", account(100, 0.2)),
        ]);
        assert_eq!(
            choose_slot(
                &exhausted,
                &slot_list(&["opencode-go", "opencode-go#2"]),
                None,
                0.9,
                1_000
            )
            .as_deref(),
            Some("opencode-go#2")
        );
        let mut verdict = state_with(&[
            ("opencode-go", account(300, 0.1)),
            ("opencode-go#2", account(100, 0.2)),
        ]);
        verdict.accounts.get_mut("opencode-go").unwrap().verdict = Some(Verdict {
            status: 403,
            message: "Workspace access denied".to_string(),
            at: 900,
        });
        assert_eq!(
            choose_slot(
                &verdict,
                &slot_list(&["opencode-go", "opencode-go#2"]),
                None,
                0.9,
                1_000
            )
            .as_deref(),
            Some("opencode-go#2")
        );
        assert_eq!(
            choose_slot(
                &verdict,
                &slot_list(&["opencode-go", "opencode-go#2"]),
                None,
                0.9,
                900 + VERDICT_TTL_MS + 1
            )
            .as_deref(),
            Some("opencode-go"),
            "the verdict expires after its TTL"
        );
    }

    #[test]
    fn a_base_api_key_slot_is_polled_and_its_usage_verdict_serves_the_oauth_sibling() {
        let dir = tempfile::tempdir().unwrap();
        let mut data = AuthStorageData::default();
        data.insert(
            "opencode-go",
            &AuthCredential::ApiKey {
                key: "sk-legacy".to_string(),
                prime_team: None,
            },
        );
        data.insert("opencode-go#2", &credential(Some("usr_2"), Some("org_b")));
        let auth = AuthStorage::in_memory_without_env(&data, Arc::new(NoOAuth));
        let slots = slots(&auth);
        assert_eq!(slots, vec!["opencode-go", "opencode-go#2"]);
        assert_eq!(
            usage_candidates(&auth, &slots),
            vec![
                ("opencode-go".to_string(), "sk-legacy".to_string(), None),
                (
                    "opencode-go#2".to_string(),
                    "access".to_string(),
                    Some("org_b".to_string()),
                ),
            ],
            "the legacy API key is polled as a plain bearer, without a workspace"
        );
        // The usage poll's 401 lands as a verdict on the base key: routing then
        // skips it and serves the OAuth sibling.
        record_verdict(
            dir.path(),
            "opencode-go",
            401,
            "OpenCode Go usage endpoint returned 401",
        );
        assert_eq!(
            resolve(dir.path(), &slots, Some("session-a")).as_deref(),
            Some("opencode-go#2"),
            "the OAuth slot serves once the API-key slot is demoted"
        );
    }

    #[test]
    fn a_command_key_slot_is_not_polled() {
        let mut data = AuthStorageData::default();
        data.insert(
            "opencode-go",
            &AuthCredential::ApiKey {
                key: "!printf sk-legacy".to_string(),
                prime_team: None,
            },
        );
        let auth = AuthStorage::in_memory_without_env(&data, Arc::new(NoOAuth));
        assert!(usage_candidates(&auth, &slots(&auth)).is_empty());
    }

    #[test]
    fn a_pin_wins_and_expires() {
        let mut state = state_with(&[
            ("opencode-go", account(100, 0.99)),
            ("opencode-go#2", account(200, 0.1)),
        ]);
        state.pins.insert(
            "session-1".to_string(),
            Pin {
                slot: "opencode-go".to_string(),
                at: 1_000,
            },
        );
        assert_eq!(
            choose_slot(
                &state,
                &slot_list(&["opencode-go", "opencode-go#2"]),
                Some("session-1"),
                0.9,
                1_100
            )
            .as_deref(),
            Some("opencode-go"),
            "an explicit pin drains in place"
        );
        assert_eq!(
            choose_slot(
                &state,
                &slot_list(&["opencode-go", "opencode-go#2"]),
                Some("session-1"),
                0.9,
                1_000 + PIN_TTL_MS + 1
            )
            .as_deref(),
            Some("opencode-go#2"),
            "an expired pin no longer wins"
        );
    }

    #[test]
    fn a_verdict_drops_the_pins_and_rotates() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        pin_session(agent, "session-1", "opencode-go");
        record_verdict(agent, "opencode-go", 403, "Workspace access denied");
        let state = load(agent);
        assert!(
            state.pins.is_empty(),
            "the verdict drops every pin pointing at the account"
        );
        assert_eq!(
            state
                .fresh_verdict("opencode-go", epoch_ms())
                .unwrap()
                .status,
            403
        );
        assert_eq!(
            resolve(
                agent,
                &slot_list(&["opencode-go", "opencode-go#2"]),
                Some("session-1")
            )
            .as_deref(),
            Some("opencode-go#2"),
            "the next turn rotates to a healthy sibling"
        );
        assert!(is_verdict_status(401));
        assert!(is_verdict_status(402));
        assert!(is_verdict_status(403));
        assert!(!is_verdict_status(429));
    }

    #[test]
    fn a_reauthenticated_slot_is_routable_again_once_the_verdict_is_cleared() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        let slots = slot_list(&["opencode-go"]);
        note_usage(agent, "opencode-go", 0.5);
        record_verdict(agent, "opencode-go", 401, "no usable token; run /login");
        assert!(
            load(agent)
                .fresh_verdict("opencode-go", epoch_ms())
                .is_some(),
            "the recorded verdict excludes the slot"
        );
        assert_eq!(resolve(agent, &slots, Some("session-1")), None);
        clear_verdict(agent, "opencode-go");
        let state = load(agent);
        assert!(
            state.fresh_verdict("opencode-go", epoch_ms()).is_none(),
            "clearing removes only the verdict"
        );
        assert_eq!(
            state.used_fraction("opencode-go"),
            0.5,
            "clearing a verdict leaves the usage record intact"
        );
        assert_eq!(
            resolve(agent, &slots, Some("session-1")).as_deref(),
            Some("opencode-go"),
            "a re-authenticated slot is selectable again within the TTL"
        );
    }

    #[test]
    fn resetting_a_replaced_slot_drops_only_its_state_and_pins() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        note_usage(agent, "opencode-go", 1.0);
        note_served(agent, "opencode-go#2", Some("org_b"));
        record_verdict(agent, "opencode-go", 402, "insufficient funds");
        pin_session(agent, "session-a", "opencode-go");
        pin_session(agent, "session-b", "opencode-go#2");

        reset_slot_state(agent, "opencode-go");

        let state = load(agent);
        assert!(
            !state.accounts.contains_key("opencode-go"),
            "the replaced slot's usage, org and verdict are gone"
        );
        assert!(state.fresh_verdict("opencode-go", epoch_ms()).is_none());
        assert_eq!(
            state
                .accounts
                .get("opencode-go#2")
                .and_then(|sibling| sibling.org_id.as_deref()),
            Some("org_b"),
            "the sibling keeps its state"
        );
        assert_eq!(
            state.pins.get("session-b").map(|pin| pin.slot.as_str()),
            Some("opencode-go#2"),
            "the sibling's pin survives"
        );
        assert!(
            !state.pins.values().any(|pin| pin.slot == "opencode-go"),
            "pins to the replaced slot are dropped"
        );
        assert_eq!(
            resolve(
                agent,
                &slot_list(&["opencode-go", "opencode-go#2"]),
                Some("session-a")
            )
            .as_deref(),
            Some("opencode-go#2"),
            "a fresh login's slot is not demoted by the previous owner's usage"
        );
    }

    #[test]
    fn forgetting_a_logged_out_slot_drops_its_state_and_pins() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        note_usage(agent, "opencode-go", 0.25);
        note_usage(agent, "opencode-go#2", 0.5);
        record_verdict(agent, "opencode-go", 401, "no usable token; run /login");
        pin_session(agent, "session-a", "opencode-go");
        pin_session(agent, "session-b", "opencode-go#2");

        forget_slot(agent, "opencode-go");

        let state = load(agent);
        assert!(
            !state.accounts.contains_key("opencode-go"),
            "the logged-out slot's usage and verdict are gone"
        );
        assert!(
            !state.pins.values().any(|pin| pin.slot == "opencode-go"),
            "pins to the logged-out slot are dropped"
        );
        assert_eq!(
            state.used_fraction("opencode-go#2"),
            0.5,
            "the sibling's usage is untouched"
        );
        assert_eq!(
            state.pins.get("session-b").map(|pin| pin.slot.as_str()),
            Some("opencode-go#2")
        );
    }

    #[test]
    fn a_settled_429_demotes_the_slot_and_rotates() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        let slots = slot_list(&["opencode-go", "opencode-go#2"]);
        note_served(agent, "opencode-go", None);
        pin_session(agent, "session-1", "opencode-go");

        record_rate_limit(agent, "opencode-go", Some(90_000));

        let state = load(agent);
        assert_eq!(
            state.accounts["opencode-go"].used_pct,
            Some(1.0),
            "a 429 stores the exhaust"
        );
        assert!(
            state.rate_limited("opencode-go", epoch_ms()),
            "the slot is demoted for the retry-after window"
        );
        assert!(
            state.pins.is_empty(),
            "the 429 drops the pins that pointed at the slot"
        );
        assert_eq!(
            resolve(agent, &slots, Some("session-1")).as_deref(),
            Some("opencode-go#2"),
            "the next resolve uses the sibling"
        );
        assert!(
            !state.rate_limited(
                "opencode-go",
                epoch_ms() + i64::try_from(MAX_RATE_LIMIT_MS).unwrap() + 1
            ),
            "the demotion expires"
        );
    }

    #[test]
    fn a_rate_limit_defaults_to_a_minute_and_expires() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        record_rate_limit(agent, "opencode-go", None);
        let state = load(agent);
        let until = state.accounts["opencode-go"].rate_limited_until_ms;
        assert!(until > epoch_ms());
        assert!(until <= epoch_ms() + i64::try_from(DEFAULT_RATE_LIMIT_MS).unwrap() + 1_000);
        assert!(!state.rate_limited("opencode-go", until));
        assert!(state.rate_limited("opencode-go", until - 1));
    }

    #[test]
    fn a_usage_poll_under_full_clears_the_rate_limit() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        record_rate_limit(agent, "opencode-go", Some(60_000));
        assert!(load(agent).rate_limited("opencode-go", epoch_ms()));
        note_usage(agent, "opencode-go", 0.4);
        assert!(
            !load(agent).rate_limited("opencode-go", epoch_ms()),
            "a poll reporting headroom clears the 429 demotion"
        );
    }

    #[test]
    fn due_slots_repolls_stale_and_exhausted_usage() {
        let now = 10_000_000;
        let exhausted = AccountState {
            org_id: None,
            last_served_at_ms: 0,
            used_pct: Some(1.0),
            used_pct_at_ms: now - USAGE_REFRESH_MS,
            prime_attempt_at_ms: 0,
            rate_limited_until_ms: 0,
            verdict: None,
            windows: BTreeMap::new(),
        };
        let fresh = AccountState {
            used_pct_at_ms: now - USAGE_REFRESH_MS + 1,
            ..exhausted.clone()
        };
        let retrying = AccountState {
            prime_attempt_at_ms: now - PRIME_RETRY_MS + 1,
            ..exhausted.clone()
        };
        let unknown = AccountState {
            used_pct: None,
            used_pct_at_ms: 0,
            ..exhausted.clone()
        };
        let state = state_with(&[
            ("exhausted", exhausted),
            ("fresh", fresh),
            ("retrying", retrying),
            ("unknown", unknown),
        ]);
        let slots = slot_list(&["exhausted", "fresh", "retrying", "unknown", "never"]);
        assert_eq!(
            due_slots(&state, &slots, now),
            vec!["exhausted", "unknown", "never"],
            "stale and unknown usage is polled; fresh usage and a backoff attempt are not"
        );
    }

    #[test]
    fn a_recent_poll_suppresses_the_refresh() {
        let now = 10_000_000;
        let state = state_with(&[(
            "opencode-go",
            AccountState {
                used_pct: Some(1.0),
                used_pct_at_ms: now - USAGE_REFRESH_MS,
                prime_attempt_at_ms: now - PRIME_RETRY_MS,
                ..AccountState::default()
            },
        )]);
        assert_eq!(
            due_slots(
                &state,
                &slot_list(&["opencode-go"]),
                now - USAGE_REFRESH_MS / 2
            ),
            Vec::<String>::new(),
            "usage polled within five minutes is not re-polled"
        );
        assert_eq!(
            due_slots(&state, &slot_list(&["opencode-go"]), now),
            vec!["opencode-go"],
            "usage older than five minutes is re-polled"
        );
    }

    #[test]
    fn usage_parsing_takes_the_worst_well_formed_window() {
        let body = serde_json::json!({
            "usage": {
                "rolling": {"percent": 50, "status": "ok", "resetsAt": "2026-10-25T20:00:00Z"},
                "weekly": {"percent": 95, "status": "rate-limited", "resetsAt": "2026-11-01T00:00:00Z"},
                "monthly": {"percent": 10, "status": "ok", "resetsAt": "2026-11-20T00:00:00Z"}
            }
        })
        .to_string();
        assert_eq!(parse_usage(&body), Some(0.95));
    }

    #[test]
    fn malformed_or_missing_windows_are_skipped() {
        assert_eq!(parse_usage(r#"{"error":{"message":"nope"}}"#), None);
        assert_eq!(parse_usage("not json"), None);
        let out_of_range = serde_json::json!({
            "usage": {"rolling": {"percent": 120, "status": "ok", "resetsAt": "x"}}
        })
        .to_string();
        assert_eq!(parse_usage(&out_of_range), None);
        let bad_status = serde_json::json!({
            "usage": {"rolling": {"percent": 20, "status": "weird", "resetsAt": "x"}}
        })
        .to_string();
        assert_eq!(parse_usage(&bad_status), None);
        let missing_reset = serde_json::json!({
            "usage": {"rolling": {"percent": 20, "status": "ok"}}
        })
        .to_string();
        assert_eq!(parse_usage(&missing_reset), None);
        let partial = serde_json::json!({
            "usage": {
                "rolling": {"percent": 20, "status": "ok", "resetsAt": "x"},
                "weekly": {"percent": "bad"}
            }
        })
        .to_string();
        assert_eq!(parse_usage(&partial), Some(0.2), "good windows still count");
    }

    #[test]
    fn usage_report_parses_every_window() {
        let body = serde_json::json!({
            "usage": {
                "rolling": {"percent": 50, "status": "ok", "resetsAt": "2026-10-25T20:00:00Z"},
                "weekly": {"percent": 95, "status": "rate-limited", "resetsAt": "2026-11-01T00:00:00Z"},
                "monthly": {"percent": 10, "status": "ok", "resetsAt": "2026-11-20T00:00:00Z"}
            }
        })
        .to_string();
        let windows = parse_usage_windows(&body);
        assert_eq!(windows.len(), 3);
        assert_eq!(windows["rolling"].percent, 50.0);
        assert_eq!(windows["rolling"].status, "ok");
        assert_eq!(windows["rolling"].resets_at, "2026-10-25T20:00:00Z");
        assert_eq!(windows["weekly"].status, "rate-limited");
        assert_eq!(windows["monthly"].resets_at, "2026-11-20T00:00:00Z");
        assert_eq!(parse_usage(&body), Some(0.95));
    }

    #[test]
    fn malformed_windows_are_omitted_from_the_report() {
        let body = serde_json::json!({
            "usage": {
                "rolling": {"percent": 20, "status": "ok", "resetsAt": "x"},
                "weekly": {"percent": "bad", "status": "ok", "resetsAt": "x"},
                "monthly": {"percent": 10, "status": "weird", "resetsAt": "x"}
            }
        })
        .to_string();
        let windows = parse_usage_windows(&body);
        assert_eq!(
            windows.keys().collect::<Vec<_>>(),
            vec!["rolling"],
            "only the well-formed window lands"
        );
        assert!(parse_usage_windows("not json").is_empty());
    }

    #[test]
    fn a_usage_report_persists_the_windows_and_the_worst_fraction() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        let body = serde_json::json!({
            "usage": {
                "rolling": {"percent": 40, "status": "ok", "resetsAt": "2026-10-25T20:00:00Z"},
                "weekly": {"percent": 88, "status": "ok", "resetsAt": "2026-11-01T00:00:00Z"},
                "monthly": {"percent": 5, "status": "ok", "resetsAt": "2026-11-20T00:00:00Z"}
            }
        })
        .to_string();
        note_usage_report(agent, "opencode-go", parse_usage_windows(&body));
        let state = load(agent);
        let account = &state.accounts["opencode-go"];
        assert_eq!(account.windows.len(), 3);
        assert_eq!(account.windows["weekly"].percent, 88.0);
        assert_eq!(account.used_pct, Some(0.88));
        assert!(account.used_pct_at_ms > 0);
        assert!(state.usage_fetched_at_ms > 0);
        // The map round-trips through the persisted file.
        let raw = std::fs::read_to_string(state_path(agent)).unwrap();
        assert!(raw.contains("\"weekly\""), "windows persist: {raw}");
        assert!(
            raw.contains("\"usageFetchedAtMs\""),
            "poll instant persists: {raw}"
        );
    }

    #[test]
    fn an_empty_report_leaves_the_state_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let agent = dir.path();
        note_usage(agent, "opencode-go", 0.4);
        note_usage_report(agent, "opencode-go", BTreeMap::new());
        let account = &load(agent).accounts["opencode-go"];
        assert_eq!(account.used_pct, Some(0.4));
        assert!(account.windows.is_empty());
    }

    #[test]
    fn state_round_trips_with_0600_permissions() {
        let dir = tempfile::tempdir().unwrap();
        note_served(dir.path(), "opencode-go", Some("org_42"));
        note_usage(dir.path(), "opencode-go", 0.42);
        let state = load(dir.path());
        assert_eq!(
            state.accounts["opencode-go"].org_id.as_deref(),
            Some("org_42")
        );
        assert_eq!(state.accounts["opencode-go"].used_pct, Some(0.42));
        let metadata = std::fs::metadata(state_path(dir.path())).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
        }
        #[cfg(not(unix))]
        assert!(metadata.is_file(), "the routing state file exists");
    }
}
