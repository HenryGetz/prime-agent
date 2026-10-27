//! The `opencode-go` live model catalog: fetch, strict parse, cache, and the
//! stale-while-revalidate refresh.
//!
//! The embedded roster in [`crate::opencode_go_models`] is the fallback: the
//! gateway's catalog at [`CATALOG_URL`] gains and retires entries without a
//! release, so the registry prefers the last good fetch and refreshes it in the
//! background. A refresh never blocks a request or a paint, runs at most once
//! per agent directory per process, and keeps the last good cache on failure.

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex, PoisonError};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use pa_types::ai::Model;

use crate::opencode_go_models::{build_model, GoTransport, OPENCODE_GO_PROVIDER_ID};
use crate::opencode_go_spec_generated as spec;

/// The `OpenCode` Go catalog the harness fetches (the `go` lane's entries).
pub const CATALOG_URL: &str = spec::CATALOG_URL;
/// The normalized cache under the agent directory.
pub const CATALOG_FILE: &str = "opencode-go-catalog.json";
/// A cache older than this is served while a detached refresh runs.
pub const CATALOG_TTL_MS: i64 = spec::CATALOG_TTL_MS;
/// The fetch bound; the refresh is background-only, so a slow host never
/// delays a turn.
pub const CATALOG_TIMEOUT_MS: u64 = spec::CATALOG_TIMEOUT_MS as u64;
/// The response cap (the live document is ~6 MiB).
pub const CATALOG_MAX_BYTES: u64 = spec::CATALOG_MAX_BYTES;
/// The harness fingerprint the catalog host expects.
pub const CATALOG_USER_AGENT: &str = spec::USER_AGENT;
/// The cache schema version; a different version is discarded.
pub const CACHE_VERSION: u32 = 2;

/// The persisted catalog: its schema version, when it was fetched, and the
/// normalized models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogCache {
    #[serde(default)]
    pub version: u32,
    pub fetched_at_ms: i64,
    pub models: Vec<Model>,
}

/// The cache path under `agent_dir`.
#[must_use]
pub fn cache_path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(CATALOG_FILE)
}

/// Load the last good cache. A missing or unreadable file is `None`; entries
/// are accepted only when they are this provider's and carry a nonempty id, and
/// a cache with no valid entry is treated as absent (the embedded roster
/// answers instead).
#[must_use]
pub fn load_cache(agent_dir: &Path) -> Option<CatalogCache> {
    let content = std::fs::read_to_string(cache_path(agent_dir)).ok()?;
    let cache: CatalogCache = serde_json::from_str(&content).ok()?;
    if cache.version != CACHE_VERSION {
        return None;
    }
    let models: Vec<Model> = cache
        .models
        .into_iter()
        .filter(|model| !model.id.trim().is_empty() && model.provider == OPENCODE_GO_PROVIDER_ID)
        .collect();
    if models.is_empty() {
        return None;
    }
    Some(CatalogCache {
        version: cache.version,
        fetched_at_ms: cache.fetched_at_ms,
        models,
    })
}

/// Persist the cache atomically (temp file then rename), mode 0600. A failed
/// write never fails the caller: the previous cache stays the fallback.
pub fn save_cache(agent_dir: &Path, models: &[Model]) {
    let cache = CatalogCache {
        version: CACHE_VERSION,
        fetched_at_ms: now_ms(),
        models: models.to_vec(),
    };
    let Ok(content) = serde_json::to_string_pretty(&cache) else {
        return;
    };
    let path = cache_path(agent_dir);
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
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let write = options
        .open(&temporary)
        .and_then(|mut file| file.write_all(format!("{content}\n").as_bytes()));
    if write.is_ok() {
        let _ = std::fs::rename(&temporary, &path);
    }
}

/// Whether the cache is old enough to refresh in the background.
#[must_use]
pub fn is_stale(cache: &CatalogCache, now: i64) -> bool {
    now.saturating_sub(cache.fetched_at_ms) >= CATALOG_TTL_MS
}

/// The provider's current models: the cache's entries minus learned-dead ids,
/// else the embedded roster minus learned-dead ids. A missing or stale cache
/// schedules a detached refresh and serves what is available now.
#[must_use]
pub fn models(agent_dir: &Path) -> Vec<Model> {
    let dead = super::opencode_go_dead_models::load(agent_dir);
    if let Some(cache) = load_cache(agent_dir) {
        if is_stale(&cache, now_ms()) {
            refresh_in_background(agent_dir);
        }
        let cached: Vec<Model> = cache
            .models
            .into_iter()
            .filter(|model| {
                !dead.contains(&model.id) && !spec::HIDDEN_MODEL_IDS.contains(&model.id.as_str())
            })
            .collect();
        if !cached.is_empty() {
            return cached;
        }
    } else {
        refresh_in_background(agent_dir);
    }
    crate::opencode_go_models::models()
        .into_iter()
        .filter(|model| {
            !dead.contains(&model.id) && !spec::HIDDEN_MODEL_IDS.contains(&model.id.as_str())
        })
        .collect()
}

/// Strictly parse the catalog document: the `opencode-go` entry's
/// non-deprecated models. A malformed document, an absent provider entry, or
/// an entry without a usable id/limit is skipped; a document with no usable
/// entries is rejected so the caller keeps its fallback.
#[must_use]
pub fn parse_catalog(body: &str) -> Option<Vec<Model>> {
    let root: serde_json::Value = serde_json::from_str(body).ok()?;
    let entry = root.get(spec::CATALOG_PROVIDER_ID)?;
    let models = entry.get("models")?.as_object()?;
    let parsed: Vec<Model> = models
        .iter()
        .filter_map(|(key, value)| model_from_entry(key, value))
        .collect();
    if parsed.is_empty() {
        None
    } else {
        Some(parsed)
    }
}

/// One catalog entry, converted with the embedded roster's rules: deprecated
/// entries are dropped, `provider.npm == "@ai-sdk/openai"` selects Responses,
/// the effort rungs are the Prime levels the entry names, and the base URL is
/// the Go gateway (the catalog's per-model `api` inference path is ignored).
fn model_from_entry(key: &str, value: &serde_json::Value) -> Option<Model> {
    if value.get("status").and_then(serde_json::Value::as_str) == Some("deprecated") {
        return None;
    }
    // The Go gateway currently rejects @ai-sdk/google models on every
    // advertised transport (chat returns "Model does not support this
    // protocol"; the Gemini route is unrouted). Keep them out of the roster.
    if value
        .get("provider")
        .and_then(|provider| provider.get("npm"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|npm| spec::HIDDEN_NPM.contains(&npm))
    {
        return None;
    }
    let id = value
        .get("id")
        .and_then(serde_json::Value::as_str)
        .filter(|id| !id.is_empty())
        .unwrap_or(key);
    if id.is_empty() || spec::HIDDEN_MODEL_IDS.contains(&id) {
        return None;
    }
    let limit = value.get("limit")?;
    let context_window = limit.get("context").and_then(serde_json::Value::as_u64)?;
    let max_tokens = limit.get("output").and_then(serde_json::Value::as_u64)?;
    if context_window == 0 || max_tokens == 0 {
        return None;
    }
    let cost = value.get("cost");
    let number = |field: &str| {
        cost.and_then(|cost| cost.get(field))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0)
    };
    let effort: Vec<String> = value
        .get("reasoning_options")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .find(|option| option.get("type").and_then(serde_json::Value::as_str) == Some("effort"))
        .and_then(|option| option.get("values"))
        .and_then(serde_json::Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let effort: Option<Vec<&str>> = (!effort.is_empty()).then(|| {
        effort
            .iter()
            .map(String::as_str)
            .filter(|value| is_prime_level(value))
            .collect()
    });
    let image = value
        .get("modalities")
        .and_then(|modalities| modalities.get("input"))
        .and_then(serde_json::Value::as_array)
        .is_some_and(|inputs| inputs.iter().any(|input| input.as_str() == Some("image")));
    let transport = GoTransport::from_npm(
        value
            .get("provider")
            .and_then(|provider| provider.get("npm"))
            .and_then(serde_json::Value::as_str),
    );
    let name = value
        .get("name")
        .and_then(serde_json::Value::as_str)
        .filter(|name| !name.is_empty())
        .unwrap_or(id);
    let reasoning = value
        .get("reasoning")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    Some(build_model(
        id,
        name,
        reasoning,
        context_window,
        max_tokens,
        (
            number("input"),
            number("output"),
            number("cache_read"),
            number("cache_write"),
        ),
        effort.as_deref(),
        transport,
        image,
    ))
}

/// The Prime thinking levels a catalog effort rung can name. The catalog's
/// `"none"` is the Off rung (see the embedded roster's map).
fn is_prime_level(value: &str) -> bool {
    matches!(
        value,
        "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
    )
}

/// The in-flight refreshes, keyed by agent directory (single-flight).
static REFRESHING: LazyLock<Mutex<HashSet<PathBuf>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

/// A background refresh is attempted at most this often per agent directory, so
/// a down catalog host cannot spawn a fetch thread on every registry load.
pub const REFRESH_RETRY_MS: i64 = spec::CATALOG_REFRESH_RETRY_MS;

/// The last refresh attempt per agent directory (process-wide backoff).
static LAST_ATTEMPT: LazyLock<Mutex<HashMap<PathBuf, i64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Whether a new attempt is allowed: never tried, or the recorded attempt is at
/// least [`REFRESH_RETRY_MS`] old (a future timestamp never allows one).
#[must_use]
pub fn should_attempt_refresh(last_attempt_ms: Option<i64>, now: i64) -> bool {
    match last_attempt_ms {
        Some(last) => now.saturating_sub(last) >= REFRESH_RETRY_MS,
        None => true,
    }
}

/// Run one detached refresh for `agent_dir` unless one is already in flight and
/// the last attempt is older than the retry window. The attempt is recorded
/// before the spawn, so a failing fetch backs off instead of re-spawning on
/// every load. The thread owns its runtime and never touches the caller's.
fn refresh_in_background(agent_dir: &Path) {
    let path = agent_dir.to_path_buf();
    {
        let mut refreshing = REFRESHING.lock().unwrap_or_else(PoisonError::into_inner);
        if !refreshing.insert(path.clone()) {
            return;
        }
    }
    let now = now_ms();
    let allowed = {
        let mut attempts = LAST_ATTEMPT.lock().unwrap_or_else(PoisonError::into_inner);
        let last = attempts.get(&path).copied();
        if should_attempt_refresh(last, now) {
            attempts.insert(path.clone(), now);
            true
        } else {
            false
        }
    };
    if !allowed {
        REFRESHING
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&path);
        return;
    }
    let cleanup = path.clone();
    let error_cleanup = path.clone();
    let spawned = std::thread::Builder::new()
        .name("opencode-go-catalog".to_string())
        .spawn(move || {
            if let Some(models) = fetch_blocking() {
                save_cache(&path, &models);
            } else {
                tracing::debug!("opencode-go catalog refresh failed; keeping the last good cache");
            }
            REFRESHING
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&cleanup);
        });
    if spawned.is_err() {
        REFRESHING
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&error_cleanup);
    }
}

/// Fetch and parse the catalog on a dedicated current-thread runtime.
fn fetch_blocking() -> Option<Vec<Model>> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;
    runtime.block_on(fetch())
}

async fn fetch() -> Option<Vec<Model>> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(CATALOG_TIMEOUT_MS))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .ok()?;
    let session = crate::oauth::harness_session_id(None);
    let mut request = client
        .get(CATALOG_URL)
        .header("user-agent", CATALOG_USER_AGENT)
        .header(spec::HEADER_SESSION, &session)
        .header(spec::HEADER_SESSION_ID, &session)
        .header(spec::HEADER_SESSION_AFFINITY, &session)
        .header(spec::HEADER_SESSION_ID_ALIAS, &session);
    for (name, value) in crate::oauth::trace_headers() {
        request = request.header(name, value);
    }
    let response = request.send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    if response
        .content_length()
        .is_some_and(|length| length > CATALOG_MAX_BYTES)
    {
        return None;
    }
    let bytes = response.bytes().await.ok()?;
    if bytes.len() as u64 > CATALOG_MAX_BYTES {
        return None;
    }
    parse_catalog(std::str::from_utf8(&bytes).ok()?)
}

fn now_ms() -> i64 {
    // Epoch milliseconds fit i64 for every representable wall-clock time.
    #[allow(clippy::cast_possible_truncation)]
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pa_types::ai::ModelInput;

    fn fixture() -> String {
        serde_json::json!({
            "opencode-go": {
                "id": "opencode-go",
                "models": {
                    "claude-haiku-5-5": {
                        "id": "claude-haiku-5-5",
                        "name": "Claude Haiku 5.5",
                        "reasoning": true,
                        "reasoning_options": [
                            {"type": "toggle"},
                            {"type": "effort", "values": ["low", "high", "max", "turbo"]}
                        ],
                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                        "limit": {"context": 1_000_000, "output": 128_000},
                        "cost": {"input": 0.1, "output": 0.5, "cache_read": 0.01, "cache_write": 0.125},
                        "provider": {"npm": "@ai-sdk/openai", "api": "https://opencode.ai/inference/go/openai/v1"}
                    },
                    "anthropic-chat": {
                        "id": "anthropic-chat",
                        "name": "Anthropic Chat",
                        "reasoning": false,
                        "limit": {"context": 200_000, "output": 8_192},
                        "cost": {"input": 0, "output": 0},
                        "provider": {"npm": "@ai-sdk/anthropic", "api": "https://opencode.ai/zen/go"}
                    },
                    "google-chat": {
                        "id": "google-chat",
                        "name": "Google Chat",
                        "reasoning": false,
                        "limit": {"context": 200_000, "output": 8_192},
                        "cost": {"input": 0, "output": 0},
                        "provider": {"npm": "@ai-sdk/google", "api": "https://opencode.ai/zen/go/v1"}
                    },
                    "gemini-4-argon": {
                        "id": "gemini-4-argon",
                        "name": "Gemini 4 Argon",
                        "reasoning": true,
                        "reasoning_options": [{"type": "effort", "values": ["low", "high"]}],
                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                        "limit": {"context": 1_000_000, "output": 64_000},
                        "cost": {"input": 0.5, "output": 2.0},
                        "provider": {"npm": "@ai-sdk/google", "api": "https://opencode.ai/zen/go/v1"}
                    },
                    "plain-chat": {
                        "id": "plain-chat",
                        "name": "Plain Chat",
                        "limit": {"context": 128_000, "output": 8_192},
                        "cost": {"input": 0, "output": 0},
                        "provider": {"api": "https://opencode.ai/inference/go/openai/v1"}
                    },
                    "retired": {
                        "id": "retired",
                        "status": "deprecated",
                        "limit": {"context": 1, "output": 1}
                    },
                    "broken": {"name": "no id or limit"}
                }
            },
            "another-provider": {"models": {}}
        })
        .to_string()
    }

    #[test]
    fn parsing_keeps_live_entries_and_converts_the_transport() {
        let models = parse_catalog(&fixture()).expect("the fixture parses");
        assert_eq!(
            models.len(),
            3,
            "the deprecated, malformed, and @ai-sdk/google entries are dropped"
        );
        let haiku = models
            .iter()
            .find(|model| model.id == "claude-haiku-5-5")
            .unwrap();
        assert_eq!(haiku.provider, OPENCODE_GO_PROVIDER_ID);
        assert_eq!(haiku.base_url, "https://opencode.ai/zen/go/v1");
        assert_eq!(
            haiku.api, "openai-responses",
            "only @ai-sdk/openai is Responses"
        );
        assert_eq!(haiku.input, vec![ModelInput::Text, ModelInput::Image]);
        assert_eq!(haiku.context_window, 1_000_000);
        assert_eq!(haiku.max_tokens, 128_000);
        let map = haiku.thinking_level_map.as_ref().unwrap();
        assert!(map[&pa_types::ai::ModelThinkingLevel::Max].is_some());
        assert!(
            !map.values().any(|value| value.as_deref() == Some("turbo")),
            "unknown effort rungs are intersected out"
        );
        let plain = models
            .iter()
            .find(|model| model.id == "plain-chat")
            .unwrap();
        assert_eq!(
            plain.api, "openai-completions",
            "a missing npm is chat completions"
        );
        assert!(plain
            .thinking_level_map
            .as_ref()
            .unwrap()
            .values()
            .all(Option::is_none));
        let anthropic = models
            .iter()
            .find(|model| model.id == "anthropic-chat")
            .unwrap();
        assert_eq!(anthropic.api, "anthropic-messages");
        assert_eq!(anthropic.base_url, "https://opencode.ai/zen/go");
        assert!(
            !models
                .iter()
                .any(|model| model.id == "google-chat" || model.id == "gemini-4-argon"),
            "@ai-sdk/google entries are hidden"
        );
    }

    #[test]
    fn malformed_or_absent_provider_documents_are_rejected() {
        assert_eq!(parse_catalog("not json"), None);
        assert_eq!(parse_catalog("{}"), None);
        assert_eq!(parse_catalog(r#"{"opencode-go": {"models": {}}}"#), None);
        assert_eq!(
            parse_catalog(r#"{"other": {"models": {"x": {"id": "x"}}}}"#),
            None
        );
    }

    #[test]
    fn cache_round_trips_with_0600_and_stale_predicate() {
        let dir = tempfile::tempdir().unwrap();
        let models = parse_catalog(&fixture()).unwrap();
        save_cache(dir.path(), &models);
        let cache = load_cache(dir.path()).expect("the cache round-trips");
        assert_eq!(cache.models, models);
        assert!(!is_stale(&cache, cache.fetched_at_ms));
        assert!(is_stale(&cache, cache.fetched_at_ms + CATALOG_TTL_MS));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(cache_path(dir.path()))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn models_serve_the_cache_and_drop_learned_dead_ids() {
        let dir = tempfile::tempdir().unwrap();
        let cached = parse_catalog(&fixture()).unwrap();
        save_cache(dir.path(), &cached);
        super::super::opencode_go_dead_models::record(dir.path(), "plain-chat");
        // The cache is fresh, so `models` serves it without a background fetch.
        let served = models(dir.path());
        assert_eq!(
            served.len(),
            2,
            "the learned-dead plain-chat id is dropped from the three live entries"
        );
        assert!(
            !served.iter().any(|model| model.id == "plain-chat"),
            "the learned-dead id is hidden"
        );
        assert!(
            served.iter().any(|model| model.id == "claude-haiku-5-5"),
            "live entries still serve"
        );
    }

    #[test]
    fn an_empty_or_malformed_cache_falls_back_to_the_embedded_roster() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(models_without_refresh(dir.path()).len(), 31);
        std::fs::write(cache_path(dir.path()), "").unwrap();
        assert_eq!(models_without_refresh(dir.path()).len(), 31);
    }

    /// The cache/roster resolution with the background refresh elided, so a
    /// temp dir never reaches the network.
    fn models_without_refresh(agent_dir: &Path) -> Vec<Model> {
        let dead = super::super::opencode_go_dead_models::load(agent_dir);
        load_cache(agent_dir)
            .map_or_else(crate::opencode_go_models::models, |cache| cache.models)
            .into_iter()
            .filter(|model| {
                !dead.contains(&model.id) && !spec::HIDDEN_MODEL_IDS.contains(&model.id.as_str())
            })
            .collect()
    }

    #[test]
    fn the_refresh_backoff_allows_one_attempt_per_window() {
        let now = 1_000_000;
        assert!(
            should_attempt_refresh(None, now),
            "the first attempt is always allowed"
        );
        assert!(!should_attempt_refresh(Some(now), now));
        assert!(!should_attempt_refresh(
            Some(now - REFRESH_RETRY_MS + 1),
            now
        ));
        assert!(
            should_attempt_refresh(Some(now - REFRESH_RETRY_MS), now),
            "the window elapsed"
        );
        assert!(
            !should_attempt_refresh(Some(now + 5_000), now),
            "a future attempt stamp never re-opens the window"
        );
    }

    #[test]
    fn a_tampered_cache_drops_foreign_and_empty_id_entries() {
        let dir = tempfile::tempdir().unwrap();
        let models = parse_catalog(&fixture()).unwrap();
        let mut kept = models[0].clone();
        kept.id = "kept".to_string();
        let mut foreign = models[0].clone();
        foreign.id = "foreign".to_string();
        foreign.provider = "another-provider".to_string();
        let mut empty = models[0].clone();
        empty.id = "  ".to_string();
        std::fs::write(
            cache_path(dir.path()),
            serde_json::json!({
                "version": CACHE_VERSION,
                "fetchedAtMs": now_ms(),
                "models": [kept, foreign, empty]
            })
            .to_string(),
        )
        .unwrap();
        let loaded = load_cache(dir.path()).expect("the valid entry survives");
        assert_eq!(
            loaded
                .models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            vec!["kept"]
        );
    }

    #[test]
    fn a_cache_with_no_valid_entry_falls_back_to_the_embedded_roster() {
        let dir = tempfile::tempdir().unwrap();
        let mut foreign = parse_catalog(&fixture()).unwrap()[0].clone();
        foreign.provider = "another-provider".to_string();
        std::fs::write(
            cache_path(dir.path()),
            serde_json::json!({
                "version": CACHE_VERSION,
                "fetchedAtMs": now_ms(),
                "models": [foreign]
            })
            .to_string(),
        )
        .unwrap();
        assert!(
            load_cache(dir.path()).is_none(),
            "a cache with no valid entry is absent"
        );
        assert_eq!(models_without_refresh(dir.path()).len(), 31);
    }
}
