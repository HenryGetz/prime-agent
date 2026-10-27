//! The `OpenCode` Go (Console) OAuth flow: the device-code login against
//! `https://opencode.ai/console`, the identity/org lookup, and the refresh.
//!
//! The wire contract is the harness's own (`opencode` v2's
//! `packages/core/src/plugin/provider/opencode.ts`, captured in
//! `capture/golden-requests.ndjson`): JSON bodies, the harness User-Agent and
//! trace headers, poll bodies classified regardless of HTTP status, and the
//! `slow_down` interval bump. Refresh tokens rotate and are single-use.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use rand::Rng;
use url::Url;

use super::provider_http::{ProviderHttp, ProviderHttpMethod, ProviderHttpRequest};
use super::types::OAuthLoginUi;
use crate::opencode_go_spec_generated as spec;

/// The generated harness ids, stable per process input.
static HARNESS_IDS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<Option<String>, String>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// The descending-id clock and counter.
static DESCENDING_CLOCK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DESCENDING_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The base62 alphabet of the harness session-id suffix.
const BASE62: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// The device-flow client the Console registers.
pub const OPENCODE_CLIENT_ID: &str = spec::CLIENT_ID;
/// The production Console base (the login and identity endpoints).
pub const OPENCODE_CONSOLE_BASE: &str = spec::CONSOLE_BASE_URL;
/// The Go gateway base the subscription serves inference from.
pub const OPENCODE_GO_BASE_URL: &str = spec::GATEWAY_OPENAI_BASE_URL;
/// The Go gateway base the Anthropic Messages lane serves from.
pub const OPENCODE_GO_ANTHROPIC_BASE_URL: &str = spec::GATEWAY_ANTHROPIC_BASE_URL;
/// The harness User-Agent the Console and gateway fingerprint.
pub const OPENCODE_USER_AGENT: &str = spec::USER_AGENT;
/// The Console request timeout.
pub const REQUEST_TIMEOUT_MS: u64 = spec::OAUTH_REQUEST_TIMEOUT_MS as u64;
/// The refresh runs under the auth store's file lock, which a peer declares stale after 10 seconds;
/// the request must fit inside that window.
pub const REFRESH_TIMEOUT_MS: u64 = spec::OAUTH_REFRESH_TIMEOUT_MS as u64;
/// The harness's `slow_down` poll-interval bump.
const SLOW_DOWN_MS: u64 = spec::OAUTH_SLOW_DOWN_MS as u64;
/// The cancel error the driving surface maps to the silent cancelled outcome.
pub const LOGIN_CANCELLED: &str = "Login cancelled";
/// The sleep granularity that makes the device flow check its cancel flag.
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(250);

fn device_code_url(base: &str) -> String {
    format!("{base}{}", spec::DEVICE_CODE_PATH)
}

fn device_token_url(base: &str) -> String {
    format!("{base}{}", spec::DEVICE_TOKEN_PATH)
}

fn user_url(base: &str) -> String {
    format!("{base}{}", spec::USER_PATH)
}

fn orgs_url(base: &str) -> String {
    format!("{base}{}", spec::ORGS_PATH)
}

/// The device-authorization body the Console registers the client with.
#[must_use]
pub fn device_code_body() -> String {
    format!(
        "{{\"client_id\":\"{}\",\"supports_org_scope\":{}}}",
        spec::CLIENT_ID,
        spec::SUPPORTS_ORG_SCOPE
    )
}

/// The device-token poll body for `device_code`.
#[must_use]
pub fn device_token_body(device_code: &str) -> String {
    format!(
        "{{\"grant_type\":\"{}\",\"device_code\":{},\"client_id\":\"{}\"}}",
        spec::DEVICE_GRANT_TYPE,
        json_string(device_code),
        spec::CLIENT_ID
    )
}

/// The refresh body for `refresh_token`.
#[must_use]
pub fn refresh_token_body(refresh_token: &str) -> String {
    format!(
        "{{\"grant_type\":\"{}\",\"refresh_token\":{},\"client_id\":\"{}\"}}",
        spec::REFRESH_GRANT_TYPE,
        json_string(refresh_token),
        spec::CLIENT_ID
    )
}

/// One Console login: the tokens, the identity that owns them, and the
/// workspace the login selected. The expired-at value is wall-clock epoch
/// milliseconds, as the auth store stores it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpencodeCredentials {
    pub access: String,
    pub refresh: String,
    pub expires: i64,
    pub account_id: Option<String>,
    pub email: Option<String>,
    pub org_id: Option<String>,
    pub org_name: Option<String>,
    /// The Console base the login ran against (the refresh reuses it).
    pub api_endpoint: String,
}

/// The status plus the parsed body (JSON `null` for a non-JSON body).
struct OpencodeResponse {
    ok: bool,
    status: u16,
    body: serde_json::Value,
}

impl OpencodeResponse {
    fn object(&self) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.body.as_object()
    }
}

/// One logical operation's trace: the b3/traceparent ids the harness groups
/// its requests under. Every request carries its own span.
#[derive(Debug, Clone)]
struct Trace {
    trace: String,
    parent: String,
}

impl Trace {
    fn new() -> Self {
        Trace {
            trace: random_hex(16),
            parent: random_hex(8),
        }
    }

    fn headers(&self) -> Vec<(String, String)> {
        let span = random_hex(8);
        vec![
            (
                "b3".to_string(),
                format!("{}-{span}-1-{}", self.trace, self.parent),
            ),
            (
                "traceparent".to_string(),
                format!("00-{}-{span}-01", self.trace),
            ),
        ]
    }
}

fn random_hex(bytes: usize) -> String {
    use std::fmt::Write as _;
    let mut buffer = vec![0u8; bytes];
    rand::thread_rng().fill(&mut buffer[..]);
    let mut hex = String::with_capacity(bytes * 2);
    for byte in buffer {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// The Console header set: accept (and content-type on JSON bodies), the
/// optional bearer and org, the harness User-Agent, and the trace pair.
fn console_headers(
    token: Option<&str>,
    org_id: Option<&str>,
    json: bool,
    trace: &Trace,
) -> Vec<(String, String)> {
    let mut headers = vec![("accept".to_string(), "application/json".to_string())];
    if json {
        headers.push(("content-type".to_string(), "application/json".to_string()));
    }
    if let Some(token) = token {
        headers.push(("authorization".to_string(), format!("Bearer {token}")));
    }
    if let Some(org_id) = org_id {
        headers.push(("x-org-id".to_string(), org_id.to_string()));
    }
    headers.push(("user-agent".to_string(), OPENCODE_USER_AGENT.to_string()));
    headers.extend(trace.headers());
    headers
}

fn json_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

/// The device authorization, the user code, and the token poll.
///
/// # Errors
///
/// Returns an error when the login is cancelled ([`LOGIN_CANCELLED`]), the device
/// authorization fails, the code expires, or the token response fails validation.
pub async fn login_opencode(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
) -> Result<OpencodeCredentials, String> {
    login_opencode_at(http, ui, OPENCODE_CONSOLE_BASE).await
}

/// [`login_opencode`] against a fixed Console base (tests and a moved Console).
///
/// # Errors
///
/// Returns the same errors as [`login_opencode`].
#[allow(clippy::too_many_lines)] // the device protocol is one linear exchange
pub async fn login_opencode_at(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    base: &str,
) -> Result<OpencodeCredentials, String> {
    let base = base.trim_end_matches('/');
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    // One trace id across the whole login, as the harness groups it.
    let trace = Trace::new();
    let response = send(
        http,
        ui,
        ProviderHttpRequest {
            method: ProviderHttpMethod::Post,
            url: device_code_url(base),
            headers: console_headers(None, None, true, &trace),
            body: Some(device_code_body()),
            follow_redirects: false,
        },
        REQUEST_TIMEOUT_MS,
    )
    .await?;
    if !response.ok {
        return Err(format!(
            "Device authorization failed: HTTP {}",
            response.status
        ));
    }
    let Some(body) = response.object() else {
        return Err("Device authorization response is missing required fields".to_string());
    };
    let required = |field: &str| -> Result<String, String> {
        body.get(field)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .ok_or_else(|| "Device authorization response is missing required fields".to_string())
    };
    let device_code = required("device_code")?;
    let user_code = required("user_code")?;
    let verification_raw = required("verification_uri_complete")?;
    let expires_in = body
        .get("expires_in")
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| "Device authorization response is missing required fields".to_string())?;
    let interval_seconds = body
        .get("interval")
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| "Device authorization response is missing required fields".to_string())?;
    let verification = verification_url(base, &verification_raw)?;
    // The wire's seconds are positive f64 counts; u64 ms is the poll's unit.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let lifetime_ms = (expires_in * 1000.0) as u64;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let mut interval_ms = (interval_seconds * 1000.0).max(0.0) as u64;
    ui.on_auth(&verification, Some(&format!("Enter code: {user_code}")));

    let deadline = std::time::Instant::now() + Duration::from_millis(lifetime_ms);
    // The harness waits `interval` seconds before the first poll.
    cancel_aware_sleep(ui, Duration::from_millis(interval_ms)).await?;
    loop {
        if std::time::Instant::now() >= deadline {
            return Err("OpenCode device code expired; sign in again".to_string());
        }
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        #[allow(clippy::cast_possible_truncation)]
        let token = send(
            http,
            ui,
            ProviderHttpRequest {
                method: ProviderHttpMethod::Post,
                url: device_token_url(base),
                headers: console_headers(None, None, true, &trace),
                body: Some(device_token_body(&device_code)),
                follow_redirects: false,
            },
            remaining.as_millis() as u64,
        )
        .await?;
        // Poll bodies classify regardless of status, matching the harness.
        let parsed = token.object();
        let token_field = |field: &str| {
            parsed
                .and_then(|body| body.get(field))
                .and_then(serde_json::Value::as_str)
        };
        if let Some(access) = token_field("access_token") {
            let refresh = token_field("refresh_token").ok_or_else(|| {
                "Invalid OpenCode OAuth response field: refresh_token".to_string()
            })?;
            let lifetime = parsed
                .and_then(|body| body.get("expires_in"))
                .and_then(serde_json::Value::as_f64)
                .filter(|value| value.is_finite() && *value > 0.0)
                .ok_or_else(|| "Invalid OpenCode OAuth response field: expires_in".to_string())?;
            let access = access.to_string();
            let refresh = refresh.to_string();
            let token_org = token_field("org_id").map(str::to_string);
            return identity(
                http, ui, base, &access, refresh, lifetime, token_org, &trace,
            )
            .await;
        }
        match parsed
            .and_then(|body| body.get("error"))
            .and_then(serde_json::Value::as_str)
        {
            Some("authorization_pending") => {}
            Some("slow_down") => {
                // TS bumps the interval by five seconds per slow_down answer.
                interval_ms = interval_ms.saturating_add(SLOW_DOWN_MS);
            }
            Some("access_denied" | "authorization_denied") => {
                return Err("OpenCode device authorization was denied".to_string());
            }
            Some("expired_token") => {
                return Err("OpenCode device code expired; sign in again".to_string());
            }
            Some(error) => {
                return Err(format!("Device authorization failed: {error}"));
            }
            None => {
                return Err(format!(
                    "Device authorization failed: HTTP {}",
                    token.status
                ));
            }
        }
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            return Err("OpenCode device code expired; sign in again".to_string());
        }
        // A zero interval is the harness's own fast-poll test shape; the
        // cancel poll keeps the loop responsive without busy-waiting.
        cancel_aware_sleep(
            ui,
            Duration::from_millis(interval_ms).min(remaining.max(CANCEL_POLL_INTERVAL)),
        )
        .await?;
    }
}

/// The identity lookup: the user and the orgs, in parallel with the fresh
/// token, then the harness org selection.
#[allow(clippy::too_many_arguments)] // the harness identity inputs
async fn identity(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    base: &str,
    access: &str,
    refresh: String,
    lifetime_seconds: f64,
    token_org: Option<String>,
    trace: &Trace,
) -> Result<OpencodeCredentials, String> {
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    let get = |url: String| ProviderHttpRequest {
        method: ProviderHttpMethod::Get,
        url,
        headers: console_headers(Some(access), None, false, trace),
        body: None,
        follow_redirects: false,
    };
    let user_request = get(user_url(base));
    let orgs_request = get(orgs_url(base));
    let (user, orgs) = tokio::join!(
        http.request(user_request, REQUEST_TIMEOUT_MS),
        http.request(orgs_request, REQUEST_TIMEOUT_MS),
    );
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    let user = parse(user, "account lookup")?;
    let orgs = parse(orgs, "account lookup")?;
    if !user.ok || !orgs.ok {
        return Err("Account lookup failed: HTTP".to_string());
    }
    let user = user.object().cloned().unwrap_or_default();
    let account_id = user
        .get("id")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let email = user
        .get("email")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let mut available: Vec<(String, String)> = orgs
        .body
        .as_array()
        .map(|orgs| {
            orgs.iter()
                .filter_map(|org| {
                    let id = org.get("id").and_then(serde_json::Value::as_str)?;
                    let name = org
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or(id);
                    Some((id.to_string(), name.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();
    // Harness org selection: name then id, ascending.
    available.sort();
    let (org_id, org_name) = match token_org {
        Some(token_org) => {
            let found = available.iter().find(|(id, _)| *id == token_org);
            let Some((id, name)) = found else {
                return Err(format!("OpenCode organization not found: {token_org}"));
            };
            (Some(id.clone()), Some(name.clone()))
        }
        None => match available.first() {
            Some((id, name)) => (Some(id.clone()), Some(name.clone())),
            None => (None, None),
        },
    };
    // The TS expiry is the plain lifetime (the auth store applies its own
    // refresh skew); the wire's seconds fit i64 after the millisecond scale.
    #[allow(clippy::cast_possible_truncation)]
    let expires = now_ms() + (lifetime_seconds * 1000.0) as i64;
    Ok(OpencodeCredentials {
        access: access.to_string(),
        refresh,
        expires,
        account_id,
        email,
        org_id,
        org_name,
        api_endpoint: base.to_string(),
    })
}

/// Refresh an expired credential. The Console rotates the pair; the caller
/// persists the new refresh token before any other request.
///
/// # Errors
///
/// Returns an error when the session is dead (`invalid_grant`) or the refresh
/// response fails validation.
pub async fn refresh_opencode_token(
    http: &dyn ProviderHttp,
    refresh_token: &str,
    api_endpoint: Option<&str>,
) -> Result<OpencodeCredentials, String> {
    let base = api_endpoint
        .unwrap_or(OPENCODE_CONSOLE_BASE)
        .trim_end_matches('/');
    let trace = Trace::new();
    let response = send(
        http,
        &NoCancel,
        ProviderHttpRequest {
            method: ProviderHttpMethod::Post,
            url: device_token_url(base),
            headers: console_headers(None, None, true, &trace),
            body: Some(refresh_token_body(refresh_token)),
            follow_redirects: false,
        },
        REFRESH_TIMEOUT_MS,
    )
    .await?;
    if response.status == 400
        && response
            .object()
            .and_then(|body| body.get("error"))
            .and_then(serde_json::Value::as_str)
            == Some("invalid_grant")
    {
        return Err("OpenCode Console session expired; log in again".to_string());
    }
    if !response.ok {
        return Err(format!("Token refresh failed: HTTP {}", response.status));
    }
    let Some(body) = response.object() else {
        return Err("Invalid OpenCode OAuth response field: access_token".to_string());
    };
    let access = body
        .get("access_token")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Invalid OpenCode OAuth response field: access_token".to_string())?;
    let refresh = body
        .get("refresh_token")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or(refresh_token);
    let lifetime = body
        .get("expires_in")
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| "Invalid OpenCode OAuth response field: expires_in".to_string())?;
    #[allow(clippy::cast_possible_truncation)]
    let expires = now_ms() + (lifetime * 1000.0) as i64;
    Ok(OpencodeCredentials {
        access: access.to_string(),
        refresh: refresh.to_string(),
        expires,
        account_id: None,
        email: None,
        org_id: body
            .get("org_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
        org_name: None,
        api_endpoint: base.to_string(),
    })
}

/// The refresh runs detached from any interactive surface.
struct NoCancel;

impl OAuthLoginUi for NoCancel {
    fn on_auth(&self, _url: &str, _instructions: Option<&str>) {}
    fn on_prompt(
        &self,
        _prompt: &super::types::OAuthPrompt,
    ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + '_>> {
        Box::pin(std::future::pending())
    }
    fn on_progress(&self, _message: &str) {}
    fn on_manual_code_input(
        &self,
    ) -> Option<Pin<Box<dyn Future<Output = Option<String>> + Send + '_>>> {
        None
    }
}

async fn send(
    http: &dyn ProviderHttp,
    ui: &dyn OAuthLoginUi,
    request: ProviderHttpRequest,
    timeout_ms: u64,
) -> Result<OpencodeResponse, String> {
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    let response = http
        .request(request, timeout_ms.min(REQUEST_TIMEOUT_MS))
        .await
        .map_err(|message| {
            if message.contains("timed out") {
                "OpenCode OAuth request timed out. Try signing in again.".to_string()
            } else {
                "OpenCode OAuth request failed. Check your connection and try again.".to_string()
            }
        })?;
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    let parsed = serde_json::from_str(&response.body).map_err(|_| {
        format!(
            "OpenCode OAuth returned invalid JSON (HTTP {})",
            response.status
        )
    })?;
    Ok(OpencodeResponse {
        ok: response.ok(),
        status: response.status,
        body: parsed,
    })
}

fn parse(
    response: Result<super::provider_http::ProviderHttpResponse, String>,
    action: &str,
) -> Result<OpencodeResponse, String> {
    let response = response?;
    let parsed = serde_json::from_str(&response.body).map_err(|_| {
        format!(
            "OpenCode OAuth {action} returned invalid JSON (HTTP {})",
            response.status
        )
    })?;
    Ok(OpencodeResponse {
        ok: response.ok(),
        status: response.status,
        body: parsed,
    })
}

/// The harness rejects non-HTTP(S) verification URLs instead of opening them.
/// A relative value resolves against the Console base, as the harness does.
fn verification_url(base: &str, raw: &str) -> Result<String, String> {
    let base = Url::parse(&format!("{base}/"))
        .map_err(|_| format!("Invalid device verification URL: {raw}"))?;
    let resolved = base
        .join(raw)
        .map_err(|_| format!("Invalid device verification URL: {raw}"))?;
    if !matches!(resolved.scheme(), "http" | "https") {
        return Err(format!("Invalid device verification URL: {resolved}"));
    }
    Ok(resolved.to_string())
}

async fn cancel_aware_sleep(ui: &dyn OAuthLoginUi, total: Duration) -> Result<(), String> {
    let start = std::time::Instant::now();
    while start.elapsed() < total {
        if ui.is_cancelled() {
            return Err(LOGIN_CANCELLED.to_string());
        }
        let remaining = total.saturating_sub(start.elapsed());
        tokio::time::sleep(remaining.min(CANCEL_POLL_INTERVAL)).await;
    }
    if ui.is_cancelled() {
        return Err(LOGIN_CANCELLED.to_string());
    }
    Ok(())
}

fn now_ms() -> i64 {
    // Epoch milliseconds fit i64 for every representable wall-clock time.
    #[allow(clippy::cast_possible_truncation)]
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as i64)
}

/// The trace pair for one request (a fresh trace and span each call).
#[must_use]
pub fn trace_headers() -> Vec<(String, String)> {
    Trace::new().headers()
}

/// Whether `id` is the harness's own session shape (12 lowercase hex time
/// chars + 14 base62), with or without the `ses_` prefix.
#[must_use]
pub fn is_harness_session_id(id: &str) -> bool {
    let id = id.strip_prefix("ses_").unwrap_or(id);
    if id.len() != 26 {
        return false;
    }
    let (time, suffix) = id.split_at(12);
    time.bytes()
        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        && suffix.bytes().all(|byte| {
            byte.is_ascii_digit() || byte.is_ascii_uppercase() || byte.is_ascii_lowercase()
        })
}

/// A harness-shaped `ses_…` identifier for a session: an already harness-shaped
/// id is kept, anything else (a Prime session id) maps to a deterministic id
/// derived from the seed, so a restarted worker keeps sending the same
/// `x-opencode-session` for the same conversation. A missing seed maps to one
/// generated id that stays stable for the life of the process.
#[must_use]
pub fn harness_session_id(seed: Option<&str>) -> String {
    if let Some(seed) = seed {
        if is_harness_session_id(seed) {
            return if seed.starts_with("ses_") {
                seed.to_string()
            } else {
                format!("ses_{seed}")
            };
        }
        return format!("ses_{}", deterministic_id(seed));
    }
    let mut ids = HARNESS_IDS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    ids.entry(None)
        .or_insert_with(|| format!("ses_{}", descending_id()))
        .clone()
}

/// The fixed domain prefix that separates the harness-session derivation from
/// every other use of the seed.
const HARNESS_ID_DOMAIN: &[u8] = b"prime-agent:opencode-harness-session:";

/// The seed's stable harness-shaped id: the SHA-256 of the domain prefix and the
/// seed, as 12 lowercase hex chars (the time half) followed by 14 base62 chars.
fn deterministic_id(seed: &str) -> String {
    use sha2::{Digest as _, Sha256};
    use std::fmt::Write as _;
    let digest = Sha256::new()
        .chain_update(HARNESS_ID_DOMAIN)
        .chain_update(seed.as_bytes())
        .finalize();
    let mut id = String::with_capacity(26);
    for byte in &digest[..6] {
        let _ = write!(id, "{byte:02x}");
    }
    for byte in &digest[6..20] {
        id.push(BASE62[usize::from(*byte) % BASE62.len()] as char);
    }
    id
}

/// The harness's descending time encoding: `~(ms * 0x1000 + counter)` as 12
/// lowercase hex chars, then 14 base62 chars.
fn descending_id() -> String {
    use std::fmt::Write as _;
    use std::sync::atomic::Ordering;
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let timestamp = now_ms().max(0) as u64;
    if DESCENDING_CLOCK.swap(timestamp, Ordering::Relaxed) != timestamp {
        DESCENDING_COUNTER.store(0, Ordering::Relaxed);
    }
    let counter = DESCENDING_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
    let value = !(timestamp.wrapping_mul(0x1000).wrapping_add(counter));
    let mut id = String::with_capacity(26);
    for index in 0..6 {
        #[allow(clippy::cast_possible_truncation)]
        let byte = ((value >> (40 - 8 * index)) & 0xff) as u8;
        let _ = write!(id, "{byte:02x}");
    }
    let mut random = vec![0u8; 14];
    rand::thread_rng().fill(&mut random[..]);
    for byte in random {
        id.push(BASE62[usize::from(byte) % BASE62.len()] as char);
    }
    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, VecDeque};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct ScriptedResponse {
        status: u16,
        body: String,
    }

    #[derive(Default)]
    struct ScriptedHttp {
        queued: Mutex<HashMap<String, VecDeque<ScriptedResponse>>>,
        requests: Mutex<Vec<ProviderHttpRequest>>,
    }

    impl ScriptedHttp {
        fn new() -> Self {
            ScriptedHttp::default()
        }

        fn entry(status: u16, body: &str) -> ScriptedResponse {
            ScriptedResponse {
                status,
                body: body.to_string(),
            }
        }

        fn queue(self, url: &str, responses: Vec<ScriptedResponse>) -> Self {
            self.queued.lock().unwrap().insert(
                url.to_string(),
                responses.into_iter().collect::<VecDeque<_>>(),
            );
            self
        }

        fn requests_for(&self, url: &str) -> Vec<ProviderHttpRequest> {
            self.requests
                .lock()
                .unwrap()
                .iter()
                .filter(|request| request.url == url)
                .cloned()
                .collect()
        }
    }

    impl ProviderHttp for ScriptedHttp {
        fn request(
            &self,
            request: ProviderHttpRequest,
            _timeout_ms: u64,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<super::super::provider_http::ProviderHttpResponse, String>,
                    > + Send
                    + '_,
            >,
        > {
            self.requests.lock().unwrap().push(request.clone());
            let response = self
                .queued
                .lock()
                .unwrap()
                .get_mut(&request.url)
                .and_then(std::collections::VecDeque::pop_front);
            Box::pin(async move {
                let response =
                    response.ok_or_else(|| format!("{} was not scripted", request.url))?;
                Ok(super::super::provider_http::ProviderHttpResponse {
                    status: response.status,
                    body: response.body,
                })
            })
        }
    }

    struct ScriptedUi {
        auth_url: Mutex<Option<String>>,
        auth_instructions: Mutex<Option<String>>,
        cancelled: Arc<AtomicBool>,
    }

    impl ScriptedUi {
        fn new() -> Self {
            ScriptedUi {
                auth_url: Mutex::new(None),
                auth_instructions: Mutex::new(None),
                cancelled: Arc::new(AtomicBool::new(false)),
            }
        }

        fn captured_auth(&self) -> (String, Option<String>) {
            (
                self.auth_url.lock().unwrap().clone().expect("the auth url"),
                self.auth_instructions.lock().unwrap().clone(),
            )
        }
    }

    impl OAuthLoginUi for ScriptedUi {
        fn on_auth(&self, url: &str, instructions: Option<&str>) {
            *self.auth_url.lock().unwrap() = Some(url.to_string());
            *self.auth_instructions.lock().unwrap() = instructions.map(str::to_string);
        }

        fn on_prompt(
            &self,
            _prompt: &super::super::types::OAuthPrompt,
        ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + '_>> {
            Box::pin(std::future::pending())
        }

        fn on_progress(&self, _message: &str) {}

        fn on_manual_code_input(
            &self,
        ) -> Option<Pin<Box<dyn Future<Output = Option<String>> + Send + '_>>> {
            None
        }

        fn is_cancelled(&self) -> bool {
            self.cancelled.load(Ordering::Relaxed)
        }
    }

    fn device_response(org_field: &str) -> ScriptedResponse {
        ScriptedResponse {
            status: 200,
            body: format!(
                r#"{{"device_code":"dev-1","user_code":"ABCD-1234","verification_uri_complete":"/auth/device?code=ABCD-1234","interval":0,"expires_in":900{org_field}}}"#
            ),
        }
    }

    fn token_response(access: &str, refresh: &str) -> ScriptedResponse {
        ScriptedResponse {
            status: 200,
            body: format!(
                r#"{{"access_token":"{access}","refresh_token":"{refresh}","expires_in":3600}}"#
            ),
        }
    }

    fn happy_http() -> ScriptedHttp {
        ScriptedHttp::new()
            .queue(
                &device_code_url(OPENCODE_CONSOLE_BASE),
                vec![device_response("")],
            )
            .queue(
                &device_token_url(OPENCODE_CONSOLE_BASE),
                vec![token_response("oc-access", "oc-refresh")],
            )
            .queue(
                &user_url(OPENCODE_CONSOLE_BASE),
                vec![ScriptedHttp::entry(
                    200,
                    r#"{"id":"usr_1","email":"dev@example.com"}"#,
                )],
            )
            .queue(
                &orgs_url(OPENCODE_CONSOLE_BASE),
                vec![ScriptedHttp::entry(
                    200,
                    r#"[{"id":"org_b","name":"beta"},{"id":"org_a","name":"alpha"}]"#,
                )],
            )
    }

    #[tokio::test]
    async fn the_happy_flow_resolves_the_credentials_and_wire_shape() {
        let http = happy_http();
        let ui = ScriptedUi::new();
        let credentials = login_opencode(&http, &ui).await.unwrap();
        assert_eq!(credentials.access, "oc-access");
        assert_eq!(credentials.refresh, "oc-refresh");
        assert_eq!(credentials.account_id.as_deref(), Some("usr_1"));
        assert_eq!(credentials.email.as_deref(), Some("dev@example.com"));
        // The default org is the first sorted by name then id.
        assert_eq!(credentials.org_id.as_deref(), Some("org_a"));
        assert_eq!(credentials.api_endpoint, OPENCODE_CONSOLE_BASE.to_string());
        let now = now_ms();
        let skew = (credentials.expires - now - 3_600_000).abs();
        assert!(skew < 10_000, "the plain lifetime: {skew}");

        let (url, instructions) = ui.captured_auth();
        // A root-relative verification path resolves at the Console origin,
        // exactly as the harness's `new URL(path, base + "/")` does.
        assert_eq!(url, "https://opencode.ai/auth/device?code=ABCD-1234");
        assert_eq!(instructions.as_deref(), Some("Enter code: ABCD-1234"));

        let code = &http.requests_for(&device_code_url(OPENCODE_CONSOLE_BASE))[0];
        assert_eq!(code.method, ProviderHttpMethod::Post);
        assert_eq!(
            code.body.as_deref(),
            Some(r#"{"client_id":"opencode-cli","supports_org_scope":true}"#)
        );
        let headers: HashMap<&str, &str> = code
            .headers
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        assert_eq!(headers["user-agent"], OPENCODE_USER_AGENT);
        assert!(headers.contains_key("b3"));
        assert!(headers.contains_key("traceparent"));

        let poll = &http.requests_for(&device_token_url(OPENCODE_CONSOLE_BASE))[0];
        assert_eq!(
            poll.body.as_deref(),
            Some(
                r#"{"grant_type":"urn:ietf:params:oauth:grant-type:device_code","device_code":"dev-1","client_id":"opencode-cli"}"#
            )
        );
        let user = &http.requests_for(&user_url(OPENCODE_CONSOLE_BASE))[0];
        assert!(user
            .headers
            .iter()
            .any(|(name, value)| name == "authorization" && value == "Bearer oc-access"));
        // The identity lookups share the login's trace id.
        let b3 = |request: &ProviderHttpRequest| {
            request
                .headers
                .iter()
                .find(|(name, _)| name == "b3")
                .map(|(_, value)| value.split('-').next().unwrap().to_string())
                .unwrap()
        };
        assert_eq!(
            b3(&http.requests_for(&user_url(OPENCODE_CONSOLE_BASE))[0]),
            b3(&http.requests_for(&orgs_url(OPENCODE_CONSOLE_BASE))[0])
        );
    }

    #[tokio::test]
    async fn an_explicit_token_org_must_exist() {
        let http = ScriptedHttp::new()
            .queue(
                &device_code_url(OPENCODE_CONSOLE_BASE),
                vec![device_response("")],
            )
            .queue(
                &device_token_url(OPENCODE_CONSOLE_BASE),
                vec![ScriptedResponse {
                    status: 200,
                    body: r#"{"access_token":"a","refresh_token":"r","expires_in":3600,"org_id":"org_b"}"#
                        .to_string(),
                }],
            )
            .queue(
                &user_url(OPENCODE_CONSOLE_BASE),
                vec![ScriptedHttp::entry(200, r#"{"id":"u","email":"e@example.com"}"#)],
            )
            .queue(
                &orgs_url(OPENCODE_CONSOLE_BASE),
                vec![ScriptedHttp::entry(
                    200,
                    r#"[{"id":"org_b","name":"beta"},{"id":"org_a","name":"alpha"}]"#,
                )],
            );
        let credentials = login_opencode(&http, &ScriptedUi::new()).await.unwrap();
        assert_eq!(credentials.org_id.as_deref(), Some("org_b"));
        assert_eq!(credentials.org_name.as_deref(), Some("beta"));

        let missing = ScriptedHttp::new()
            .queue(
                &device_code_url(OPENCODE_CONSOLE_BASE),
                vec![device_response("")],
            )
            .queue(
                &device_token_url(OPENCODE_CONSOLE_BASE),
                vec![ScriptedResponse {
                    status: 200,
                    body: r#"{"access_token":"a","refresh_token":"r","expires_in":3600,"org_id":"org_z"}"#
                        .to_string(),
                }],
            )
            .queue(
                &user_url(OPENCODE_CONSOLE_BASE),
                vec![ScriptedHttp::entry(200, r#"{"id":"u","email":"e@example.com"}"#)],
            )
            .queue(
                &orgs_url(OPENCODE_CONSOLE_BASE),
                vec![ScriptedHttp::entry(200, r#"[{"id":"org_b","name":"beta"}]"#)],
            );
        let error = login_opencode(&missing, &ScriptedUi::new())
            .await
            .unwrap_err();
        assert_eq!(error, "OpenCode organization not found: org_z");
    }

    #[tokio::test]
    async fn a_slow_down_bumps_the_poll_interval() {
        let http = ScriptedHttp::new()
            .queue(
                &device_code_url(OPENCODE_CONSOLE_BASE),
                vec![device_response("")],
            )
            .queue(
                &device_token_url(OPENCODE_CONSOLE_BASE),
                vec![
                    ScriptedHttp::entry(400, r#"{"error":"authorization_pending"}"#),
                    ScriptedHttp::entry(400, r#"{"error":"slow_down"}"#),
                    token_response("oc-access", "oc-refresh"),
                ],
            )
            .queue(
                &user_url(OPENCODE_CONSOLE_BASE),
                vec![ScriptedHttp::entry(
                    200,
                    r#"{"id":"u","email":"e@example.com"}"#,
                )],
            )
            .queue(
                &orgs_url(OPENCODE_CONSOLE_BASE),
                vec![ScriptedHttp::entry(
                    200,
                    r#"[{"id":"org_a","name":"alpha"}]"#,
                )],
            );
        let start = std::time::Instant::now();
        let credentials = login_opencode(&http, &ScriptedUi::new()).await.unwrap();
        assert_eq!(credentials.access, "oc-access");
        // Five seconds per slow_down answer (the harness's bump).
        assert!(
            start.elapsed() >= Duration::from_secs(5),
            "the bump waited: {:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn a_denied_or_expired_code_surfaces_the_ts_messages() {
        let denied = ScriptedHttp::new()
            .queue(
                &device_code_url(OPENCODE_CONSOLE_BASE),
                vec![device_response("")],
            )
            .queue(
                &device_token_url(OPENCODE_CONSOLE_BASE),
                vec![ScriptedHttp::entry(403, r#"{"error":"access_denied"}"#)],
            );
        let error = login_opencode(&denied, &ScriptedUi::new())
            .await
            .unwrap_err();
        assert_eq!(error, "OpenCode device authorization was denied");

        let expired = ScriptedHttp::new()
            .queue(
                &device_code_url(OPENCODE_CONSOLE_BASE),
                vec![device_response("")],
            )
            .queue(
                &device_token_url(OPENCODE_CONSOLE_BASE),
                vec![ScriptedHttp::entry(400, r#"{"error":"expired_token"}"#)],
            );
        let error = login_opencode(&expired, &ScriptedUi::new())
            .await
            .unwrap_err();
        assert_eq!(error, "OpenCode device code expired; sign in again");
    }

    #[tokio::test]
    async fn a_cancelled_surface_ends_the_poll() {
        let http = Arc::new(
            ScriptedHttp::new()
                .queue(
                    &device_code_url(OPENCODE_CONSOLE_BASE),
                    vec![ScriptedResponse {
                        status: 200,
                        body: r#"{"device_code":"d","user_code":"C","verification_uri_complete":"https://opencode.ai/console/auth/device","interval":5,"expires_in":900}"#
                            .to_string(),
                    }],
                )
                .queue(
                    &device_token_url(OPENCODE_CONSOLE_BASE),
                    vec![ScriptedHttp::entry(400, r#"{"error":"authorization_pending"}"#)],
                ),
        );
        let ui = Arc::new(ScriptedUi::new());
        let flow = {
            let http = Arc::clone(&http);
            let ui = Arc::clone(&ui);
            tokio::spawn(async move { login_opencode(http.as_ref(), ui.as_ref()).await })
        };
        // Readiness-wait for the URL: the first poll waits `interval`.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            if ui.auth_url.lock().unwrap().is_some() {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the flow never presented its url"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        ui.cancelled.store(true, Ordering::Relaxed);
        let error = tokio::time::timeout(Duration::from_secs(10), flow)
            .await
            .expect("the cancelled wait settles promptly")
            .unwrap()
            .unwrap_err();
        assert_eq!(error, LOGIN_CANCELLED);
    }

    #[tokio::test]
    async fn an_invalid_grant_refresh_names_the_cause() {
        let http = ScriptedHttp::new().queue(
            &device_token_url(OPENCODE_CONSOLE_BASE),
            vec![ScriptedHttp::entry(400, r#"{"error":"invalid_grant"}"#)],
        );
        let error = refresh_opencode_token(&http, "oc-old", None)
            .await
            .unwrap_err();
        assert_eq!(error, "OpenCode Console session expired; log in again");
    }

    #[tokio::test]
    async fn the_refresh_posts_the_rotated_grant() {
        let http = ScriptedHttp::new().queue(
            &device_token_url(OPENCODE_CONSOLE_BASE),
            vec![ScriptedResponse {
                status: 200,
                body: r#"{"access_token":"oc-fresh","refresh_token":"oc-next","expires_in":600,"org_id":"org_b"}"#
                    .to_string(),
            }],
        );
        let credentials = refresh_opencode_token(&http, "oc-old", Some(OPENCODE_CONSOLE_BASE))
            .await
            .unwrap();
        assert_eq!(credentials.access, "oc-fresh");
        assert_eq!(credentials.refresh, "oc-next");
        assert_eq!(credentials.org_id.as_deref(), Some("org_b"));
        let request = &http.requests_for(&device_token_url(OPENCODE_CONSOLE_BASE))[0];
        assert_eq!(
            request.body.as_deref(),
            Some(
                r#"{"grant_type":"refresh_token","refresh_token":"oc-old","client_id":"opencode-cli"}"#
            )
        );
    }

    #[tokio::test]
    async fn an_untrusted_verification_uri_fails_the_flow() {
        for raw in ["ftp://opencode.ai/device", "javascript:alert(1)"] {
            let http = ScriptedHttp::new().queue(
                &device_code_url(OPENCODE_CONSOLE_BASE),
                vec![ScriptedResponse {
                    status: 200,
                    body: format!(
                        r#"{{"device_code":"d","user_code":"C","verification_uri_complete":"{raw}","interval":0,"expires_in":900}}"#
                    ),
                }],
            );
            let error = login_opencode(&http, &ScriptedUi::new()).await.unwrap_err();
            assert!(
                error.starts_with("Invalid device verification URL:"),
                "{error}"
            );
        }
    }

    #[tokio::test]
    async fn a_missing_device_field_surfaces_the_validation_error() {
        let http = ScriptedHttp::new().queue(
            &device_code_url(OPENCODE_CONSOLE_BASE),
            vec![ScriptedHttp::entry(
                200,
                r#"{"device_code":"d","user_code":"C","interval":0,"expires_in":900}"#,
            )],
        );
        let error = login_opencode(&http, &ScriptedUi::new()).await.unwrap_err();
        assert_eq!(
            error,
            "Device authorization response is missing required fields"
        );
    }

    #[test]
    fn the_seeded_harness_id_is_deterministic() {
        let first = harness_session_id(Some("conversation"));
        assert_eq!(first, harness_session_id(Some("conversation")));
        assert!(first.starts_with("ses_"));
        assert_eq!(first.len(), "ses_".len() + 26);
        assert!(is_harness_session_id(&first), "{first}");
        // The seeded derivation is a pure function of the seed: it never fills
        // the process memo, so a restarted worker reproduces the same id.
        assert!(HARNESS_IDS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty());
        assert_ne!(first, harness_session_id(Some("another-conversation")));
    }

    #[test]
    fn an_explicit_harness_id_is_kept_verbatim() {
        let id = "0a1b2c3d4e5fAbCdEfGhIjKlMn".to_string();
        assert!(is_harness_session_id(&id));
        assert_eq!(harness_session_id(Some(&id)), format!("ses_{id}"));
        let prefixed = format!("ses_{id}");
        assert_eq!(harness_session_id(Some(&prefixed)), prefixed);
    }
}
