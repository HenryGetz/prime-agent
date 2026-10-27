//! Shared provider-failure recording for every surface that settles a turn.
//!
//! The daemon worker and the CLI `--print` path both settle turns; without one
//! seam only the daemon learned from a settled failure, so a `--print` run never
//! demoted a rate-limited `OpenCode` Go account, recorded a workspace verdict,
//! or retired a dead model. This module owns that policy in one place, over the
//! existing dead-model, account-routing, and registry APIs.

use std::path::Path;

use pa_agent::types::AssistantMessage;

use crate::auth::AuthStorage;
use crate::models::ModelRegistry;
use crate::session_engine::provider_retry::{
    provider_stream_failure_retry_after_ms, provider_stream_failure_status,
};

use pa_ai::opencode_go_spec_generated as spec;

/// Record a settled turn's provider failure against the session's routing
/// state. A failure from any provider other than `opencode-go` says nothing
/// about the Go account stack, so it is a no-op; the model is marked dead on a
/// 410 or a retirement message, a 429 demotes the serving account for its
/// retry-after window, and a 401/402/403 records the account verdict.
///
/// `session_key` is the stable per-session routing identity; `None` lets the
/// routing state pick the slot (the CLI print path has no daemon session id).
pub fn record_provider_failure(
    agent_dir: &Path,
    session_key: Option<&str>,
    message: &AssistantMessage,
) {
    if message.provider != pa_ai::opencode_go_models::OPENCODE_GO_PROVIDER_ID {
        return;
    }
    let status = provider_stream_failure_status(message);
    let error_message = message.error_message.as_deref().unwrap_or_default();
    if pa_ai::opencode_go_dead_models::is_dead_failure(status, error_message) {
        pa_ai::opencode_go_dead_models::record(agent_dir, &message.model);
    }
    let rate_limited = status == Some(spec::RATE_LIMIT_STATUS);
    let verdict = status.is_some_and(|status| spec::VERDICT_STATUSES.contains(&status));
    if !rate_limited && !verdict {
        return;
    }
    let auth = AuthStorage::create(agent_dir);
    let registry = ModelRegistry::create(auth, agent_dir.join("models.json"));
    if rate_limited {
        registry.record_rate_limit(session_key, provider_stream_failure_retry_after_ms(message));
    }
    if let Some(status) = status {
        if spec::VERDICT_STATUSES.contains(&status) {
            registry.record_inference_verdict(session_key, status, error_message);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::opencode_accounts;
    use serde_json::json;

    /// The first slot's persisted state, whatever the routing choice picked.
    fn only_account_state(agent_dir: &Path) -> opencode_accounts::AccountState {
        let state = opencode_accounts::load(agent_dir);
        state
            .accounts
            .values()
            .next()
            .cloned()
            .expect("the helper recorded one account")
    }

    fn write_slots(agent_dir: &Path) {
        std::fs::create_dir_all(agent_dir).unwrap();
        std::fs::write(
            agent_dir.join("auth.json"),
            json!({
                "opencode-go": {
                    "type": "oauth",
                    "access": "access-a",
                    "refresh": "refresh-a",
                    "expires": i64::MAX / 2,
                    "accountId": "usr_a",
                    "orgId": "org_a",
                    "apiEndpoint": "https://opencode.ai",
                },
            })
            .to_string(),
        )
        .unwrap();
    }

    fn failure(status: u16, message: &str) -> AssistantMessage {
        failure_for("opencode-go", status, message)
    }

    fn failure_for(provider: &str, status: u16, message: &str) -> AssistantMessage {
        serde_json::from_value(json!({
            "content": [],
            "api": provider,
            "provider": provider,
            "model": "deepseek-v4.1-flash",
            "usage": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 0,
                "cost": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "total": 0,
                },
            },
            "stopReason": "error",
            "diagnostics": [{
                "type": "provider_stream_failure",
                "timestamp": 0,
                "details": {"kind": "auth", "status": status, "retryAfterMs": 30_000},
            }],
            "errorMessage": message,
            "timestamp": 0,
        }))
        .unwrap()
    }

    #[test]
    fn a_429_demotes_the_serving_slot() {
        let dir = tempfile::tempdir().unwrap();
        let agent_dir = dir.path().join("agent");
        write_slots(&agent_dir);

        record_provider_failure(
            &agent_dir,
            Some("session-a"),
            &failure(429, "Too Many Requests"),
        );

        let account = only_account_state(&agent_dir);
        assert!(
            account.rate_limited_until_ms > opencode_accounts::epoch_ms(),
            "the settled 429 sets rateLimitedUntilMs"
        );
    }

    #[test]
    fn a_403_records_the_account_verdict() {
        let dir = tempfile::tempdir().unwrap();
        let agent_dir = dir.path().join("agent");
        write_slots(&agent_dir);

        record_provider_failure(
            &agent_dir,
            Some("session-a"),
            &failure(403, "Workspace access denied"),
        );

        let account = only_account_state(&agent_dir);
        let verdict = account.verdict.expect("the settled 403 records a verdict");
        assert_eq!(verdict.status, 403);
        assert_eq!(verdict.message, "Workspace access denied");
    }

    #[test]
    fn a_410_learns_the_model_as_dead() {
        let dir = tempfile::tempdir().unwrap();
        let agent_dir = dir.path().join("agent");
        write_slots(&agent_dir);

        record_provider_failure(&agent_dir, Some("session-a"), &failure(410, "Gone"));

        assert_eq!(
            pa_ai::opencode_go_dead_models::load(&agent_dir),
            std::collections::BTreeSet::from(["deepseek-v4.1-flash".to_string()])
        );
    }

    #[test]
    fn another_provider_leaves_the_go_state_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let agent_dir = dir.path().join("agent");
        write_slots(&agent_dir);
        let before = opencode_accounts::load(&agent_dir);

        record_provider_failure(
            &agent_dir,
            Some("session-a"),
            &failure_for("anthropic", 403, "Workspace access denied"),
        );

        assert_eq!(opencode_accounts::load(&agent_dir), before);
        assert!(pa_ai::opencode_go_dead_models::load(&agent_dir).is_empty());
    }
}
