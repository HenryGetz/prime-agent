//! `OpenCode` routing identity is conversation-scoped, independently of prompt caching.

use crate::opencode_go_spec_generated as spec;

pub(super) fn apply(
    headers: &mut Vec<(String, String)>,
    provider: &str,
    session_id: Option<&str>,
    base_url: &str,
) {
    // Provider-id matching misses custom names and registry aliases, while the
    // gateway answers `400 … missing x-opencode-session` before auth for any
    // Go/zen inference URL. Match the endpoint too: one identity for every
    // request that lands on OpenCode's gateway.
    let opencode_gateway = base_url.contains("opencode.ai/zen/go")
        || base_url.contains("opencode.ai/inference/go")
        || base_url.contains("opencode.ai/zen")
        || base_url.contains("opencode.ai/inference");
    if !opencode_gateway && !matches!(provider, "opencode" | "opencode-go") {
        return;
    }
    // The gateway validates the harness's descending-time session encoding, so
    // an ordinary conversation id maps to a stable harness-shaped `ses_…` id.
    let session = crate::oauth::harness_session_id(session_id);
    // TS `withOpenCodeHeaders`: explicit caller headers win.
    for (name, value) in [
        ("User-Agent", spec::USER_AGENT.to_string()),
        (spec::HEADER_SESSION, session.clone()),
        (spec::HEADER_SESSION_ID, session.clone()),
        (spec::HEADER_SESSION_AFFINITY, session.clone()),
        (spec::HEADER_SESSION_ID_ALIAS, session),
        (spec::HEADER_CLIENT, "cli".to_string()),
    ] {
        if !headers
            .iter()
            .any(|(existing, _)| existing.eq_ignore_ascii_case(name))
        {
            headers.push((name.to_string(), value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
        headers
            .iter()
            .find(|(existing, _)| existing.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    fn apply_to(provider: &str, base_url: &str) -> Vec<(String, String)> {
        let mut headers = vec![("Authorization".to_string(), "Bearer key".to_string())];
        apply(&mut headers, provider, Some("conversation"), base_url);
        headers
    }

    /// The reported failure: a custom provider id routed at the Go gateway sent no
    /// `x-opencode-session`, so the gateway answered `400 … missing x-opencode-session`
    /// before authenticating. The endpoint match must supply the full identity.
    #[test]
    fn custom_provider_on_opencode_gateway_gets_full_identity() {
        let expected = crate::oauth::harness_session_id(Some("conversation"));
        let headers = apply_to("my-gateway", "https://opencode.ai/zen/go/v1");
        for name in [
            spec::HEADER_SESSION,
            spec::HEADER_SESSION_ID,
            spec::HEADER_SESSION_AFFINITY,
            spec::HEADER_SESSION_ID_ALIAS,
        ] {
            assert_eq!(value(&headers, name), Some(expected.as_str()), "{name}");
        }
        assert_eq!(value(&headers, spec::HEADER_CLIENT), Some("cli"));
        assert_eq!(value(&headers, "User-Agent"), Some(spec::USER_AGENT));
        assert_eq!(value(&headers, "Authorization"), Some("Bearer key"));
    }

    #[test]
    fn every_opencode_gateway_url_gets_identity_for_custom_providers() {
        for base_url in [
            "https://opencode.ai/zen/go/v1",
            "https://opencode.ai/inference/go/v1",
            "https://opencode.ai/zen/v1",
            "https://opencode.ai/inference/v1",
        ] {
            let headers = apply_to("custom-name", base_url);
            assert!(
                value(&headers, spec::HEADER_SESSION).is_some(),
                "{base_url}"
            );
        }
    }

    #[test]
    fn opencode_provider_ids_get_identity_on_any_base_url() {
        for provider in ["opencode", "opencode-go"] {
            let headers = apply_to(provider, "https://example.invalid/v1");
            assert!(
                value(&headers, spec::HEADER_SESSION).is_some(),
                "{provider}"
            );
        }
    }

    #[test]
    fn unrelated_provider_and_url_get_no_identity() {
        let headers = apply_to("custom-gateway", "https://example.invalid/v1");
        assert!(value(&headers, spec::HEADER_SESSION).is_none());
        assert!(value(&headers, "User-Agent").is_none());
    }
}
