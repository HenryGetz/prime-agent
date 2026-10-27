//! Conformance of the hand-written `OpenCode` Go integration with the generated
//! protocol spec: routes, grant bodies, and the transport rule table must all
//! come from [`pa_ai::opencode_go_spec_generated`], never from a local copy.

use pa_ai::opencode_go_models::GoTransport;
use pa_ai::opencode_go_spec_generated as spec;

/// Every protocol constant is generated from the spec's own digest.
#[test]
fn spec_sha256_is_64_lowercase_hex() {
    assert_eq!(spec::SPEC_SHA256.len(), 64, "sha256 hex length");
    assert!(
        spec::SPEC_SHA256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "sha256 must be lowercase hex: {}",
        spec::SPEC_SHA256
    );
}

/// The device grant bodies are built from the spec's grant type, client id, and
/// org-scope switch.
#[test]
fn device_grant_bodies_use_the_spec_fields() {
    let code = pa_ai::oauth::device_code_body();
    assert!(
        code.contains(&format!("\"{}\"", spec::CLIENT_ID)),
        "device code body lacks the spec client id: {code}"
    );
    assert!(
        code.contains(&format!(
            "\"supports_org_scope\":{}",
            spec::SUPPORTS_ORG_SCOPE
        )),
        "device code body lacks the spec org-scope switch: {code}"
    );

    let token = pa_ai::oauth::device_token_body("device-1");
    assert!(
        token.contains(&format!("\"grant_type\":\"{}\"", spec::DEVICE_GRANT_TYPE)),
        "device token body lacks the spec grant type: {token}"
    );
    assert!(
        token.contains(&format!("\"client_id\":\"{}\"", spec::CLIENT_ID)),
        "device token body lacks the spec client id: {token}"
    );

    let refresh = pa_ai::oauth::refresh_token_body("refresh-1");
    assert!(
        refresh.contains(&format!("\"grant_type\":\"{}\"", spec::REFRESH_GRANT_TYPE)),
        "refresh body lacks the spec grant type: {refresh}"
    );
    assert!(
        refresh.contains(&format!("\"client_id\":\"{}\"", spec::CLIENT_ID)),
        "refresh body lacks the spec client id: {refresh}"
    );
}

/// The npm→api→base rule table is the spec's; an unlisted hint (including the
/// hidden `@ai-sdk/google`) is chat/completions.
#[test]
fn npm_rules_map_to_spec_apis_and_bases() {
    for ((npm, api), base) in spec::TRANSPORT_RULES_NPM
        .iter()
        .zip(spec::TRANSPORT_RULES_API)
        .zip(spec::TRANSPORT_RULES_BASE)
    {
        let transport = GoTransport::from_npm(Some(npm));
        assert_eq!(transport.api(), *api, "api for {npm}");
        assert_eq!(transport.spec_base(), *base, "base for {npm}");
    }
    assert!(
        spec::HIDDEN_NPM.contains(&"@ai-sdk/google"),
        "the unverified gemini route must stay hidden"
    );
    assert_eq!(
        GoTransport::from_npm(Some("@ai-sdk/google")),
        GoTransport::Completions
    );
    assert_eq!(GoTransport::from_npm(None), GoTransport::Completions);
    assert_eq!(
        GoTransport::Completions.api(),
        spec::DEFAULT_TRANSPORT_API,
        "the default wire api is the spec's"
    );
    assert_eq!(
        GoTransport::Completions.spec_base(),
        spec::DEFAULT_TRANSPORT_BASE,
        "the default provider family is the spec's"
    );
}

/// Each lane joins its gateway base to the spec's path.
#[test]
fn gateway_routes_join_the_spec_paths() {
    assert_eq!(
        GoTransport::Completions.request_url(),
        format!("{}{}", spec::GATEWAY_OPENAI_BASE_URL, spec::CHAT_PATH)
    );
    assert_eq!(
        GoTransport::Responses.request_url(),
        format!("{}{}", spec::GATEWAY_OPENAI_BASE_URL, spec::RESPONSES_PATH)
    );
    assert_eq!(
        GoTransport::Anthropic.request_url(),
        format!(
            "{}{}",
            spec::GATEWAY_ANTHROPIC_BASE_URL,
            spec::MESSAGES_PATH
        )
    );
    assert_eq!(
        GoTransport::Completions.request_url(),
        "https://opencode.ai/zen/go/v1/chat/completions"
    );
    assert_eq!(
        GoTransport::Responses.request_url(),
        "https://opencode.ai/zen/go/v1/responses"
    );
    assert_eq!(
        GoTransport::Anthropic.request_url(),
        "https://opencode.ai/zen/go/v1/messages"
    );
}

/// The usage route is the spec's gateway base plus the spec's usage path.
#[test]
fn usage_url_is_the_spec_base_plus_path() {
    assert_eq!(
        format!("{}{}", spec::GATEWAY_OPENAI_BASE_URL, spec::USAGE_PATH),
        "https://opencode.ai/zen/go/v1/usage"
    );
    assert_eq!(spec::USAGE_WINDOWS, ["rolling", "weekly", "monthly"]);
    assert_eq!(spec::VERDICT_STATUSES, [401, 402, 403]);
    assert_eq!(spec::RATE_LIMIT_STATUS, 429);
}
