//! Auxiliary-model routing for background summarizer passes: compaction
//! and branch summaries resolve their model through the `auxiliaryModel`
//! setting, so their one-off prompts stay off the prompt-cache prefix; an
//! unset, equal, or unusable selector falls back to the session model.
//! Also sizes the output budget of side calls (refine, the auto-refine
//! review, branch summaries).

use std::path::PathBuf;

use pa_types::ai::Model;

/// The directories the settings (`auxiliaryModel`, `allowedModels`) and the
/// model registry resolve against; `None` keeps the session model.
#[derive(Debug, Clone)]
pub struct AuxiliaryModelContext {
    pub cwd: PathBuf,
    pub agent_dir: PathBuf,
}

/// The session fallback for a summarizer call, WITH its merged request
/// headers — the registry's single team header, shipped like the session's own requests.
pub(crate) fn session_fallback_with_headers(
    context: &AuxiliaryModelContext,
    session_model: &Model,
    session_api_key: Option<String>,
) -> ResolvedAuxiliaryModel {
    let auth = crate::auth::AuthStorage::create(&context.agent_dir);
    let mut registry =
        crate::models::ModelRegistry::create(auth, context.agent_dir.join("models.json"));
    let resolved = registry.get_api_key_and_headers(session_model, session_model.headers.as_ref());
    ResolvedAuxiliaryModel {
        model: session_model.clone(),
        api_key: session_api_key,
        headers: resolved.headers,
    }
}

/// One routed summarizer target: the model, the key, and the merged
/// request headers its provider needs.
#[derive(Debug, Clone)]
pub struct ResolvedAuxiliaryModel {
    pub model: Model,
    pub api_key: Option<String>,
    pub headers: Option<std::collections::BTreeMap<String, String>>,
}

/// The output budget a side call asks for: the model's whole `max_tokens`
/// while the call runs with reasoning (thinking and the answer share it, so a
/// cap can cut the answer off), and at most `no_reasoning_cap` without
/// reasoning. The caller still fits it to the context room its input leaves.
pub(crate) fn side_call_output_budget(model: &Model, no_reasoning_cap: u64) -> u64 {
    match pa_types::ai::minimum_reasoning(model) {
        Some(_) => model.max_tokens,
        None => model.max_tokens.min(no_reasoning_cap),
    }
}

/// The fallback warning: the selector is logged, never the auth-stack
/// error details (those can embed credential material).
fn warn_fallback(selector: &str, purpose: &str) {
    eprintln!(
        "Warning: auxiliaryModel \"{selector}\" unusable for {purpose}; using the session model."
    );
}

/// Resolve a background summarizer pass's model through the `auxiliaryModel`
/// setting; `session_model` + `session_api_key` are the fallback.
/// `required_context_tokens` is the pass's request size (None when unknown):
/// a smaller known auxiliary window falls back instead of failing on the wire.
#[must_use]
pub fn resolve_auxiliary_model(
    context: &AuxiliaryModelContext,
    purpose: &str,
    session_model: &Model,
    session_api_key: Option<&str>,
    required_context_tokens: Option<u64>,
) -> ResolvedAuxiliaryModel {
    let fallback = || {
        session_fallback_with_headers(context, session_model, session_api_key.map(str::to_string))
    };
    let settings = crate::settings::SettingsManager::create(&context.cwd, &context.agent_dir);
    // A malformed or whitespace value behaves as unset.
    let selector = settings
        .get_auxiliary_model()
        .map(|selector| selector.trim().to_lowercase())
        .filter(|selector| !selector.is_empty());
    let Some(selector) = selector else {
        return fallback();
    };
    if format!("{}/{}", session_model.provider, session_model.id).to_lowercase() == selector {
        return fallback();
    }
    // The daemon's `allowedModels` pin (rust-only guardrail, no TS
    // equivalent) covers every pass model: a selector outside the pin
    // falls back to the session model.
    if let Some(allowlist) = settings.get_allowed_models() {
        if !crate::models::model_allowed(&selector, &allowlist) {
            warn_fallback(&selector, purpose);
            return fallback();
        }
    }
    // The TS find runs over the authenticated, non-stale catalog
    // (`_authenticatedRlmModels`); the registry's set is the same filter.
    let auth = crate::auth::AuthStorage::create(&context.agent_dir);
    let mut registry =
        crate::models::ModelRegistry::create(auth, context.agent_dir.join("models.json"));
    registry.load_private_authorization_from_cache();
    let model = registry
        .get_rlm_searchable_models()
        .into_iter()
        .find(|model| format!("{}/{}", model.provider, model.id).to_lowercase() == selector)
        .cloned();
    let Some(model) = model else {
        warn_fallback(&selector, purpose);
        return fallback();
    };
    let resolved = registry.get_api_key_and_headers(&model, model.headers.as_ref());
    if !resolved.ok {
        warn_fallback(&selector, purpose);
        return fallback();
    }
    if let Some(required) = required_context_tokens {
        if model.context_window > 0 && model.context_window < required {
            warn_fallback(&selector, purpose);
            return fallback();
        }
    }
    ResolvedAuxiliaryModel {
        model,
        api_key: resolved.api_key,
        headers: resolved.headers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(id: &str, provider: &str, context_window: u64) -> Model {
        serde_json::from_value(serde_json::json!({
            "id": id, "name": id, "api": "openai-completions", "provider": provider,
            "baseUrl": "http://localhost", "reasoning": false, "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": context_window, "maxTokens": 100
        }))
        .unwrap()
    }

    /// The custom provider the auxiliary model lives under (an `apiKey` on
    /// the provider config is configured auth for the registry).
    const MODELS_JSON: &str = r#"{
      "providers": {
        "testaux": {
          "baseUrl": "http://localhost:9",
          "apiKey": "aux-key",
          "api": "openai-completions",
          "models": [ { "id": "aux-model", "name": "Aux Model", "contextWindow": 128000 } ]
        }
      }
    }"#;

    /// The tempdir must outlive the resolve call (the registry reads
    /// models.json on demand); each test keeps the handle in scope.
    fn context_with_settings(
        settings: &serde_json::Value,
    ) -> (tempfile::TempDir, AuxiliaryModelContext) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("settings.json"),
            serde_json::to_string_pretty(&settings).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.path().join("models.json"), MODELS_JSON).unwrap();
        let context = AuxiliaryModelContext {
            cwd: dir.path().to_path_buf(),
            agent_dir: dir.path().to_path_buf(),
        };
        (dir, context)
    }

    /// The session fallback resolves its MERGED headers through the same
    /// registry the session's own requests use.
    #[test]
    fn the_session_fallback_keeps_the_merged_headers() {
        let (dir, context) = context_with_settings(&serde_json::json!({}));
        std::fs::write(
            dir.path().join("auth.json"),
            serde_json::json!({
                "prime-inference": {
                    "type": "api_key",
                    "key": "test-key",
                    "primeTeam": { "teamId": "team-1", "name": "Test Team" }
                }
            })
            .to_string(),
        )
        .unwrap();
        let session = model("session-model", "prime-inference", 8_000);
        let routed = session_fallback_with_headers(&context, &session, Some("override".into()));
        assert_eq!(routed.model.id, "session-model");
        assert_eq!(routed.api_key.as_deref(), Some("override"));
        let headers = routed.headers.expect("the fallback keeps merged headers");
        assert_eq!(
            headers.get("X-Prime-Team-ID").map(String::as_str),
            Some("team-1"),
            "the stored team ships on the fallback summarizer call"
        );
    }

    #[test]
    fn unset_selector_keeps_the_session_model() {
        let (_dir, context) = context_with_settings(&serde_json::json!({}));
        let session = model("session-model", "faux", 8_000);
        let routed = resolve_auxiliary_model(
            &context,
            "compaction summary",
            &session,
            Some("session-key"),
            None,
        );
        assert_eq!(routed.model.id, "session-model");
        assert_eq!(routed.api_key.as_deref(), Some("session-key"));
    }

    #[test]
    fn selector_equal_to_the_session_model_keeps_the_session_model() {
        let (_dir, context) =
            context_with_settings(&serde_json::json!({ "auxiliaryModel": "faux/session-model" }));
        let session = model("session-model", "faux", 8_000);
        let routed = resolve_auxiliary_model(
            &context,
            "compaction summary",
            &session,
            Some("session-key"),
            None,
        );
        assert_eq!(routed.model.id, "session-model");
        assert_eq!(routed.api_key.as_deref(), Some("session-key"));
    }

    #[test]
    fn configured_selector_routes_to_the_auxiliary_model() {
        let (_dir, context) =
            context_with_settings(&serde_json::json!({ "auxiliaryModel": "testaux/aux-model" }));
        let session = model("session-model", "faux", 8_000);
        let routed = resolve_auxiliary_model(
            &context,
            "compaction summary",
            &session,
            Some("session-key"),
            None,
        );
        assert_eq!(routed.model.id, "aux-model");
        assert_eq!(routed.api_key.as_deref(), Some("aux-key"));
        assert_eq!(routed.headers, None);
    }

    #[test]
    fn unknown_selector_falls_back_to_the_session_model() {
        let (_dir, context) =
            context_with_settings(&serde_json::json!({ "auxiliaryModel": "testaux/missing" }));
        let session = model("session-model", "faux", 8_000);
        let routed = resolve_auxiliary_model(
            &context,
            "branch summary",
            &session,
            Some("session-key"),
            None,
        );
        assert_eq!(routed.model.id, "session-model");
        assert_eq!(routed.api_key.as_deref(), Some("session-key"));
    }

    #[test]
    fn too_small_window_falls_back_to_the_session_model() {
        let (_dir, context) =
            context_with_settings(&serde_json::json!({ "auxiliaryModel": "testaux/aux-model" }));
        let session = model("session-model", "faux", 8_000);
        let routed = resolve_auxiliary_model(
            &context,
            "compaction summary",
            &session,
            Some("session-key"),
            Some(200_000),
        );
        assert_eq!(routed.model.id, "session-model");
        let routed = resolve_auxiliary_model(
            &context,
            "compaction summary",
            &session,
            Some("session-key"),
            Some(1_000),
        );
        assert_eq!(routed.model.id, "aux-model");
    }

    #[test]
    fn selector_outside_the_allowlist_falls_back() {
        let (_dir, context) = context_with_settings(&serde_json::json!({
            "auxiliaryModel": "testaux/aux-model",
            "allowedModels": ["faux/session-model"],
        }));
        let session = model("session-model", "faux", 8_000);
        let routed = resolve_auxiliary_model(
            &context,
            "compaction summary",
            &session,
            Some("session-key"),
            None,
        );
        assert_eq!(routed.model.id, "session-model");
        assert_eq!(routed.api_key.as_deref(), Some("session-key"));
    }

    /// Whether the faux model can run with reasoning off.
    enum Thinking {
        Optional,
        Mandatory,
    }

    /// Run refine, the auto-refine review, and a branch summary on a faux model
    /// and return the output budget each request resolves to on the wire (an
    /// unset budget resolves like `build_base_options`). Every budget must fit
    /// the context room the request's input leaves (pa-ai's chars/4 estimate).
    async fn side_call_budgets(
        thinking: Thinking,
        context_window: u64,
        max_tokens: u64,
    ) -> Vec<u64> {
        let registration =
            pa_ai::faux::register_faux_provider(pa_ai::faux::RegisterFauxProviderOptions {
                models: Some(vec![pa_ai::faux::FauxModelDefinition {
                    id: "side-call-model".to_string(),
                    reasoning: Some(true),
                    context_window: Some(context_window),
                    max_tokens: Some(max_tokens),
                    ..Default::default()
                }]),
                ..Default::default()
            });
        let mut model = registration.get_model();
        if let Thinking::Mandatory = thinking {
            model.thinking_level_map = Some(pa_types::ai::thinking_level_map(&[(
                pa_types::ai::ModelThinkingLevel::Off,
                None,
            )]));
        }
        let requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorder = std::sync::Arc::clone(&requests);
        let step = pa_ai::faux::FauxResponseStep::Factory(std::sync::Arc::new(
            move |context: &pa_types::ai::Context,
                  options: Option<&pa_ai::types::StreamOptions>,
                  _call: u64,
                  model: &Model| {
                let budget = options
                    .and_then(|options| options.max_tokens)
                    .or_else(|| pa_ai::default_request_max_tokens(model))
                    .unwrap();
                let input_chars = context
                    .system_prompt
                    .as_deref()
                    .unwrap_or("")
                    .chars()
                    .count()
                    + context
                        .messages
                        .iter()
                        .map(|message| match message {
                            pa_types::ai::Message::User(user) => {
                                user.content.text().chars().count()
                            }
                            _ => 0,
                        })
                        .sum::<usize>();
                let room = context_window - (input_chars as u64).div_ceil(4);
                recorder.lock().unwrap().push((budget, room));
                Ok(pa_ai::faux::faux_assistant_text_message(
                    r#"{"summary":"s","edits":[],"shouldRefine":false}"#,
                    pa_ai::faux::FauxAssistantMessageOptions::default(),
                ))
            },
        ));
        registration.set_responses(vec![step; 3]);

        let message = pa_types::session::AgentMessage::User(pa_types::ai::UserMessage {
            content: pa_types::ai::UserContent::Text("explore the widget".to_string()),
            timestamp: 0,
            rest: serde_json::Map::default(),
        });
        let state = crate::refinement::empty_harness_state();
        let refiner =
            || crate::session_engine::refine::default_refiner_call(None, "session".to_string());
        crate::refinement::executor::plan_refinement(
            std::slice::from_ref(&message),
            &state,
            &[],
            &model,
            &crate::refinement::executor::RefineOptions::default(),
            refiner(),
        )
        .await
        .unwrap();
        crate::refinement::executor::review_auto_refine(
            std::slice::from_ref(&message),
            &state,
            &[],
            &model,
            &crate::refinement::executor::AutoRefineReviewContext {
                reason: "checkpoint".to_string(),
                turns_since_last_review: 1,
            },
            refiner(),
        )
        .await
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut session = crate::session::manager::SessionManager::in_memory(dir.path());
        session.append_message(message).unwrap();
        let summary = super::super::branch_summarization::generate_branch_summary(
            session.retained_entries(),
            super::super::branch_summarization::GenerateBranchSummaryOptions {
                model: &model,
                api_key: None,
                custom_instructions: None,
                replace_instructions: false,
                reserve_tokens: super::super::branch_summarization::DEFAULT_BRANCH_RESERVE_TOKENS,
                auxiliary: None,
                session_id: None,
            },
        )
        .await;
        assert_eq!(summary.error, None);
        registration.unregister();
        let requests = requests.lock().unwrap().clone();
        assert!(
            requests.iter().all(|(budget, room)| budget <= room),
            "{requests:?}"
        );
        requests.into_iter().map(|(budget, _)| budget).collect()
    }

    /// Refine, the auto-refine review, and a branch summary on a model that must
    /// reason get its whole output budget; with reasoning off they keep their
    /// caps; no budget exceeds the model's max output or the context room.
    #[tokio::test]
    async fn side_calls_size_their_output_budget_for_reasoning() {
        assert_eq!(
            side_call_budgets(Thinking::Mandatory, 1_000_000, 128_000).await,
            [128_000; 3]
        );
        assert_eq!(
            side_call_budgets(Thinking::Optional, 1_000_000, 128_000).await,
            [32_000, 4_096, 2_048]
        );
        assert_eq!(
            side_call_budgets(Thinking::Optional, 1_000_000, 1_000).await,
            [1_000; 3]
        );
        // The room, not the model's max output, bounds a small window.
        let budgets = side_call_budgets(Thinking::Mandatory, 40_000, 128_000).await;
        assert!(budgets.iter().all(|budget| *budget < 40_000), "{budgets:?}");
    }
}
