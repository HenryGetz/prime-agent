//! The `OpenCode` Go subscription's model roster: every non-deprecated entry of
//! the `opencode-go` catalog (the `go` lane of `https://models.opencode.ai/api.json`,
//! snapshotted in `omp-go/src/models-catalog.json`).
//!
//! The roster is hand-maintained because the provider is a subscription: the
//! gateway serves exactly this set, and a model list fetched live would ship
//! the free lane's entries too. `api` and `base_url` follow the catalog's
//! transport hint (`provider.npm`): `@ai-sdk/openai` → the Responses API,
//! `@ai-sdk/anthropic` → Anthropic Messages, everything else (including
//! `@ai-sdk/google`, whose gemini route is unverified) → chat/completions.

use pa_types::ai::{
    AnthropicMessagesCompat, CompatKind, MaxTokensField, Model, ModelCompat, ModelCost, ModelInput,
    ModelThinkingLevel, OpenAiCompletionsCompat, OpenAiResponsesCompat, ThinkingLevelMap,
};

use crate::opencode_go_spec_generated as spec;

/// The subscription provider id (a wire identifier; branding never renames a
/// provider id).
pub const OPENCODE_GO_PROVIDER_ID: &str = spec::PROVIDER_ID;
/// The Go gateway base the OpenAI-compatible lanes serve from.
pub const OPENCODE_GO_BASE_URL: &str = crate::oauth::OPENCODE_GO_BASE_URL;
/// The Go gateway base the Anthropic Messages lane serves from.
pub const OPENCODE_GO_ANTHROPIC_BASE_URL: &str = crate::oauth::OPENCODE_GO_ANTHROPIC_BASE_URL;
/// The `/login` row's display name.
pub const OPENCODE_GO_PROVIDER_NAME: &str = spec::PROVIDER_NAME;

/// The gateway lane a model serves on, selected by the catalog's
/// `provider.npm` hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoTransport {
    /// `@ai-sdk/openai`: the Responses API.
    Responses,
    /// `@ai-sdk/anthropic`: Anthropic Messages.
    Anthropic,
    /// Everything else, including the unverified `@ai-sdk/google` gemini
    /// route: chat/completions.
    Completions,
}

impl GoTransport {
    /// The transport `npm` selects, by the spec's npm→api rule table. A missing
    /// or unlisted hint (including `@ai-sdk/google`, whose gemini route is
    /// unverified) is chat/completions.
    #[must_use]
    pub fn from_npm(npm: Option<&str>) -> Self {
        let Some(npm) = npm.filter(|npm| !spec::HIDDEN_NPM.contains(npm)) else {
            return Self::Completions;
        };
        let api = spec::TRANSPORT_RULES_NPM
            .iter()
            .zip(spec::TRANSPORT_RULES_API)
            .find_map(|(rule, api)| (*rule == npm).then_some(*api))
            .unwrap_or(spec::DEFAULT_TRANSPORT_API);
        Self::from_api(api)
    }

    /// The transport a spec wire `api` id names; an unknown id is
    /// chat/completions.
    #[must_use]
    pub fn from_api(api: &str) -> Self {
        match api {
            api if api == spec::TRANSPORT_RULES_API[0] => Self::Responses,
            api if api == spec::TRANSPORT_RULES_API[1] => Self::Anthropic,
            _ => Self::Completions,
        }
    }

    /// The wire `api` id the session engine dispatches on.
    #[must_use]
    pub fn api(self) -> &'static str {
        match self {
            Self::Responses => spec::TRANSPORT_RULES_API[0],
            Self::Anthropic => spec::TRANSPORT_RULES_API[1],
            Self::Completions => spec::DEFAULT_TRANSPORT_API,
        }
    }

    /// The spec rule's provider family (`openai`/`anthropic`).
    #[must_use]
    pub fn spec_base(self) -> &'static str {
        match self {
            Self::Responses => spec::TRANSPORT_RULES_BASE[0],
            Self::Anthropic => spec::TRANSPORT_RULES_BASE[1],
            Self::Completions => spec::DEFAULT_TRANSPORT_BASE,
        }
    }

    /// The gateway base URL this lane serves from.
    #[must_use]
    pub fn base_url(self) -> &'static str {
        match self {
            Self::Anthropic => spec::GATEWAY_ANTHROPIC_BASE_URL,
            Self::Responses | Self::Completions => spec::GATEWAY_OPENAI_BASE_URL,
        }
    }

    /// The gateway route this lane's requests append to [`Self::base_url`].
    #[must_use]
    pub fn request_path(self) -> &'static str {
        match self {
            Self::Responses => spec::RESPONSES_PATH,
            Self::Anthropic => spec::MESSAGES_PATH,
            Self::Completions => spec::CHAT_PATH,
        }
    }

    /// The full gateway URL this lane's requests are sent to.
    #[must_use]
    pub fn request_url(self) -> String {
        format!("{}{}", self.base_url(), self.request_path())
    }
}

struct ModelSpec {
    id: &'static str,
    name: &'static str,
    reasoning: bool,
    context_window: u64,
    max_tokens: u64,
    /// Base tier: (input, output, cache read, cache write) per million tokens.
    cost: (f64, f64, f64, f64),
    /// The catalog's `reasoning_options` effort rungs; `None` when the model has
    /// no controllable effort surface (toggle/budget-only models).
    effort: Option<&'static [&'static str]>,
    /// The catalog transport hint for this model.
    transport: GoTransport,
    image: bool,
}

const ROSTER: &[ModelSpec] = &[
    ModelSpec {
        id: "mimo-v2.6-pro",
        name: "MiMo-V2.6-Pro",
        reasoning: true,
        context_window: 1_048_576,
        max_tokens: 131_072,
        cost: (0.435, 0.87, 0.003_625, 0.0),
        effort: None,
        transport: GoTransport::Completions,
        image: true,
    },
    ModelSpec {
        id: "mimo-v2.5",
        name: "MiMo V2.5",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 128_000,
        cost: (0.14, 0.28, 0.0028, 0.0),
        effort: None,
        transport: GoTransport::Completions,
        image: true,
    },
    ModelSpec {
        id: "grok-4.7",
        name: "Grok 4.7",
        reasoning: true,
        context_window: 500_000,
        max_tokens: 500_000,
        cost: (2.0, 6.0, 0.5, 0.0),
        effort: Some(&["low", "medium", "high", "xhigh"]),
        transport: GoTransport::Responses,
        image: true,
    },
    ModelSpec {
        id: "longcat-2.5-preview-free",
        name: "LongCat 2.5 Preview Free",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 131_072,
        cost: (0.0, 0.0, 0.0, 0.0),
        effort: None,
        transport: GoTransport::Completions,
        image: true,
    },
    ModelSpec {
        id: "glm-5.3-flash",
        name: "GLM-5.3-Flash",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 131_072,
        cost: (0.15, 0.5, 0.03, 0.0),
        effort: Some(&["low", "high", "max"]),
        transport: GoTransport::Completions,
        image: true,
    },
    ModelSpec {
        id: "qwen3.8-max",
        name: "Qwen3.8 Max",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 131_072,
        cost: (2.0, 6.0, 0.25, 2.5),
        effort: Some(&["low", "medium", "xhigh"]),
        transport: GoTransport::Anthropic,
        image: true,
    },
    ModelSpec {
        id: "kimi-k3",
        name: "Kimi K3",
        reasoning: true,
        context_window: 1_048_576,
        max_tokens: 131_072,
        cost: (3.0, 15.0, 0.3, 0.0),
        effort: Some(&["max"]),
        transport: GoTransport::Completions,
        image: true,
    },
    ModelSpec {
        id: "deepseek-v4.1-flash",
        name: "DeepSeek V4.1 Flash",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 384_000,
        cost: (0.15, 0.6, 0.003, 0.0),
        effort: Some(&["low", "high", "max"]),
        transport: GoTransport::Completions,
        image: true,
    },
    ModelSpec {
        id: "deepseek-v4-flash-vision-exp",
        name: "DeepSeek V4 Flash Vision Exp",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 384_000,
        cost: (0.15, 0.6, 0.003, 0.0),
        effort: Some(&["low", "high", "max"]),
        transport: GoTransport::Completions,
        image: true,
    },
    ModelSpec {
        id: "longcat-2.0",
        name: "LongCat-2.0",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 131_072,
        cost: (0.3, 1.2, 0.006, 0.0),
        effort: None,
        transport: GoTransport::Completions,
        image: false,
    },
    ModelSpec {
        id: "space-bunny",
        name: "Space Bunny",
        reasoning: true,
        context_window: 1_048_576,
        max_tokens: 524_288,
        cost: (0.15, 0.6, 0.03, 0.0),
        effort: Some(&["low", "medium", "high", "xhigh", "max"]),
        transport: GoTransport::Completions,
        image: true,
    },
    ModelSpec {
        id: "minimax-m2.7",
        name: "MiniMax-M2.7",
        reasoning: true,
        context_window: 204_800,
        max_tokens: 131_072,
        cost: (0.3, 1.2, 0.06, 0.375),
        effort: None,
        transport: GoTransport::Completions,
        image: false,
    },
    ModelSpec {
        id: "mimo-v2.5-pro",
        name: "MiMo V2.5 Pro",
        reasoning: true,
        context_window: 1_048_576,
        max_tokens: 128_000,
        cost: (0.435, 0.87, 0.003_625, 0.0),
        effort: None,
        transport: GoTransport::Completions,
        image: false,
    },
    ModelSpec {
        id: "mimo-v2.6-flash",
        name: "MiMo-V2.6-Flash",
        reasoning: true,
        context_window: 1_048_576,
        max_tokens: 131_072,
        cost: (0.14, 0.28, 0.0028, 0.0),
        effort: None,
        transport: GoTransport::Completions,
        image: true,
    },
    ModelSpec {
        id: "minimax-m3",
        name: "MiniMax-M3",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 131_072,
        cost: (0.3, 1.2, 0.06, 0.0),
        effort: None,
        transport: GoTransport::Anthropic,
        image: true,
    },
    ModelSpec {
        id: "gpt-5.6-luna",
        name: "GPT-5.6 Luna",
        reasoning: true,
        context_window: 1_050_000,
        max_tokens: 128_000,
        cost: (0.2, 1.2, 0.02, 0.25),
        effort: Some(&["none", "low", "medium", "high", "xhigh", "max"]),
        transport: GoTransport::Responses,
        image: true,
    },
    ModelSpec {
        id: "qwen3.8-flash",
        name: "Qwen3.8 Flash",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 131_072,
        cost: (0.15, 0.47, 0.016, 0.2),
        effort: Some(&["low", "medium", "xhigh"]),
        transport: GoTransport::Anthropic,
        image: true,
    },
    ModelSpec {
        id: "glm-5.2",
        name: "GLM-5.2",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 131_072,
        cost: (1.4, 4.4, 0.26, 0.0),
        effort: Some(&["high", "max"]),
        transport: GoTransport::Completions,
        image: false,
    },
    ModelSpec {
        id: "hy3",
        name: "Hy3",
        reasoning: true,
        context_window: 256_000,
        max_tokens: 128_000,
        cost: (0.14, 0.58, 0.035, 0.0),
        effort: Some(&["none", "low", "high"]),
        transport: GoTransport::Completions,
        image: false,
    },
    ModelSpec {
        id: "muse-spark-1.2-contributor",
        name: "Muse Spark 1.2 Contributor",
        reasoning: true,
        context_window: 1_048_576,
        max_tokens: 131_072,
        cost: (0.1, 0.2, 0.002, 0.0),
        effort: Some(&["minimal", "low", "medium", "high", "xhigh"]),
        transport: GoTransport::Responses,
        image: true,
    },
    ModelSpec {
        id: "claude-haiku-5-5",
        name: "Claude Haiku 5.5",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 128_000,
        cost: (0.1, 0.5, 0.01, 0.125),
        effort: Some(&["low", "medium", "high", "xhigh", "max"]),
        transport: GoTransport::Anthropic,
        image: true,
    },
    ModelSpec {
        id: "gpt-6-luna",
        name: "GPT-6 Luna",
        reasoning: true,
        context_window: 1_050_000,
        max_tokens: 128_000,
        cost: (0.1, 0.5, 0.01, 0.125),
        effort: Some(&["none", "low", "medium", "high", "xhigh", "max"]),
        transport: GoTransport::Responses,
        image: true,
    },
    ModelSpec {
        id: "deepseek-v4-pro",
        name: "DeepSeek V4 Pro (New)",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 384_000,
        cost: (0.66, 1.98, 0.022, 0.0),
        effort: Some(&["high", "max"]),
        transport: GoTransport::Completions,
        image: false,
    },
    ModelSpec {
        id: "hy4-preview",
        name: "Hy4 preview",
        reasoning: true,
        context_window: 1_024_000,
        max_tokens: 64_000,
        cost: (0.834, 2.501, 0.042, 0.0),
        effort: Some(&["none", "high"]),
        transport: GoTransport::Completions,
        image: false,
    },
    ModelSpec {
        id: "muse-spark-1.3-contributor",
        name: "Muse Spark 1.3 Contributor",
        reasoning: true,
        context_window: 1_048_576,
        max_tokens: 131_072,
        cost: (0.1, 0.2, 0.002, 0.0),
        effort: Some(&["minimal", "low", "medium", "high", "xhigh"]),
        transport: GoTransport::Responses,
        image: true,
    },
    ModelSpec {
        id: "step-5-preview-free",
        name: "Step 5 Preview Free",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 65_536,
        cost: (0.0, 0.0, 0.0, 0.0),
        effort: Some(&["low", "medium", "high"]),
        transport: GoTransport::Completions,
        image: true,
    },
    ModelSpec {
        id: "glm-5.3",
        name: "GLM-5.3",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 131_072,
        cost: (1.4, 4.4, 0.26, 0.0),
        effort: Some(&["low", "high", "max"]),
        transport: GoTransport::Completions,
        image: false,
    },
    ModelSpec {
        id: "kimi-k2.7-code",
        name: "Kimi K2.7 Code",
        reasoning: true,
        context_window: 262_144,
        max_tokens: 262_144,
        cost: (0.95, 4.0, 0.19, 0.0),
        effort: None,
        transport: GoTransport::Completions,
        image: true,
    },
    ModelSpec {
        id: "grok-4.6",
        name: "Grok 4.6",
        reasoning: true,
        context_window: 500_000,
        max_tokens: 500_000,
        cost: (2.0, 6.0, 0.5, 0.0),
        effort: Some(&["low", "medium", "high", "xhigh"]),
        transport: GoTransport::Responses,
        image: true,
    },
    ModelSpec {
        id: "qwen3.7-plus",
        name: "Qwen3.7 Plus",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 65_536,
        cost: (0.4, 1.6, 0.04, 0.5),
        effort: None,
        transport: GoTransport::Anthropic,
        image: true,
    },
    ModelSpec {
        id: "deepseek-v4-flash",
        name: "DeepSeek V4 Flash",
        reasoning: true,
        context_window: 1_000_000,
        max_tokens: 384_000,
        cost: (0.15, 0.6, 0.003, 0.0),
        effort: Some(&["low", "high", "max"]),
        transport: GoTransport::Completions,
        image: false,
    },
];

const ALL_LEVELS: [ModelThinkingLevel; 7] = [
    ModelThinkingLevel::Off,
    ModelThinkingLevel::Minimal,
    ModelThinkingLevel::Low,
    ModelThinkingLevel::Medium,
    ModelThinkingLevel::High,
    ModelThinkingLevel::Xhigh,
    ModelThinkingLevel::Max,
];

/// The catalog's effort rungs mapped onto Prime's levels. An unsupported level
/// maps to `None` (never sent); a model with no effort surface maps every level
/// to `None`, so no `reasoning_effort` rides its requests.
fn thinking_level_map(effort: Option<&[&str]>) -> ThinkingLevelMap {
    let mut map = ThinkingLevelMap::new();
    for level in ALL_LEVELS {
        let value = match effort {
            Some(values) => {
                let wire = level.wire_name();
                if level == ModelThinkingLevel::Off && values.contains(&"none") {
                    Some("none".to_string())
                } else if values.contains(&wire) {
                    Some(wire.to_string())
                } else {
                    None
                }
            }
            None => None,
        };
        map.insert(level, value);
    }
    map
}

fn compat(transport: GoTransport, effort_surface: bool) -> ModelCompat {
    match transport {
        GoTransport::Responses => {
            // The Responses protocol names the session itself; the gateway's
            // Responses lane has no long-cache-retention switch.
            ModelCompat::from_kind(CompatKind::OpenAiResponses(OpenAiResponsesCompat {
                send_session_id_header: Some(true),
                supports_long_cache_retention: None,
            }))
        }
        GoTransport::Anthropic => ModelCompat::from_kind(CompatKind::AnthropicMessages(
            AnthropicMessagesCompat::default(),
        )),
        GoTransport::Completions => ModelCompat::from_kind(CompatKind::OpenAiCompletions(
            Box::new(OpenAiCompletionsCompat {
                supports_store: Some(true),
                supports_reasoning_effort: effort_surface.then_some(true),
                supports_usage_in_streaming: Some(true),
                max_tokens_field: Some(MaxTokensField::MaxCompletionTokens),
                ..Default::default()
            }),
        )),
    }
}

/// Build a registry model from catalog fields: the embedded roster and the
/// live catalog cache share this one conversion, so both speak the same
/// transport and thinking-level surface.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_model(
    id: &str,
    name: &str,
    reasoning: bool,
    context_window: u64,
    max_tokens: u64,
    cost: (f64, f64, f64, f64),
    effort: Option<&[&str]>,
    transport: GoTransport,
    image: bool,
) -> Model {
    let mut input = vec![ModelInput::Text];
    if image {
        input.push(ModelInput::Image);
    }
    Model {
        id: id.to_string(),
        name: name.to_string(),
        api: transport.api().to_string(),
        provider: OPENCODE_GO_PROVIDER_ID.to_string(),
        base_url: transport.base_url().to_string(),
        reasoning,
        thinking_level_map: Some(thinking_level_map(effort)),
        input,
        cost: ModelCost {
            input: cost.0.into(),
            output: cost.1.into(),
            cache_read: cost.2.into(),
            cache_write: cost.3.into(),
        },
        context_window,
        max_tokens,
        max_tokens_explicit: false,
        featured: None,
        headers: None,
        compat: Some(compat(transport, effort.is_some())),
    }
}

fn model_from_spec(spec: &ModelSpec) -> Model {
    build_model(
        spec.id,
        spec.name,
        spec.reasoning,
        spec.context_window,
        spec.max_tokens,
        spec.cost,
        spec.effort,
        spec.transport,
        spec.image,
    )
}

/// The roster as registry models (provider `opencode-go`).
#[must_use]
pub fn models() -> Vec<Model> {
    ROSTER
        .iter()
        .map(model_from_spec)
        .filter(|model| !spec::HIDDEN_MODEL_IDS.contains(&model.id.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(id: &str) -> Model {
        models()
            .into_iter()
            .find(|model| model.id == id)
            .unwrap_or_else(|| panic!("{id} is in the roster"))
    }

    #[test]
    fn the_roster_is_every_live_catalog_entry() {
        let models = models();
        assert_eq!(models.len(), 31, "36 catalog entries minus 5 deprecated");
        assert!(
            models
                .iter()
                .all(|model| model.provider == OPENCODE_GO_PROVIDER_ID),
            "every roster model serves the subscription"
        );
        assert!(
            !models.iter().any(|model| model.id == "qwen3.7-max"),
            "deprecated entries are hidden"
        );
        assert_eq!(models[0].base_url, OPENCODE_GO_BASE_URL);
    }

    #[test]
    fn effort_models_map_their_rungs_and_nulls() {
        let deepseek = model("deepseek-v4.1-flash");
        let map = deepseek.thinking_level_map.as_ref().unwrap();
        assert_eq!(map[&ModelThinkingLevel::Low].as_deref(), Some("low"));
        assert_eq!(map[&ModelThinkingLevel::High].as_deref(), Some("high"));
        assert_eq!(map[&ModelThinkingLevel::Max].as_deref(), Some("max"));
        assert_eq!(
            map[&ModelThinkingLevel::Medium],
            None,
            "an unsupported rung is null"
        );
        assert_eq!(map[&ModelThinkingLevel::Off], None);
        assert_eq!(
            deepseek.compat.as_ref().unwrap().kind().unwrap(),
            CompatKind::OpenAiCompletions(Box::new(OpenAiCompletionsCompat {
                supports_store: Some(true),
                supports_reasoning_effort: Some(true),
                supports_usage_in_streaming: Some(true),
                max_tokens_field: Some(MaxTokensField::MaxCompletionTokens),
                ..Default::default()
            }))
        );
    }

    #[test]
    fn luna_maps_off_to_none_and_uses_responses() {
        let luna = model("gpt-5.6-luna");
        assert_eq!(luna.api, "openai-responses");
        let map = luna.thinking_level_map.as_ref().unwrap();
        assert_eq!(
            map[&ModelThinkingLevel::Off].as_deref(),
            Some("none"),
            "the catalog's `none` effort is the Off rung"
        );
        assert_eq!(map[&ModelThinkingLevel::Minimal], None);
        assert_eq!(map[&ModelThinkingLevel::Max].as_deref(), Some("max"));
        assert!(matches!(
            luna.compat.as_ref().unwrap().kind().unwrap(),
            CompatKind::OpenAiResponses(_)
        ));
    }

    #[test]
    fn chat_models_use_completions_and_no_effort_surface_sends_nothing() {
        let mimo = model("mimo-v2.6-pro");
        assert_eq!(mimo.api, "openai-completions");
        let map = mimo.thinking_level_map.as_ref().unwrap();
        assert!(
            map.values().all(Option::is_none),
            "a toggle-only model has no effort surface"
        );
        let compat = mimo.compat.as_ref().unwrap().kind().unwrap();
        let CompatKind::OpenAiCompletions(compat) = compat else {
            panic!("chat models carry the completions compat");
        };
        assert_eq!(compat.supports_reasoning_effort, None);
    }

    #[test]
    fn transports_select_api_and_base_url() {
        let roster = models();
        let mut anthropic: Vec<&str> = roster
            .iter()
            .filter(|model| model.api == "anthropic-messages")
            .map(|model| model.id.as_str())
            .collect();
        anthropic.sort_unstable();
        let mut expected = vec![
            "claude-haiku-5-5",
            "minimax-m3",
            "qwen3.7-plus",
            "qwen3.8-flash",
            "qwen3.8-max",
        ];
        expected.sort_unstable();
        assert_eq!(anthropic, expected, "the five anthropic transports");
        for id in expected {
            let model = model(id);
            assert_eq!(model.base_url, OPENCODE_GO_ANTHROPIC_BASE_URL, "{id}");
            assert!(matches!(
                model.compat.as_ref().unwrap().kind().unwrap(),
                CompatKind::OpenAiCompletions(_) | CompatKind::AnthropicMessages(_)
            ));
        }
        let luna = model("gpt-5.6-luna");
        assert_eq!(luna.api, "openai-responses");
        assert_eq!(luna.base_url, OPENCODE_GO_BASE_URL);
        let mimo = model("mimo-v2.6-pro");
        assert_eq!(mimo.api, "openai-completions");
        assert_eq!(mimo.base_url, OPENCODE_GO_BASE_URL);
    }

    #[test]
    fn input_and_cost_follow_the_catalog() {
        let kimi = model("kimi-k2.7-code");
        assert_eq!(kimi.input, vec![ModelInput::Text, ModelInput::Image]);
        assert_eq!(kimi.context_window, 262_144);
        assert_eq!(kimi.max_tokens, 262_144);
        assert!((kimi.cost.input.as_f64() - 0.95).abs() < 1e-9);
        assert!((kimi.cost.cache_read.as_f64() - 0.19).abs() < 1e-9);
        let glm = model("glm-5.2");
        assert_eq!(
            glm.input,
            vec![ModelInput::Text],
            "text-only models stay text"
        );
    }

    #[test]
    fn generated_spec_is_the_single_source_of_truth() {
        use crate::opencode_go_spec_generated as spec;

        assert_eq!(spec::SPEC_SHA256.len(), 64);
        assert!(spec::SPEC_SHA256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));
        assert_eq!(OPENCODE_GO_PROVIDER_ID, spec::PROVIDER_ID);
        assert_eq!(OPENCODE_GO_BASE_URL, spec::GATEWAY_OPENAI_BASE_URL);
        assert_eq!(
            OPENCODE_GO_ANTHROPIC_BASE_URL,
            spec::GATEWAY_ANTHROPIC_BASE_URL
        );
        assert_eq!(crate::opencode_go_catalog::CATALOG_URL, spec::CATALOG_URL);
        assert_eq!(
            GoTransport::from_npm(Some(spec::TRANSPORT_RULES_NPM[0])),
            GoTransport::Responses
        );
        assert_eq!(
            GoTransport::from_npm(Some(spec::TRANSPORT_RULES_NPM[1])),
            GoTransport::Anthropic
        );
        assert_eq!(
            GoTransport::from_npm(Some("@ai-sdk/google")),
            GoTransport::Completions
        );
        assert!(!spec::DEVICE_CODE_PATH.is_empty());
        assert!(models()
            .iter()
            .all(|model| !spec::HIDDEN_MODEL_IDS.contains(&model.id.as_str())));
    }
}
