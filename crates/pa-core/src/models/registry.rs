//! `ModelRegistry`: composes built-in, custom (models.json), and Prime Inference
//! catalogs; resolves request auth per provider/model.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use pa_types::ai::{
    CompatKind, Model, ModelCompat, ModelThinkingLevel, OpenAiResponsesCompat, ThinkingLevelMap,
};

use crate::auth::manager::AuthStorage;
use crate::auth::types::{AuthCredential, PRIME_INFERENCE_PROVIDER_ID};

use super::catalog_chain;
use super::custom::{apply_model_override, load_custom_models, merge_compat, CustomModelsResult};
use super::prime_inference::is_private_prime_inference_model;
use super::private_auth::{
    fetch_authorized_private_prime_inference_models, is_offline_mode_enabled,
    private_prime_authorization_fingerprint, read_private_prime_authorization_cache,
    write_private_prime_authorization_cache, PrivatePrimeAuthorizationCache,
    PRIVATE_BACKGROUND_TIMEOUT_MS, PRIVATE_MODEL_TIMEOUT_MS,
    PRIVATE_PRIME_AUTHORIZATION_CACHE_TTL_MS,
};

/// Request-auth bits a provider can configure in models.json.
#[derive(Debug, Clone, Default)]
pub struct ProviderRequestConfig {
    pub api_key: Option<String>,
    /// Ordered (`BTreeMap`): merged into request-header maps that
    /// providers iterate deterministically.
    pub headers: Option<BTreeMap<String, String>>,
    pub auth_header: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedRequestAuth {
    pub ok: bool,
    pub api_key: Option<String>,
    /// Ordered (`BTreeMap`): unordered iteration would order the
    /// composed request headers randomly.
    pub headers: Option<BTreeMap<String, String>>,
    pub error: Option<String>,
}

/// Why a `set_model` selection failed to resolve: the provider is not
/// signed in (the client offers the sign-in flow and retries) or the model
/// is not in the catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetModelSelectionError {
    /// The model exists, but its provider has no credential and none is
    /// stale: sign in, then retry the switch.
    ProviderUnauthenticated { provider: String },
    /// No such model in the catalog.
    NotFound { provider: String, model_id: String },
}

impl SetModelSelectionError {
    /// The refused provider of the sign-in variant; `None` on not-found.
    #[must_use]
    pub fn unauthenticated_provider(&self) -> Option<&str> {
        match self {
            SetModelSelectionError::ProviderUnauthenticated { provider } => Some(provider),
            SetModelSelectionError::NotFound { .. } => None,
        }
    }
}

impl std::fmt::Display for SetModelSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SetModelSelectionError::ProviderUnauthenticated { provider } => write!(
                f,
                "Provider \"{provider}\" is not signed in. Sign in to the provider (the TUI's /login command), then set the model again."
            ),
            SetModelSelectionError::NotFound { provider, model_id } => {
                write!(f, "Model not found: {provider}/{model_id}")
            }
        }
    }
}

impl std::error::Error for SetModelSelectionError {}

/// Composed model catalog with auth-aware availability.
pub struct ModelRegistry {
    pub auth: AuthStorage,
    models_json_path: Option<PathBuf>,
    models: Vec<Model>,
    load_error: Option<String>,
    provider_request_configs: HashMap<String, ProviderRequestConfig>,
    model_request_headers: HashMap<String, BTreeMap<String, String>>,
    explicit_private_ids: HashSet<String>,
    authorized_private_ids: HashSet<String>,
    authorized_private_models: Vec<Model>,
    authorized_team_id: Option<String>,
    /// The process-shared live catalog chain (`catalog_chain::catalog_for`);
    /// `resolve()` sources built-ins from it.
    catalog: std::sync::Arc<pa_models::ModelCatalog>,
}

/// Under a stored xAI subscription the provider's models switch onto the
/// openai-responses API with the subscription thinking maps and
/// `supportsLongCacheRetention: false` compat.
fn xai_subscription_model(model: &Model) -> Model {
    let mut adapted = model.clone();
    adapted.api = "openai-responses".to_string();
    adapted.base_url = "https://api.x.ai/v1".to_string();
    // TS fills only an absent map: a model that already declares its
    // levels keeps them.
    if adapted.thinking_level_map.is_none() {
        adapted.thinking_level_map = Some(xai_subscription_thinking_map(&model.id));
    }
    adapted.compat = Some(ModelCompat::from_kind(CompatKind::OpenAiResponses(
        OpenAiResponsesCompat {
            send_session_id_header: None,
            supports_long_cache_retention: Some(false),
        },
    )));
    adapted
}

/// The TS flow's explicit thinking maps for the grok models it names,
/// and the all-levels-null default for every other xAI model (reasoning
/// stays; unverified effort controls never send).
fn xai_subscription_thinking_map(model_id: &str) -> ThinkingLevelMap {
    // The value mirrors the TS table: `None` for the null mapping
    // (unsupported), the wire string for a named level; an absent level
    // keeps its default support.
    let null = || None;
    let value = |text: &str| Some(text.to_string());
    let mut map = ThinkingLevelMap::new();
    match model_id {
        "grok-4.3" => {
            map.insert(ModelThinkingLevel::Off, value("none"));
            map.insert(ModelThinkingLevel::Minimal, null());
        }
        "grok-4.5" => {
            map.insert(ModelThinkingLevel::Off, null());
            map.insert(ModelThinkingLevel::Minimal, null());
        }
        "grok-4.6" | "grok-4.7" => {
            map.insert(ModelThinkingLevel::Off, null());
            map.insert(ModelThinkingLevel::Minimal, null());
            map.insert(ModelThinkingLevel::Xhigh, value("xhigh"));
        }
        _ => {
            for level in [
                ModelThinkingLevel::Off,
                ModelThinkingLevel::Minimal,
                ModelThinkingLevel::Low,
                ModelThinkingLevel::Medium,
                ModelThinkingLevel::High,
                ModelThinkingLevel::Xhigh,
                ModelThinkingLevel::Max,
            ] {
                map.insert(level, null());
            }
        }
    }
    map
}

impl ModelRegistry {
    pub fn create(auth: AuthStorage, models_json_path: impl Into<PathBuf>) -> Self {
        Self::new(auth, Some(models_json_path.into()))
    }

    pub fn in_memory(auth: AuthStorage) -> Self {
        Self::new(auth, None)
    }

    fn new(auth: AuthStorage, models_json_path: Option<PathBuf>) -> Self {
        let catalog = catalog_chain::catalog_for(models_json_path.as_deref());
        let mut registry = Self {
            auth,
            models_json_path,
            models: Vec::new(),
            load_error: None,
            provider_request_configs: HashMap::new(),
            model_request_headers: HashMap::new(),
            explicit_private_ids: HashSet::new(),
            authorized_private_ids: HashSet::new(),
            authorized_private_models: Vec::new(),
            authorized_team_id: None,
            catalog,
        };
        registry.load_models();
        registry
    }

    /// The Prime Inference credentials for the credentialed catalog layer
    /// (Bearer + team headers, as the TS refresh sends).
    fn prime_credentials(&mut self) -> Option<pa_models::PrimeCredentials> {
        let api_key = self.auth.get_api_key(PRIME_INFERENCE_PROVIDER_ID)?;
        let team_id = self
            .auth
            .get_provider_headers(PRIME_INFERENCE_PROVIDER_ID)
            .and_then(|headers| headers.get("X-Prime-Team-ID").cloned());
        Some(pa_models::PrimeCredentials { api_key, team_id })
    }

    pub fn get_error(&self) -> Option<&str> {
        self.load_error.as_deref()
    }

    /// Built-in + custom models (auth not filtered).
    pub fn get_all(&self) -> &[Model] {
        &self.models
    }

    /// Models whose provider has configured auth, with unauthorized private
    /// Prime Inference models gated out. The per-model auth probe is answered
    /// once per provider: each probe rebuilds the auth-source candidates.
    pub fn get_available(&self) -> Vec<&Model> {
        let mut auth_by_provider: HashMap<&str, bool> = HashMap::new();
        self.models
            .iter()
            .filter(|model| {
                (!is_private_prime_inference_model(model)
                    || self.is_authorized_private_model(model))
                    && *auth_by_provider
                        .entry(model.provider.as_str())
                        .or_insert_with(|| self.has_configured_auth(model))
            })
            .collect()
    }

    /// Models `rlm.find_models` may search: auth-configured, and not on a
    /// stale or expired provider credential (probe answered once per provider).
    pub fn get_rlm_searchable_models(&self) -> Vec<&Model> {
        let mut status_by_provider: HashMap<&str, bool> = HashMap::new();
        self.get_available()
            .into_iter()
            .filter(|model| {
                *status_by_provider
                    .entry(model.provider.as_str())
                    .or_insert_with(|| {
                        let status = self.get_provider_auth_status(&model.provider);
                        status.source != Some(crate::auth::types::AuthSource::Stale)
                            && status.label.as_deref() != Some("expired")
                    })
            })
            .collect()
    }

    pub fn has_configured_auth(&self, model: &Model) -> bool {
        if self.opencode_go_slot(&model.provider).is_some() {
            return true;
        }
        self.auth.has_auth(&model.provider)
            || self.has_configured_provider_request_auth(&model.provider)
    }

    /// The provider's auth status without credential values: callers
    /// classifying unauthenticated vs stale providers read this instead of
    /// probing for keys.
    pub fn get_provider_auth_status(&self, provider: &str) -> crate::auth::types::AuthStatus {
        if let Some(slot) = self.opencode_go_slot(provider) {
            return self.auth.get_auth_status(&slot);
        }
        self.auth.get_auth_status(provider)
    }

    /// The first stored `OpenCode` Go slot: stacked logins may leave the base
    /// key absent (`opencode-go#2`, `#3`, …). `None` for other providers.
    fn opencode_go_slot(&self, provider: &str) -> Option<String> {
        if provider != pa_ai::opencode_go_models::OPENCODE_GO_PROVIDER_ID {
            return None;
        }
        crate::auth::opencode_accounts::slots(&self.auth)
            .into_iter()
            .next()
    }

    /// `set_model`'s model resolution: an available model resolves directly; a
    /// model whose provider has no credential (none stale) is the typed sign-in
    /// refusal; stale-auth providers keep the switch (the lookup never mutates
    /// stale state).
    ///
    /// # Errors
    ///
    /// `ProviderUnauthenticated` when the provider has no credential;
    /// `NotFound` when no catalog model matches.
    pub fn resolve_set_model_selection(
        &self,
        provider: &str,
        model_id: &str,
    ) -> Result<&Model, SetModelSelectionError> {
        let matches = |model: &Model| model.provider == provider && model.id == model_id;
        if let Some(model) = self
            .get_available()
            .into_iter()
            .find(|model| matches(model))
        {
            return Ok(model);
        }
        // A model can exist while its provider is not signed in — the
        // picker's discovery path. No credential at all is the sign-in refusal;
        // the unauthorized private Prime Inference model keeps the not-found
        // refusal — never a switch to an unentitled model.
        let Some(model) = self.get_all().iter().find(|model| matches(model)) else {
            return Err(SetModelSelectionError::NotFound {
                provider: provider.to_string(),
                model_id: model_id.to_string(),
            });
        };
        if self.get_provider_auth_status(provider).source
            == Some(crate::auth::types::AuthSource::Stale)
        {
            return Ok(model);
        }
        if !self.has_configured_auth(model) {
            return Err(SetModelSelectionError::ProviderUnauthenticated {
                provider: provider.to_string(),
            });
        }
        Err(SetModelSelectionError::NotFound {
            provider: provider.to_string(),
            model_id: model_id.to_string(),
        })
    }

    fn has_configured_provider_request_auth(&self, provider: &str) -> bool {
        let Some(config) = self.provider_request_configs.get(provider) else {
            return false;
        };
        if config.headers.is_some() || config.auth_header.is_some() {
            return true;
        }
        config.api_key.as_deref().is_some_and(|key| {
            crate::auth::resolve_config_value::resolve_config_value(key).is_some()
        })
    }

    /// Reload built-in + custom models and provider request configs from disk,
    /// adopting the on-disk private Prime Inference authorization cache without
    /// network access. Sync callers must call this before `get_available`.
    pub fn load_private_authorization_from_cache(&mut self) {
        let api_key = self.auth.get_api_key(PRIME_INFERENCE_PROVIDER_ID);
        let team_headers = self.auth.get_provider_headers(PRIME_INFERENCE_PROVIDER_ID);
        let team_id = team_headers
            .as_ref()
            .and_then(|headers| headers.get("X-Prime-Team-ID").cloned());
        let (Some(api_key), Some(team_id)) = (api_key, team_id) else {
            return;
        };
        let fingerprint = private_prime_authorization_fingerprint(&api_key, &team_id);
        let Some(models_json_path) = self.models_json_path.clone() else {
            return;
        };
        let Some(PrivatePrimeAuthorizationCache {
            fingerprint: cached_fingerprint,
            models,
            ..
        }) = read_private_prime_authorization_cache(&models_json_path)
        else {
            return;
        };
        if cached_fingerprint != fingerprint {
            return;
        }
        self.authorized_private_models = models;
        self.authorized_private_ids = self
            .authorized_private_models
            .iter()
            .map(|model| model.id.clone())
            .collect();
        self.authorized_team_id = Some(team_id);
        self.load_models();
    }

    pub fn refresh(&mut self) {
        self.provider_request_configs.clear();
        self.model_request_headers.clear();
        self.explicit_private_ids.clear();
        self.load_error = None;
        self.auth.reload();
        // Direct refreshes invalidate changed auth immediately but preserve
        // same-team stale recovery.
        let team_id = self
            .auth
            .get_provider_headers(PRIME_INFERENCE_PROVIDER_ID)
            .and_then(|headers| headers.get("X-Prime-Team-ID").cloned());
        let stale_status = matches!(
            self.auth
                .get_auth_status(PRIME_INFERENCE_PROVIDER_ID)
                .source,
            Some(crate::auth::types::AuthSource::Stale)
        );
        if !stale_status || team_id.is_none() || team_id != self.authorized_team_id {
            self.authorized_private_ids.clear();
            self.authorized_private_models.clear();
            self.authorized_team_id = None;
        }
        self.load_models();
    }

    fn load_models(&mut self) {
        let path = self.models_json_path.clone();
        let result = match &path {
            Some(path) => self.load_custom_models_file(path),
            None => CustomModelsResult::default(),
        };
        if let Some(error) = &result.error {
            self.load_error = Some(error.clone());
        }
        self.explicit_private_ids = result
            .models
            .iter()
            .filter(|model| is_private_prime_inference_model(model))
            .map(|model| model.id.clone())
            .collect();

        // Private models: bundled table + authorized set, deduped by id.
        let mut private_models: HashMap<String, Model> = HashMap::new();
        for model in super::private_auth::get_private_prime_inference_models() {
            private_models.insert(model.id.clone(), model);
        }
        for model in &self.authorized_private_models {
            private_models.insert(model.id.clone(), model.clone());
        }

        let credentials = self.prime_credentials();
        let mut built_in = self.load_built_in_models(&result, credentials.as_ref());
        // The OpenCode Go subscription's live catalog (cache first, embedded
        // roster fallback) is authoritative for the provider: a base-catalog
        // entry the roster no longer lists is retired. Learned-dead ids are
        // already filtered out of both sources.
        built_in
            .retain(|model| model.provider != pa_ai::opencode_go_models::OPENCODE_GO_PROVIDER_ID);
        let go_models = self
            .agent_dir()
            .map_or_else(pa_ai::opencode_go_models::models, |agent_dir| {
                pa_ai::opencode_go_catalog::models(&agent_dir)
            });
        built_in = Self::merge_custom_models(built_in, go_models);
        built_in.extend(private_models.into_values());
        self.models = Self::merge_custom_models(built_in, result.models);
        self.apply_subscription_model_adaptations();
    }

    /// Per-model base-URL/API overrides plus the xAI subscription switch,
    /// applied at load: stored credentials rewrite the affected models, and
    /// the registry reloads after every credential change.
    fn apply_subscription_model_adaptations(&mut self) {
        let copilot_credential = self
            .auth
            .get_all()
            .credential(crate::auth::GITHUB_COPILOT_PROVIDER_ID);
        if let Some(AuthCredential::Oauth {
            access,
            enterprise_url,
            ..
        }) = copilot_credential
        {
            let base_url =
                pa_ai::oauth::get_github_copilot_base_url(Some(&access), enterprise_url.as_deref());
            for model in &mut self.models {
                if model.provider == crate::auth::GITHUB_COPILOT_PROVIDER_ID {
                    model.base_url.clone_from(&base_url);
                }
            }
        }
        // The stored credential is OAuth and the store is the winning
        // source (a stale credential does not serve it).
        let xai_stored = matches!(
            self.auth.get_all().credential(crate::auth::XAI_PROVIDER_ID),
            Some(AuthCredential::Oauth { .. })
        ) && self
            .auth
            .get_auth_status(crate::auth::XAI_PROVIDER_ID)
            .source
            == Some(crate::auth::types::AuthSource::Stored);
        if xai_stored {
            for model in &mut self.models {
                if model.provider == crate::auth::XAI_PROVIDER_ID {
                    *model = xai_subscription_model(model);
                }
            }
        }
    }

    fn load_custom_models_file(&mut self, path: &Path) -> CustomModelsResult {
        if !path.exists() {
            return CustomModelsResult::default();
        }
        let Ok(content) = std::fs::read_to_string(path) else {
            return CustomModelsResult {
                error: Some(format!(
                    "Failed to load models.json\n\nFile: {}",
                    path.display()
                )),
                ..Default::default()
            };
        };
        let mut result = load_custom_models(
            &content,
            &|provider| !pa_ai::models_generated::get_models(provider).is_empty(),
            &|provider| {
                pa_ai::models_generated::get_models(provider)
                    .first()
                    .map(|model| (model.api.clone(), model.base_url.clone()))
            },
        );
        // Provider/model request config from the parsed document.
        if result.error.is_none() {
            if let Ok(config) =
                super::custom::parse_models_config(&super::custom::strip_json_comments(&content))
            {
                for (provider, provider_config) in config.providers {
                    if provider_config.api_key.is_some()
                        || provider_config.headers.is_some()
                        || provider_config.auth_header.is_some()
                    {
                        self.provider_request_configs.insert(
                            provider.clone(),
                            ProviderRequestConfig {
                                api_key: provider_config.api_key,
                                headers: provider_config.headers,
                                auth_header: provider_config.auth_header,
                            },
                        );
                    }
                    if let Some(model_overrides) = provider_config.model_overrides {
                        for (model_id, model_override) in model_overrides {
                            self.store_model_headers(
                                &provider,
                                &model_id,
                                model_override.headers.clone(),
                            );
                        }
                    }
                    if let Some(model_defs) = provider_config.models {
                        for model_def in model_defs {
                            self.store_model_headers(
                                &provider,
                                &model_def.id,
                                model_def.headers.clone(),
                            );
                        }
                    }
                }
            }
        }
        if let Some(error) = &result.error {
            // Keep built-ins only; message mirrors the TS load failure format.
            if !error.contains("models.json") {
                result.error = Some(format!("{error}\n\nFile: {}", path.display()));
            }
        }
        result
    }

    fn store_model_headers(
        &mut self,
        provider: &str,
        model_id: &str,
        headers: Option<BTreeMap<String, String>>,
    ) {
        let key = format!("{provider}:{model_id}");
        match headers {
            Some(headers) if !headers.is_empty() => {
                self.model_request_headers.insert(key, headers);
            }
            _ => {
                self.model_request_headers.remove(&key);
            }
        }
    }

    /// Built-ins through the no-cold-start chain (validated disk snapshot |
    /// bundled asset | compiled fallback), merged with the credentialed
    /// live Prime Inference snapshot for `credentials`.
    fn load_built_in_models(
        &self,
        custom: &CustomModelsResult,
        credentials: Option<&pa_models::PrimeCredentials>,
    ) -> Vec<Model> {
        self.catalog
            .resolve(credentials)
            .into_iter()
            .map(|mut model| {
                if let Some(provider_override) = custom.provider_overrides.get(&model.provider) {
                    if provider_override.base_url.is_some() {
                        model.base_url = provider_override
                            .base_url
                            .clone()
                            .unwrap_or_else(|| model.base_url.clone());
                    }
                    if provider_override.compat.is_some() {
                        model.compat =
                            merge_compat(model.compat.as_ref(), provider_override.compat.clone());
                    }
                }
                if let Some(model_override) = custom
                    .model_overrides
                    .get(&model.provider)
                    .and_then(|overrides| overrides.get(&model.id))
                {
                    model = apply_model_override(&model, model_override);
                }
                model
            })
            .collect()
    }

    /// Custom models win on provider+id conflicts.
    fn merge_custom_models(mut built_in: Vec<Model>, custom: Vec<Model>) -> Vec<Model> {
        for custom_model in custom {
            match built_in.iter().position(|model| {
                model.provider == custom_model.provider && model.id == custom_model.id
            }) {
                Some(index) => built_in[index] = custom_model,
                None => built_in.push(custom_model),
            }
        }
        built_in
    }

    /// Reload local state and refresh entitlements (live catalog + private
    /// auth). Both fetches are awaited, so the resolved catalog reflects
    /// them when the call returns.
    pub async fn refresh_available_models(&mut self) -> Vec<Model> {
        self.refresh_available_models_forced(false).await
    }

    /// [`ModelRegistry::refresh_available_models`] with the refresh trigger's
    /// gating: forced triggers (startup, auth change) skip the hourly catalog
    /// gate, picker-open keeps it; private authorization keeps its own gating.
    pub async fn refresh_available_models_with_trigger(
        &mut self,
        trigger: pa_models::RefreshTrigger,
    ) -> Vec<Model> {
        self.refresh_available_models_forced(trigger.forced()).await
    }

    /// The awaited refresh body: reload, the gated-or-forced chain fetch, the
    /// private-authorization refresh, then the auth-filtered catalog.
    async fn refresh_available_models_forced(&mut self, force: bool) -> Vec<Model> {
        let previous_ids = self.authorized_private_ids.clone();
        let previous_team = self.authorized_team_id.clone();
        let previous_models = self.authorized_private_models.clone();
        self.refresh();
        let credentials = self.prime_credentials();
        self.catalog
            .refresh_with_credentials(force, credentials.as_ref())
            .await;
        self.load_models();
        self.refresh_private_prime_inference_authorization(
            previous_ids,
            previous_team,
            previous_models,
        )
        .await;
        self.get_available().into_iter().cloned().collect()
    }

    #[allow(clippy::too_many_lines)]
    async fn refresh_private_prime_inference_authorization(
        &mut self,
        previous_ids: HashSet<String>,
        previous_team: Option<String>,
        previous_models: Vec<Model>,
    ) {
        let api_key = self.auth.get_api_key(PRIME_INFERENCE_PROVIDER_ID);
        let team_headers = self.auth.get_provider_headers(PRIME_INFERENCE_PROVIDER_ID);
        let team_id = team_headers
            .as_ref()
            .and_then(|headers| headers.get("X-Prime-Team-ID").cloned());
        let Some(api_key) = api_key else {
            return self.clear_private_authorization();
        };
        let (Some(team_headers), Some(team_id)) = (team_headers, team_id) else {
            return self.clear_private_authorization();
        };

        let fingerprint = private_prime_authorization_fingerprint(&api_key, &team_id);
        if let Some(cache_path) = self.models_json_path.clone() {
            let cached = read_private_prime_authorization_cache(&cache_path);
            if let Some(PrivatePrimeAuthorizationCache {
                fingerprint: cached_fingerprint,
                models,
                refreshed_at,
                ..
            }) = cached
            {
                if cached_fingerprint == fingerprint {
                    self.authorized_private_models.clone_from(&models);
                    self.authorized_private_ids =
                        models.iter().map(|model| model.id.clone()).collect();
                    self.authorized_team_id = Some(team_id.clone());
                    self.load_models();
                    let fresh =
                        now_millis() - refreshed_at < PRIVATE_PRIME_AUTHORIZATION_CACHE_TTL_MS;
                    if is_offline_mode_enabled() || fresh {
                        return;
                    }
                    return self
                        .background_private_refresh(
                            api_key,
                            team_headers,
                            team_id,
                            fingerprint,
                            cache_path,
                        )
                        .await;
                }
            }
        }
        if is_offline_mode_enabled() {
            return self.clear_private_authorization();
        }

        let public_ids = self.public_prime_inference_ids();
        let fetched = fetch_authorized_private_prime_inference_models(
            self.catalog.prime_inference_base_url(),
            &api_key,
            &team_headers,
            &public_ids,
            PRIVATE_MODEL_TIMEOUT_MS,
        )
        .await;
        if let Ok(authorized) = fetched {
            self.authorized_private_ids = authorized
                .models
                .iter()
                .map(|model| model.id.clone())
                .collect();
            self.authorized_private_models
                .clone_from(&authorized.models);
            self.authorized_team_id = Some(team_id);
            self.load_models();
            if let Some(cache_path) = self.models_json_path.clone() {
                write_private_prime_authorization_cache(
                    &cache_path,
                    &PrivatePrimeAuthorizationCache {
                        fingerprint,
                        items: authorized.items,
                        models: authorized.models,
                        refreshed_at: now_millis(),
                    },
                );
            }
        } else {
            // Fetch failed: keep previous state for the same team.
            self.authorized_private_ids = previous_ids;
            self.authorized_private_models = previous_models;
            self.authorized_team_id = previous_team;
            self.load_models();
        }
    }

    async fn background_private_refresh(
        &mut self,
        api_key: String,
        team_headers: HashMap<String, String>,
        team_id: String,
        fingerprint: String,
        cache_path: PathBuf,
    ) {
        let public_ids = self.public_prime_inference_ids();
        let Ok(authorized) = fetch_authorized_private_prime_inference_models(
            self.catalog.prime_inference_base_url(),
            &api_key,
            &team_headers,
            &public_ids,
            PRIVATE_BACKGROUND_TIMEOUT_MS,
        )
        .await
        else {
            return;
        };
        // Apply only if the credentials did not change while fetching.
        if private_prime_authorization_fingerprint(&api_key, &team_id) != fingerprint {
            return;
        }
        self.authorized_private_ids = authorized
            .models
            .iter()
            .map(|model| model.id.clone())
            .collect();
        self.authorized_private_models
            .clone_from(&authorized.models);
        self.authorized_team_id = Some(team_id);
        self.load_models();
        write_private_prime_authorization_cache(
            &cache_path,
            &PrivatePrimeAuthorizationCache {
                fingerprint,
                items: authorized.items,
                models: authorized.models,
                refreshed_at: now_millis(),
            },
        );
    }

    /// The public Prime Inference ids the resolved catalog serves (the
    /// live-or-offline snapshot): the private-authorization fetch skips
    /// these and reports only the account's private entitlements.
    fn public_prime_inference_ids(&self) -> HashSet<String> {
        self.models
            .iter()
            .filter(|model| {
                model.provider == PRIME_INFERENCE_PROVIDER_ID
                    && !is_private_prime_inference_model(model)
            })
            .map(|model| model.id.clone())
            .collect()
    }

    fn clear_private_authorization(&mut self) {
        self.authorized_private_ids.clear();
        self.authorized_private_models.clear();
        self.authorized_team_id = None;
        self.load_models();
    }

    fn is_authorized_private_model(&self, model: &Model) -> bool {
        self.explicit_private_ids.contains(&model.id)
            || self.authorized_private_ids.contains(&model.id)
    }

    /// `assumeAuthConfigured` validates an explicit stale-provider selection.
    pub async fn can_use_model(&mut self, model: &Model, assume_auth_configured: bool) -> bool {
        if assume_auth_configured {
            return !is_private_prime_inference_model(model)
                || self.is_authorized_private_model(model);
        }
        if !self.has_configured_auth(model) {
            return false;
        }
        if !is_private_prime_inference_model(model) {
            return true;
        }
        let available = self.refresh_available_models().await;
        available
            .iter()
            .any(|candidate| candidate.provider == model.provider && candidate.id == model.id)
    }

    /// Resolve request auth: API key + merged headers for a model.
    pub fn get_api_key_and_headers(
        &mut self,
        model: &Model,
        request_headers: Option<&BTreeMap<String, String>>,
    ) -> ResolvedRequestAuth {
        self.get_api_key_and_headers_for_session(model, request_headers, None)
    }

    /// Resolve request auth for one session: the `OpenCode` Go subscriptions pick
    /// their stacked serving account from the routing state, every other
    /// provider takes the plain lookup.
    pub fn get_api_key_and_headers_for_session(
        &mut self,
        model: &Model,
        request_headers: Option<&BTreeMap<String, String>>,
        session_id: Option<&str>,
    ) -> ResolvedRequestAuth {
        if model.provider == pa_ai::opencode_go_models::OPENCODE_GO_PROVIDER_ID {
            if let Some(resolved) =
                self.opencode_go_request_auth(model, request_headers, session_id)
            {
                return resolved;
            }
        }
        self.default_request_auth(model, request_headers)
    }

    /// The `OpenCode` Go serving account: the session pin (drain in place) or the
    /// routing choice, with the workspace header the gateway attributes requests
    /// by. `None` falls back to the plain lookup (no routing state, no slots).
    fn opencode_go_request_auth(
        &mut self,
        model: &Model,
        request_headers: Option<&BTreeMap<String, String>>,
        session_id: Option<&str>,
    ) -> Option<ResolvedRequestAuth> {
        let agent_dir = self.agent_dir()?;
        let slots = crate::auth::opencode_accounts::slots(&self.auth);
        if slots.is_empty() {
            return None;
        }
        // The first routed request needs the real usage so exhausted accounts
        // are skipped before the gateway 429s. The prime is background-only: an
        // unknown percentage proceeds and the 429/verdict path rotates.
        crate::auth::opencode_accounts::seed_org_ids(&agent_dir, &self.auth, &slots);
        crate::auth::opencode_accounts::refresh_usage_before_routing(
            &agent_dir, &self.auth, &slots,
        );
        let Some(slot) = crate::auth::opencode_accounts::resolve(&agent_dir, &slots, session_id)
        else {
            // The base key would answer a guaranteed 429: every slot is
            // exhausted or carries a fresh 401/402/403 verdict.
            return Some(ResolvedRequestAuth {
                ok: false,
                api_key: None,
                headers: None,
                error: Some(
                    "All OpenCode Go accounts are rate-limited or rejected; wait for a usage window to reset or add another account with /login."
                        .to_string(),
                ),
            });
        };
        // The serving account's workspace id comes from the stored credential
        // first (a never-served slot has no routing state yet), then the state.
        let org_id = crate::auth::opencode_accounts::slot_org_id(&self.auth, &slot).or_else(|| {
            crate::auth::opencode_accounts::load(&agent_dir)
                .accounts
                .get(&slot)
                .and_then(|account| account.org_id.clone())
        });
        // A dead grant (a failed OAuth refresh) or a failed API-key command
        // resolves to no token. Never serve the base key and let the gateway
        // answer a bare 401: fail the turn with an actionable message, and skip
        // the pin/serve bookkeeping for a slot that cannot answer.
        let Some(token) = self.auth.get_api_key(&slot) else {
            // Demote the slot so the pin `resolve` just wrote is dropped and the
            // next turn rotates to a healthy sibling instead of retrying it.
            crate::auth::opencode_accounts::record_verdict(
                &agent_dir,
                &slot,
                401,
                "no usable token; run /login",
            );
            return Some(ResolvedRequestAuth {
                ok: false,
                api_key: None,
                headers: None,
                error: Some(format!(
                    "OpenCode Go account {slot} has no usable token; run /login to sign in again."
                )),
            });
        };
        crate::auth::opencode_accounts::note_served(&agent_dir, &slot, org_id.as_deref());
        crate::auth::opencode_accounts::maybe_poll_usage(
            &agent_dir,
            &slot,
            &token,
            org_id.as_deref(),
            session_id,
        );
        let mut headers: BTreeMap<String, String> = BTreeMap::new();
        if let Some(model_headers) = &model.headers {
            headers.extend(model_headers.clone());
        }
        if let Some(provider_headers) = self
            .provider_request_configs
            .get(&model.provider)
            .and_then(|config| config.headers.clone())
        {
            headers.extend(provider_headers);
        }
        if let Some(model_headers) = self
            .model_request_headers
            .get(&format!("{}:{}", model.provider, model.id))
            .cloned()
        {
            headers.extend(model_headers);
        }
        if let Some(org_id) = org_id {
            headers.insert("x-opencode-org-id".to_string(), org_id);
        }
        if let Some(request_headers) = request_headers {
            headers.extend(request_headers.clone());
        }
        Some(ResolvedRequestAuth {
            ok: true,
            api_key: Some(token),
            headers: (!headers.is_empty()).then_some(headers),
            error: None,
        })
    }

    /// Record a 401/402/403 verdict on the session's serving `OpenCode` Go
    /// account (dropping its pins) so the next turn rotates to a healthy
    /// sibling. Non-verdict statuses are ignored.
    pub fn record_inference_verdict(&self, session_id: Option<&str>, status: u16, message: &str) {
        if !crate::auth::opencode_accounts::is_verdict_status(status) {
            return;
        }
        let Some(agent_dir) = self.agent_dir() else {
            return;
        };
        let slots = crate::auth::opencode_accounts::slots(&self.auth);
        if slots.is_empty() {
            return;
        }
        let state = crate::auth::opencode_accounts::load(&agent_dir);
        let now = crate::auth::opencode_accounts::epoch_ms();
        let slot =
            crate::auth::opencode_accounts::session_pin(&state, session_id, now).or_else(|| {
                crate::auth::opencode_accounts::choose_slot(
                    &state,
                    &slots,
                    None,
                    crate::auth::opencode_accounts::spill_fraction(),
                    now,
                )
            });
        let Some(slot) = slot else {
            return;
        };
        crate::auth::opencode_accounts::record_verdict(&agent_dir, &slot, status, message);
    }

    /// Record a settled 429 on the session's serving `OpenCode` Go account
    /// (dropping its pins) so the next turn rotates immediately instead of
    /// waiting for the usage poll.
    pub fn record_rate_limit(&self, session_id: Option<&str>, retry_after_ms: Option<u64>) {
        let Some(agent_dir) = self.agent_dir() else {
            return;
        };
        let slots = crate::auth::opencode_accounts::slots(&self.auth);
        if slots.is_empty() {
            return;
        }
        let state = crate::auth::opencode_accounts::load(&agent_dir);
        let now = crate::auth::opencode_accounts::epoch_ms();
        let slot =
            crate::auth::opencode_accounts::session_pin(&state, session_id, now).or_else(|| {
                crate::auth::opencode_accounts::choose_slot(
                    &state,
                    &slots,
                    None,
                    crate::auth::opencode_accounts::spill_fraction(),
                    now,
                )
            });
        let Some(slot) = slot else {
            return;
        };
        crate::auth::opencode_accounts::record_rate_limit(&agent_dir, &slot, retry_after_ms);
    }

    /// The agent directory (`models.json`'s parent) the routing state lives in.
    fn agent_dir(&self) -> Option<PathBuf> {
        self.models_json_path
            .as_deref()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
    }

    fn default_request_auth(
        &mut self,
        model: &Model,
        request_headers: Option<&BTreeMap<String, String>>,
    ) -> ResolvedRequestAuth {
        let stored = self
            .auth
            .get_api_key_with_source_token(&model.provider, false);
        let mut api_key = stored.api_key;
        let provider_config = self.provider_request_configs.get(&model.provider).cloned();
        if api_key.is_none() {
            if let Some(config) = &provider_config {
                if let Some(configured) = &config.api_key {
                    if let Some(resolved) =
                        crate::auth::resolve_config_value::resolve_config_value(configured)
                    {
                        api_key = Some(resolved);
                    }
                }
            }
        }
        let provider_headers = provider_config
            .as_ref()
            .and_then(|config| config.headers.clone());
        let auth_storage_headers = self.auth.get_provider_headers(&model.provider);
        let model_request_key = format!("{}:{}", model.provider, model.id);
        let model_request_headers = self.model_request_headers.get(&model_request_key).cloned();

        let mut headers: BTreeMap<String, String> = BTreeMap::new();
        if let Some(model_headers) = &model.headers {
            headers.extend(model_headers.clone());
        }
        if let Some(auth_storage_headers) = auth_storage_headers {
            headers.extend(auth_storage_headers);
        }
        if let Some(provider_headers) = provider_headers {
            headers.extend(provider_headers);
        }
        if let Some(model_request_headers) = model_request_headers {
            headers.extend(model_request_headers);
        }
        if provider_config
            .as_ref()
            .and_then(|config| config.auth_header)
            .unwrap_or(false)
        {
            let Some(api_key) = &api_key else {
                return ResolvedRequestAuth {
                    ok: false,
                    error: Some(format!("No API key found for \"{}\"", model.provider)),
                    ..Default::default()
                };
            };
            headers.insert("Authorization".to_string(), format!("Bearer {api_key}"));
        }
        if let Some(request_headers) = request_headers {
            headers.extend(request_headers.clone());
        }
        ResolvedRequestAuth {
            ok: true,
            api_key,
            headers: (!headers.is_empty()).then_some(headers),
            error: None,
        }
    }
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}

#[cfg(test)]
mod tests;
