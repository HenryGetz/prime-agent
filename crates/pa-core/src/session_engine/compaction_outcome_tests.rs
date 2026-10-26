use super::*;
use crate::session_engine::messages::{
    convert_to_llm, create_compaction_outcome_message, CompactionOutcomeKind,
    CompactionOutcomeReason,
};
use pa_agent::agent::{AgentInitialState, AgentOptions};
use pa_agent::scripted::ScriptedProvider;
use pa_types::ai::AssistantMessage;

fn test_model() -> pa_agent::types::Model {
    serde_json::from_value(serde_json::json!({
        "id": "m", "name": "m", "api": "openai-completions", "provider": "test",
        "baseUrl": "http://localhost", "reasoning": false, "input": ["text"],
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
        "contextWindow": 1000, "maxTokens": 100
    }))
    .unwrap()
}

async fn scripted_session_over(session: SessionManager) -> AgentSession {
    let provider = Arc::new(ScriptedProvider::new(test_model()));
    let options = AgentOptions {
        initial_state: AgentInitialState {
            model: Some(test_model()),
            ..Default::default()
        },
        stream_fn: Some(provider.stream_fn()),
        ..Default::default()
    };
    let agent = Agent::new(options);
    AgentSession::new(Arc::new(agent), session, vec![])
        .await
        .unwrap()
}

fn seeded_assistant() -> SessionAgentMessage {
    SessionAgentMessage::Assistant(AssistantMessage {
        content: vec![],
        api: "openai-completions".to_string(),
        provider: "test".to_string(),
        model: "m".to_string(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: pa_types::ai::Usage::default(),
        stop_reason: pa_types::ai::StopReason::Stop,
        stop_reason_raw: None,
        error_message: None,
        timestamp: 0,
        rest: serde_json::Map::default(),
    })
}

#[test]
fn outcome_row_shape_matches_ts() {
    let row = create_compaction_outcome_message(
        "Auto-compaction skipped: Session is too short to compact — try again once it grows",
        CompactionOutcomeReason::Threshold,
        CompactionOutcomeKind::Skipped,
    );
    assert_eq!(row.custom_type, "compaction_outcome");
    assert_eq!(
        row.content.text(),
        "Auto-compaction skipped: Session is too short to compact — try again once it grows"
    );
    assert!(row.display);
    assert_eq!(
        row.details,
        Some(serde_json::json!({ "reason": "threshold", "outcome": "skipped" }))
    );
    assert!(row.timestamp > 0);
    let wire = serde_json::to_value(SessionAgentMessage::Custom(row)).unwrap();
    assert_eq!(wire["role"], "custom");
    assert_eq!(wire["customType"], "compaction_outcome");
}

/// The row lands in the entry chain and on the live loop context, a rebuild
/// keeps it, and the LLM conversion drops it — the model never sees the disclosure.
#[tokio::test]
async fn record_appends_row_to_entries_and_live_context_but_not_llm_input() {
    let tmp = tempfile::tempdir().unwrap();
    let session = scripted_session_over(SessionManager::in_memory(tmp.path())).await;
    let row = session
        .record_compaction_outcome(
            CompactionOutcomeReason::Requested,
            CompactionOutcomeKind::Failed,
            "Requested compaction failed: Summarization failed",
        )
        .await
        .unwrap();
    // The entry chain owns the row (context rebuilds read it).
    let entries = session.entries().await;
    let outcome_entries: Vec<_> = entries
        .iter()
        .filter_map(|entry| match entry {
            FileEntry::CustomMessage { payload, .. }
                if payload.custom_type == "compaction_outcome" =>
            {
                Some(payload.clone())
            }
            _ => None,
        })
        .collect();
    assert_eq!(outcome_entries.len(), 1, "one durable outcome row");
    assert_eq!(
        outcome_entries[0].content.text(),
        "Requested compaction failed: Summarization failed"
    );
    assert_eq!(
        outcome_entries[0].details,
        Some(serde_json::json!({ "reason": "requested", "outcome": "failed" }))
    );
    assert!(outcome_entries[0].display);
    let state = session.agent().state().await;
    assert!(
        matches!(
            state.messages.last(),
            Some(AgentMessage::Custom(custom)) if custom.role == "custom"
        ),
        "the live context carries the outcome row"
    );
    let guard = session.session.lock().await;
    let context =
        crate::session::build_session_context(guard.get_all_entries(), guard.get_leaf_id());
    drop(guard);
    assert!(
        context
            .messages
            .iter()
            .any(|message| matches!(message, SessionAgentMessage::Custom(custom) if custom.custom_type == "compaction_outcome")),
        "the rebuilt context keeps the outcome row"
    );
    assert!(convert_to_llm(std::slice::from_ref(&SessionAgentMessage::Custom(row))).is_empty());
}

/// The disclosure survives a failed disk write: the entry chain keeps
/// the row in memory, so a rebuild never drops it.
#[tokio::test]
async fn record_survives_a_failed_disk_write() {
    let tmp = tempfile::tempdir().unwrap();
    let sessions = tmp.path().join("sessions");
    std::fs::create_dir_all(&sessions).unwrap();
    let mut manager = SessionManager::persisted(tmp.path(), &sessions);
    manager.append_message(seeded_assistant()).unwrap();
    let file = manager.get_session_file().unwrap().to_path_buf();
    assert!(file.exists(), "the session file materialized");
    // Replace the session file with a directory at the same path: every
    // disk write path fails, even for root.
    std::fs::remove_file(&file).unwrap();
    std::fs::create_dir(&file).unwrap();
    let session = scripted_session_over(manager).await;
    session
        .record_compaction_outcome(
            CompactionOutcomeReason::Threshold,
            CompactionOutcomeKind::Skipped,
            "Auto-compaction skipped: Already compacted",
        )
        .await
        .unwrap();
    let entries = session.entries().await;
    assert!(
        entries.iter().any(
            |entry| matches!(entry, FileEntry::CustomMessage { payload, .. }
            if payload.custom_type == "compaction_outcome")
        ),
        "the outcome row stays in the entry chain after the failed write"
    );
    let guard = session.session.lock().await;
    let context =
        crate::session::build_session_context(guard.get_all_entries(), guard.get_leaf_id());
    drop(guard);
    assert!(
        context
            .messages
            .iter()
            .any(|message| matches!(message, SessionAgentMessage::Custom(custom) if custom.custom_type == "compaction_outcome")),
        "a rebuild cannot drop the disclosure"
    );
}

/// The subscriber arm never fails the run for a write error: the session
/// retains the row and the error only logs.
#[tokio::test]
async fn message_end_persist_failure_retains_the_row_and_swallows() {
    let tmp = tempfile::tempdir().unwrap();
    let sessions = tmp.path().join("sessions");
    std::fs::create_dir_all(&sessions).unwrap();
    let mut manager = SessionManager::persisted(tmp.path(), &sessions);
    manager.append_message(seeded_assistant()).unwrap();
    let file = manager.get_session_file().unwrap().to_path_buf();
    std::fs::remove_file(&file).unwrap();
    std::fs::create_dir(&file).unwrap();
    let session = scripted_session_over(manager).await;
    let before = session
        .session
        .lock()
        .await
        .get_all_entries()
        .to_vec()
        .len();
    persist_event(
        &session.session,
        AgentEvent::MessageEnd {
            message: AgentMessage::user("retained after the failed write"),
        },
    )
    .await
    .expect("a failed disk write must not fail the event queue");
    let guard = session.session.lock().await;
    let entries = guard.get_all_entries().to_vec();
    drop(guard);
    assert_eq!(entries.len(), before + 1, "the row stays live-indexed");
    assert!(
        !entries
            .iter()
            .any(|entry| matches!(entry, FileEntry::Message {
                message: SessionAgentMessage::Assistant(assistant),
                ..
            } if assistant.error_message.is_some())),
        "no phantom error row for a persistence failure"
    );
    let context = crate::session::build_session_context(
        &entries,
        entries
            .last()
            .and_then(|entry| entry.id().map(str::to_owned))
            .as_deref(),
    );
    assert!(
        serde_json::to_string(&context.messages)
            .unwrap()
            .contains("retained after the failed write"),
        "a context rebuild keeps the retained row"
    );
}

fn session_ai_model() -> pa_types::ai::Model {
    serde_json::from_value(serde_json::json!({
        "id": "m", "name": "m", "api": "openai-completions", "provider": "test",
        "baseUrl": "http://localhost", "reasoning": false, "input": ["text"],
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
        "contextWindow": 100_000, "maxTokens": 8000
    }))
    .unwrap()
}

/// A refiner seam reply: the text with the given stop reason (the
/// truncation class stops at `Length`).
fn refiner_reply(text: &str, stop_reason: pa_types::ai::StopReason) -> AssistantMessage {
    AssistantMessage {
        content: vec![pa_types::ai::AssistantContentBlock::Text(
            pa_types::ai::TextContent {
                text: text.to_string(),
                text_signature: None,
                rest: serde_json::Map::default(),
            },
        )],
        api: "openai-completions".to_string(),
        provider: "test".to_string(),
        model: "m".to_string(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: pa_types::ai::Usage::default(),
        stop_reason,
        stop_reason_raw: None,
        error_message: None,
        timestamp: 0,
        rest: serde_json::Map::default(),
    }
}

fn refine_assistant_row(error: bool) -> SessionAgentMessage {
    SessionAgentMessage::Assistant(pa_types::ai::AssistantMessage {
        content: vec![pa_types::ai::AssistantContentBlock::Text(
            pa_types::ai::TextContent {
                text: if error {
                    "provider failed".to_string()
                } else {
                    "did the thing".to_string()
                },
                text_signature: None,
                rest: serde_json::Map::default(),
            },
        )],
        api: "openai-completions".to_string(),
        provider: "test".to_string(),
        model: "m".to_string(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: pa_types::ai::Usage::default(),
        stop_reason: if error {
            pa_types::ai::StopReason::Error
        } else {
            pa_types::ai::StopReason::Stop
        },
        stop_reason_raw: None,
        error_message: if error {
            Some("provider failed".to_string())
        } else {
            None
        },
        timestamp: 0,
        rest: serde_json::Map::default(),
    })
}

fn refine_plan_call(plan: &'static str) -> crate::refinement::executor::RefinerFn {
    Box::new(move |_model, _system, _prompt| {
        Box::pin(async move {
            Ok(pa_types::ai::AssistantMessage {
                content: vec![pa_types::ai::AssistantContentBlock::Text(
                    pa_types::ai::TextContent {
                        text: plan.to_string(),
                        text_signature: None,
                        rest: serde_json::Map::default(),
                    },
                )],
                api: "openai-completions".to_string(),
                provider: "test".to_string(),
                model: "m".to_string(),
                response_model: None,
                response_id: None,
                diagnostics: None,
                usage: pa_types::ai::Usage::default(),
                stop_reason: pa_types::ai::StopReason::Stop,
                stop_reason_raw: None,
                error_message: None,
                timestamp: 0,
                rest: serde_json::Map::default(),
            })
        })
    })
}

const APPLIED_PLAN: &str = r#"{"summary":"note it","rationale":"repeated","expectedOutcome":"recall","edits":[{"action":"create","kind":"memory","id":"m1","title":"Tactic","content":"Use tactic A"}]}"#;
const EMPTY_PLAN: &str = r#"{"summary":"bench","edits":[]}"#;

/// A persisted session over two seeded rows, with the live loop context
/// built the way the resume path builds it (one rebuild).
async fn refine_test_session() -> (AgentSession, tempfile::TempDir) {
    let provider = Arc::new(ScriptedProvider::new(test_model()));
    let options = AgentOptions {
        initial_state: AgentInitialState {
            model: Some(test_model()),
            ..Default::default()
        },
        stream_fn: Some(provider.stream_fn()),
        ..Default::default()
    };
    let agent = Agent::new(options);
    let tmp = tempfile::tempdir().unwrap();
    let session_dir = tmp.path().join("session");
    std::fs::create_dir_all(&session_dir).unwrap();
    let mut manager = SessionManager::in_memory(tmp.path());
    manager.materialize_session_file(Some(session_dir));
    manager
        .append_message(SessionAgentMessage::User(pa_types::ai::UserMessage {
            content: pa_types::ai::UserContent::Text("do a thing twice".to_string()),
            timestamp: 0,
            rest: serde_json::Map::default(),
        }))
        .unwrap();
    manager.append_message(refine_assistant_row(false)).unwrap();
    let session = AgentSession::new(Arc::new(agent), manager, vec![])
        .await
        .unwrap();
    let live = {
        let persistence = session.shared_persistence();
        let manager = persistence.lock().await;
        rebuilt_loop_messages(
            crate::session_engine::compact_session::rebuilt_context_after_compaction(&manager),
        )
    };
    session.agent().set_messages(live).await;
    (session, tmp)
}

/// The pre-change mechanism, verbatim: the full active-context rebuild
/// plus the shared-shape conversion.
async fn full_rebuild_reference(session: &AgentSession) -> Vec<AgentMessage> {
    let persistence = session.shared_persistence();
    let manager = persistence.lock().await;
    rebuilt_loop_messages(
        crate::session_engine::compact_session::rebuilt_context_after_compaction(&manager),
    )
}

fn is_error_assistant(message: &AgentMessage) -> bool {
    matches!(
        message,
        AgentMessage::Standard(pa_agent::types::Message::Assistant(assistant))
            if assistant.stop_reason == pa_agent::types::StopReason::Error
    )
}

fn custom_types(messages: &[AgentMessage]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Custom(custom) => custom
                .payload
                .get("customType")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            AgentMessage::Standard(_) => None,
        })
        .collect()
}

#[tokio::test]
async fn refine_pushed_live_context_matches_a_full_rebuild_byte_for_byte() {
    let (session, tmp) = refine_test_session().await;
    let global_dir = tmp.path().join("harness");
    let result = session
        .refine_with_refiner(
            &refine::RefineOptions::default(),
            refine::RefinementSource::User,
            &session_ai_model(),
            refine_plan_call(APPLIED_PLAN),
            global_dir,
        )
        .await
        .unwrap();
    assert!(result.applied_edits.iter().any(|edit| edit.applied));
    let candidate = session.agent().state().await.messages;
    let reference = full_rebuild_reference(&session).await;
    assert_eq!(
        serde_json::to_value(&candidate).unwrap(),
        serde_json::to_value(&reference).unwrap(),
        "the pushed live context is byte-identical to the full rebuild"
    );
    assert_eq!(candidate.len(), 4);
    assert_eq!(
        custom_types(&candidate[candidate.len() - 2..]),
        vec!["refinement_outcome", "refinement_notice"]
    );
    assert!(custom_types(&candidate)
        .iter()
        .all(|custom_type| custom_type != "prime-agent.refinement"));
}

#[tokio::test]
async fn refine_with_an_empty_plan_pushes_only_the_outcome_row() {
    let (session, tmp) = refine_test_session().await;
    let global_dir = tmp.path().join("harness");
    let result = session
        .refine_with_refiner(
            &refine::RefineOptions::default(),
            refine::RefinementSource::User,
            &session_ai_model(),
            refine_plan_call(EMPTY_PLAN),
            global_dir,
        )
        .await
        .unwrap();
    assert!(result.applied_edits.iter().all(|edit| !edit.applied));
    let candidate = session.agent().state().await.messages;
    let reference = full_rebuild_reference(&session).await;
    assert_eq!(
        serde_json::to_value(&candidate).unwrap(),
        serde_json::to_value(&reference).unwrap(),
        "the pushed live context is byte-identical to the full rebuild"
    );
    assert_eq!(candidate.len(), 3, "only the outcome row is pushed");
    assert_eq!(
        custom_types(&candidate[candidate.len() - 1..]),
        vec!["refinement_outcome"],
        "no notice row without an applied edit"
    );
}

#[tokio::test]
async fn sequential_refines_push_exactly_their_own_rows() {
    // The second run must select exactly its own rows (by id), never
    // the first run's.
    let (session, tmp) = refine_test_session().await;
    let global_dir = tmp.path().join("harness");
    let first = r#"{"summary":"one","edits":[{"action":"create","kind":"memory","id":"m1","title":"A","content":"a"}]}"#;
    let second = r#"{"summary":"two","edits":[{"action":"create","kind":"memory","id":"m2","title":"B","content":"b"}]}"#;
    session
        .refine_with_refiner(
            &refine::RefineOptions::default(),
            refine::RefinementSource::User,
            &session_ai_model(),
            refine_plan_call(first),
            global_dir.clone(),
        )
        .await
        .unwrap();
    session
        .refine_with_refiner(
            &refine::RefineOptions::default(),
            refine::RefinementSource::User,
            &session_ai_model(),
            refine_plan_call(second),
            global_dir,
        )
        .await
        .unwrap();
    let candidate = session.agent().state().await.messages;
    let reference = full_rebuild_reference(&session).await;
    assert_eq!(
        serde_json::to_value(&candidate).unwrap(),
        serde_json::to_value(&reference).unwrap(),
        "two sequential refines hold exactly one copy of each appended row"
    );
    assert_eq!(candidate.len(), 6);
    assert_eq!(
        custom_types(&candidate[candidate.len() - 4..]),
        vec![
            "refinement_outcome",
            "refinement_notice",
            "refinement_outcome",
            "refinement_notice"
        ]
    );
}

/// A plan whose one edit creates a factory entry (a stored state-machine
/// workflow spec): valid shape, so only the opt-in gate decides it.
const FACTORY_PLAN: &str = r#"{"summary":"sweep","edits":[{"action":"create","kind":"factory","id":"sweep","title":"Factory","content":"Sweep review.","arguments":{"machine":{"states":[{"id":"collect","entry":true,"subagent":"worker"}],"transitions":[]}}}]}"#;

#[tokio::test]
async fn refine_factory_edits_follow_the_agent_dir_opt_in_setting() {
    // The host /refine flow reads `factory.enabled` live from the session's
    // agent dir on every run — the same settings.json the kernel-side
    // factory gate reads — so a refinement cannot author factories the
    // user has not opted into. Without a wired agent dir (or with the
    // setting off), the factory create refuses with the one exact
    // disabled message; opted in, the same proposal applies.
    let (mut session, tmp) = refine_test_session().await;
    let global_dir = tmp.path().join("harness");
    let refused = session
        .refine_with_refiner(
            &refine::RefineOptions::default(),
            refine::RefinementSource::User,
            &session_ai_model(),
            refine_plan_call(FACTORY_PLAN),
            global_dir.clone(),
        )
        .await
        .unwrap();
    assert!(!refused.applied_edits[0].applied);
    assert_eq!(
        refused.applied_edits[0].error.as_deref(),
        Some(crate::refinement::FACTORY_DISABLED_MESSAGE)
    );
    // The user opts in: the next run of the same proposal applies.
    let agent_dir = tmp.path().join("agent");
    std::fs::create_dir_all(&agent_dir).unwrap();
    std::fs::write(
        agent_dir.join("settings.json"),
        r#"{"factory": {"enabled": true}}"#,
    )
    .unwrap();
    session.set_agent_dir(agent_dir);
    let applied = session
        .refine_with_refiner(
            &refine::RefineOptions::default(),
            refine::RefinementSource::User,
            &session_ai_model(),
            refine_plan_call(FACTORY_PLAN),
            global_dir,
        )
        .await
        .unwrap();
    assert!(applied.applied_edits[0].applied);
    assert!(applied.applied_edits[0].error.is_none());
}

#[tokio::test]
async fn refine_keeps_the_retry_drop_instead_of_resurrecting_the_error_row() {
    let (session, tmp) = refine_test_session().await;
    // The retry arm's sanctioned live/durable divergence: the failed turn's
    // error assistant row is durable, then dropped from the live context.
    {
        let persistence = session.shared_persistence();
        let mut manager = persistence.lock().await;
        manager.append_message(refine_assistant_row(true)).unwrap();
    }
    let rebuilt = {
        let persistence = session.shared_persistence();
        let manager = persistence.lock().await;
        crate::session_engine::compact_session::rebuilt_context_after_compaction(&manager)
    };
    session
        .agent()
        .set_messages(rebuilt_loop_messages(rebuilt))
        .await;
    session
        .drop_trailing_assistant(TrailingAssistantFilter::ErrorOnly)
        .await;
    let pre_live = session.agent().state().await.messages;
    assert_eq!(
        pre_live.len(),
        2,
        "the error row is dropped from the live context"
    );
    let global_dir = tmp.path().join("harness");
    session
        .refine_with_refiner(
            &refine::RefineOptions::default(),
            refine::RefinementSource::User,
            &session_ai_model(),
            refine_plan_call(APPLIED_PLAN),
            global_dir,
        )
        .await
        .unwrap();
    // The push preserves the live list: the dropped error row stays
    // out of the live context.
    let live = session.agent().state().await.messages;
    assert_eq!(live.len(), 4);
    assert!(
        !live.iter().any(is_error_assistant),
        "the retry-dropped error row is not resurrected"
    );
    assert_eq!(
        custom_types(&live[live.len() - 2..]),
        vec!["refinement_outcome", "refinement_notice"]
    );
    // The deliberate divergence: the full rebuild WOULD resurrect the
    // row (the pre-change mechanism's behavior).
    let reference = full_rebuild_reference(&session).await;
    assert_eq!(reference.len(), 5);
    assert!(
        reference.iter().any(is_error_assistant),
        "the reference rebuild resurrects the dropped row; the push does not"
    );
}

#[tokio::test]
async fn refinement_and_auto_review_preserve_conversation_identity() {
    let (session, tmp) = refine_test_session().await;
    let expected = session.session_id().await;
    let registration =
        pa_ai::faux::register_faux_provider(pa_ai::faux::RegisterFauxProviderOptions::default());
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let step = |text: &'static str| {
        let observed = Arc::clone(&observed);
        pa_ai::faux::FauxResponseStep::Factory(Arc::new(move |_, options, _, _| {
            observed.lock().unwrap().push((
                options.and_then(|o| o.session_id.clone()),
                options.and_then(|o| o.transport),
            ));
            Ok(pa_ai::faux::faux_assistant_text_message(
                text,
                pa_ai::faux::FauxAssistantMessageOptions::default(),
            ))
        }))
    };
    registration.set_responses(vec![
        step(EMPTY_PLAN),
        step(r#"{"shouldRefine":false,"rationale":"fixture"}"#),
    ]);
    let model = registration.get_model();
    let global = tmp.path().join("harness");
    session
        .refine(
            &refine::RefineOptions::default(),
            refine::RefinementSource::User,
            &model,
            None,
            global.clone(),
        )
        .await
        .unwrap();
    session
        .review_compact_auto_refine(&model, None, &global, 5, 0)
        .await
        .unwrap();
    assert_eq!(
        *observed.lock().unwrap(),
        vec![(Some(expected), Some(pa_ai::types::Transport::Sse)); 2]
    );
}

/// A faux refine model with the given catalog output budget and context
/// window.
fn register_refine_model(
    max_tokens: u64,
    context_window: u64,
) -> pa_ai::faux::FauxProviderRegistration {
    pa_ai::faux::register_faux_provider(pa_ai::faux::RegisterFauxProviderOptions {
        models: Some(vec![pa_ai::faux::FauxModelDefinition {
            id: "refine-m".to_string(),
            name: Some("Refine Model".to_string()),
            reasoning: Some(true),
            input: Some(vec![pa_types::ai::ModelInput::Text]),
            cost: None,
            context_window: Some(context_window),
            max_tokens: Some(max_tokens),
        }]),
        ..Default::default()
    })
}

/// A faux step replying with `message` while recording the request's wire
/// output budget.
fn budget_recording_step(
    budgets: &Arc<std::sync::Mutex<Vec<Option<u64>>>>,
    message: pa_types::ai::AssistantMessage,
) -> pa_ai::faux::FauxResponseStep {
    let budgets = Arc::clone(budgets);
    pa_ai::faux::FauxResponseStep::Factory(Arc::new(move |_, options, _, _| {
        budgets
            .lock()
            .unwrap()
            .push(options.and_then(|options| options.max_tokens));
        Ok(message.clone())
    }))
}

/// A refiner reply cut by the output budget (the model burned it on
/// reasoning and stopped at the length limit) retries once through the
/// session refine path, with the refinement budget capped at the model's
/// catalog maximum output.
#[tokio::test]
async fn a_budget_truncated_refiner_reply_retries_once_and_lands() {
    let (session, tmp) = refine_test_session().await;
    let registration = register_refine_model(8_192, 128_000);
    let budgets: Arc<std::sync::Mutex<Vec<Option<u64>>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    registration.set_responses(vec![
        budget_recording_step(
            &budgets,
            pa_ai::faux::faux_assistant_text_message(
                r#"{"summary":"cut","edits":[{"action":"cre"#,
                pa_ai::faux::FauxAssistantMessageOptions {
                    stop_reason: Some(pa_types::ai::StopReason::Length),
                    ..Default::default()
                },
            ),
        ),
        budget_recording_step(
            &budgets,
            pa_ai::faux::faux_assistant_text_message(
                r#"{"summary":"one edit","edits":[{"action":"create","kind":"memory","id":"m9","title":"t","content":"c"}]}"#,
                pa_ai::faux::FauxAssistantMessageOptions::default(),
            ),
        ),
    ]);
    let model = registration.get_model();
    let global = tmp.path().join("harness");
    let result = session
        .refine(
            &refine::RefineOptions::default(),
            refine::RefinementSource::User,
            &model,
            None,
            global,
        )
        .await
        .unwrap();
    assert_eq!(result.summary, "one edit");
    assert!(result.applied_edits[0].applied);
    assert_eq!(
        *budgets.lock().unwrap(),
        vec![Some(8_192), Some(8_192)],
        "both requests stay at the 8k catalog maximum output; the floor never raises past it"
    );
}

/// Some providers cut the reply without flagging the length stop; the
/// mid-JSON diagnosis alone must drive the retry.
#[tokio::test]
async fn a_mid_json_cut_without_a_length_stop_retries_once_and_lands() {
    let (session, tmp) = refine_test_session().await;
    let registration = register_refine_model(8_192, 128_000);
    let budgets: Arc<std::sync::Mutex<Vec<Option<u64>>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    registration.set_responses(vec![
        budget_recording_step(
            &budgets,
            pa_ai::faux::faux_assistant_text_message(
                r#"{"summary":"cut","edits":[{"action":"cre"#,
                pa_ai::faux::FauxAssistantMessageOptions::default(),
            ),
        ),
        budget_recording_step(
            &budgets,
            pa_ai::faux::faux_assistant_text_message(
                r#"{"summary":"one edit","edits":[{"action":"create","kind":"memory","id":"m9","title":"t","content":"c"}]}"#,
                pa_ai::faux::FauxAssistantMessageOptions::default(),
            ),
        ),
    ]);
    let model = registration.get_model();
    let global = tmp.path().join("harness");
    let result = session
        .refine(
            &refine::RefineOptions::default(),
            refine::RefinementSource::User,
            &model,
            None,
            global,
        )
        .await
        .unwrap();
    assert_eq!(result.summary, "one edit");
    assert!(result.applied_edits[0].applied);
    assert_eq!(
        *budgets.lock().unwrap(),
        vec![Some(8_192), Some(8_192)],
        "the mid-JSON diagnosis alone retried, both requests capped at the 8k catalog maximum output"
    );
}

/// A parse failure outside the budget class (balanced garbage) surfaces
/// immediately; only the budget-exhausted class retries.
#[tokio::test]
async fn a_non_budget_parse_failure_fails_without_a_retry() {
    let (session, tmp) = refine_test_session().await;
    let registration = register_refine_model(8_192, 128_000);
    let budgets: Arc<std::sync::Mutex<Vec<Option<u64>>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    registration.set_responses(vec![budget_recording_step(
        &budgets,
        pa_ai::faux::faux_assistant_text_message(
            "not json",
            pa_ai::faux::FauxAssistantMessageOptions::default(),
        ),
    )]);
    let model = registration.get_model();
    let global = tmp.path().join("harness");
    let error = session
        .refine(
            &refine::RefineOptions::default(),
            refine::RefinementSource::User,
            &model,
            None,
            global,
        )
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "Refiner did not return a JSON object");
    assert_eq!(budgets.lock().unwrap().len(), 1, "garbage never retries");
}

/// A reply truncated on both attempts surfaces the truncation error.
#[tokio::test]
async fn a_twice_truncated_reply_surfaces_the_truncation_error() {
    let (session, tmp) = refine_test_session().await;
    let registration = register_refine_model(8_192, 128_000);
    let budgets: Arc<std::sync::Mutex<Vec<Option<u64>>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let truncated = || {
        pa_ai::faux::faux_assistant_text_message(
            r#"{"summary":"cut","edits":[{"action":"cre"#,
            pa_ai::faux::FauxAssistantMessageOptions {
                stop_reason: Some(pa_types::ai::StopReason::Length),
                ..Default::default()
            },
        )
    };
    registration.set_responses(vec![
        budget_recording_step(&budgets, truncated()),
        budget_recording_step(&budgets, truncated()),
    ]);
    let model = registration.get_model();
    let global = tmp.path().join("harness");
    let error = session
        .refine(
            &refine::RefineOptions::default(),
            refine::RefinementSource::User,
            &model,
            None,
            global,
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        crate::refinement::planner::TRUNCATED_JSON_ERROR
    );
    assert_eq!(budgets.lock().unwrap().len(), 2, "the retry fired once");
}

/// An explicitly configured `maxTokens` is honored, not raised to the
/// refinement floor.
#[tokio::test]
async fn an_explicitly_configured_budget_is_honored_not_raised() {
    let (session, tmp) = refine_test_session().await;
    let registration = register_refine_model(8_192, 128_000);
    let budgets: Arc<std::sync::Mutex<Vec<Option<u64>>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    registration.set_responses(vec![budget_recording_step(
        &budgets,
        pa_ai::faux::faux_assistant_text_message(
            r#"{"summary":"one edit","edits":[{"action":"create","kind":"memory","id":"m9","title":"t","content":"c"}]}"#,
            pa_ai::faux::FauxAssistantMessageOptions::default(),
        ),
    )]);
    let mut model = registration.get_model();
    model.max_tokens_explicit = true;
    let global = tmp.path().join("harness");
    session
        .refine(
            &refine::RefineOptions::default(),
            refine::RefinementSource::User,
            &model,
            None,
            global,
        )
        .await
        .unwrap();
    assert_eq!(
        *budgets.lock().unwrap(),
        vec![Some(8_192)],
        "the explicit budget stays as configured"
    );
}

/// A roomy catalog model still gets the 32k refinement ceiling.
#[tokio::test]
async fn a_roomy_catalog_model_keeps_the_ceiling_capped_budget() {
    let (session, tmp) = refine_test_session().await;
    let registration = register_refine_model(131_072, 400_000);
    let budgets: Arc<std::sync::Mutex<Vec<Option<u64>>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    registration.set_responses(vec![budget_recording_step(
        &budgets,
        pa_ai::faux::faux_assistant_text_message(
            r#"{"summary":"one edit","edits":[{"action":"create","kind":"memory","id":"m9","title":"t","content":"c"}]}"#,
            pa_ai::faux::FauxAssistantMessageOptions::default(),
        ),
    )]);
    let model = registration.get_model();
    let global = tmp.path().join("harness");
    session
        .refine(
            &refine::RefineOptions::default(),
            refine::RefinementSource::User,
            &model,
            None,
            global,
        )
        .await
        .unwrap();
    assert_eq!(
        *budgets.lock().unwrap(),
        vec![Some(
            crate::refinement::planner::REFINEMENT_MAX_OUTPUT_TOKENS
        )],
        "the refinement ceiling still caps the budget"
    );
}

/// The retry doubles the planning window: a kernel harness write landing
/// between the two attempts must still be rejected by the baseline recheck,
/// not clobbered by the retry's plan.
#[tokio::test]
async fn a_retry_rejects_an_entry_changed_during_planning() {
    let (session, tmp) = refine_test_session().await;
    let global = tmp.path().join("harness");
    let local_dir = {
        let persistence = session.shared_persistence();
        let manager = persistence.lock().await;
        refine::local_harness_state_dir(&manager)
    };
    let seed_entry = |content: &str| {
        crate::refinement::try_update_harness_state(
            &local_dir,
            crate::refinement::HarnessScope::Local,
            |state| {
                state
                    .entries
                    .get_mut(&crate::refinement::RefinementKind::Memory)
                    .unwrap()
                    .insert(
                        "m1".to_string(),
                        crate::refinement::HarnessEntry {
                            id: "m1".to_string(),
                            kind: crate::refinement::RefinementKind::Memory,
                            title: "Concurrent".to_string(),
                            content: content.to_string(),
                            path: "/memories/m1".to_string(),
                            scope: Some(crate::refinement::HarnessScope::Local),
                            reference: serde_json::Map::default(),
                            arguments: serde_json::Map::default(),
                            metadata: serde_json::Map::default(),
                            source: "test".to_string(),
                            created_at: String::new(),
                            updated_at: String::new(),
                            version: 1,
                        },
                    );
                anyhow::Ok(())
            },
        )
        .unwrap();
    };
    seed_entry("original content");
    let calls = Arc::new(std::sync::Mutex::new(0usize));
    let call_counter = Arc::clone(&calls);
    let mid_plan_write = local_dir.clone();
    let seam: crate::refinement::executor::RefinerFn = Box::new(move |_model, _system, _prompt| {
        let first = {
            let mut calls = call_counter.lock().unwrap();
            *calls += 1;
            *calls == 1
        };
        let mid_plan_write = mid_plan_write.clone();
        Box::pin(async move {
            if first {
                // A kernel write racing the planning window the retry widened.
                crate::refinement::try_update_harness_state(
                    &mid_plan_write,
                    crate::refinement::HarnessScope::Local,
                    |state| {
                        state
                            .entries
                            .get_mut(&crate::refinement::RefinementKind::Memory)
                            .unwrap()
                            .get_mut("m1")
                            .unwrap()
                            .content = "changed by the kernel mid-planning".to_string();
                        anyhow::Ok(())
                    },
                )
                .unwrap();
                Ok(refiner_reply(
                    r#"{"summary":"cut","edits":[{"action":"cre"#,
                    pa_types::ai::StopReason::Length,
                ))
            } else {
                Ok(refiner_reply(
                    r#"{"summary":"update","edits":[{"action":"update","kind":"memory","id":"m1","title":"Concurrent","content":"the plan's content"}]}"#,
                    pa_types::ai::StopReason::Stop,
                ))
            }
        })
    });
    let result = session
        .refine_with_refiner(
            &refine::RefineOptions::default(),
            refine::RefinementSource::User,
            &session_ai_model(),
            seam,
            global,
        )
        .await
        .unwrap();
    assert_eq!(*calls.lock().unwrap(), 2, "the truncated reply retried");
    assert!(
        !result.applied_edits[0].applied,
        "the stale-baseline edit is rejected"
    );
    assert_eq!(
        result.applied_edits[0].error.as_deref(),
        Some("entry changed during refinement planning")
    );
    let after = {
        let persistence = session.shared_persistence();
        let manager = persistence.lock().await;
        crate::refinement::load_harness_state(
            &refine::local_harness_state_dir(&manager),
            crate::refinement::HarnessScope::Local,
        )
    };
    assert_eq!(
        after.entries[&crate::refinement::RefinementKind::Memory]["m1"].content,
        "changed by the kernel mid-planning",
        "the concurrent kernel write survives"
    );
}
