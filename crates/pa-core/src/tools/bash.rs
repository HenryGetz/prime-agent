//! The `bash` tool: shell command execution with output truncation and a destructive-git dirty-tree
//! guard (execution, guard, truncation, and formatting match the TS tool).

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::json;

use crate::tools::bash_guard::{
    find_destructive_git_discard_commands, format_dirty_tree_refusal, is_truthy_env_value,
    resolve_discard_probe_target, DiscardProbeResolution, BASH_DESTRUCTIVE_GIT_BYPASS_ENV,
};
use crate::tools::output_accumulator::{OutputAccumulator, OutputAccumulatorOptions};
use crate::tools::shell_utils::get_shell_env;
use crate::tools::tool_definition::{
    AbortSignal, OnUpdate, ToolContentBlock, ToolDefinition, ToolExecutionResult, ToolUpdate,
};
use crate::tools::truncate::{
    format_size, TruncatedBy, TruncationResult, DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES,
};

/// The default porcelain status command the guard probes with.
const GIT_STATUS_PORCELAIN_COMMAND: &str = "git status --porcelain --untracked-files=all";

/// Throttle for streamed output updates (TS: `BASH_UPDATE_THROTTLE_MS`).
const BASH_UPDATE_THROTTLE_MS: Duration = Duration::from_millis(100);

pub use crate::tools::bash_local::LocalBashOperations;

pub(crate) type ExecFuture<'a> =
    Pin<Box<dyn Future<Output = anyhow::Result<Option<i32>>> + Send + 'a>>;

/// Execution options for [`BashOperations::exec`].
pub struct ExecOptions<'a> {
    /// Receives streamed stdout/stderr chunks.
    pub on_data: &'a (dyn Fn(&[u8]) + Send + Sync),
    /// Reports the spawned child's pid right after spawn (the
    /// auto-background job id and notice identity). Optional: backends
    /// without process ids leave it unset.
    pub on_spawn: Option<&'a (dyn Fn(u32) + Send + Sync)>,
    /// Cancellation handle; kills the process tree when fired.
    pub signal: Option<AbortSignal>,
    /// Timeout in seconds.
    pub timeout: Option<f64>,
    /// Environment for the child; defaults to the agent shell env.
    pub env: Option<HashMap<String, String>>,
}

/// Pluggable operations for the bash tool. Implementations stream merged stdout/stderr through
/// `on_data`, kill the process tree on cancellation, resolve with the child exit code (`None` when
/// killed by a signal), and reject with `"aborted"` / `"timeout:<seconds>"`. Object-safe on purpose
/// (`&dyn` injection).
pub trait BashOperations: Send + Sync {
    /// Execute a command and stream output through `on_data`; resolves with the exit code (`None`
    /// when killed by a signal), or an error: `"aborted"` or `"timeout:<seconds>"`.
    fn exec<'a>(
        &'a self,
        command: &'a str,
        cwd: &'a str,
        options: ExecOptions<'a>,
    ) -> ExecFuture<'a>;
}

/// Context for the spawn hook: command, cwd, env.
pub struct BashSpawnContext {
    pub command: String,
    pub cwd: String,
    pub env: HashMap<String, String>,
}

pub type BashSpawnHook = Arc<dyn Fn(BashSpawnContext) -> BashSpawnContext + Send + Sync>;

fn resolve_spawn_context(
    command: &str,
    cwd: &str,
    spawn_hook: Option<&BashSpawnHook>,
) -> BashSpawnContext {
    let base = BashSpawnContext {
        command: command.to_string(),
        cwd: cwd.to_string(),
        env: get_shell_env(),
    };
    match spawn_hook {
        Some(hook) => hook(base),
        None => base,
    }
}

/// Options for the bash tool.
#[derive(Default)]
pub struct BashToolOptions {
    /// Custom operations for command execution. Default: local shell.
    pub operations: Option<Arc<dyn BashOperations>>,
    /// Command prefix prepended to every command.
    pub command_prefix: Option<String>,
    /// Optional explicit shell path from settings.
    pub shell_path: Option<String>,
    /// Hook to adjust command, cwd, or env before execution.
    pub spawn_hook: Option<BashSpawnHook>,
    /// The inline promotion window: a call still running this long is
    /// promoted to a background job. `None` uses the default
    /// ([`pa_agent::auto_background::DEFAULT_AUTO_BACKGROUND_AFTER_MS`]);
    /// `Some` of a zero duration disables promotion. The agent loop's
    /// per-call arm overrides this.
    pub auto_background_after: Option<Duration>,
    /// The completion notice for promoted jobs (direct calls; the agent
    /// loop's arm carries the loop's notice lane instead).
    pub on_background_completion: Option<pa_agent::auto_background::AutoBackgroundNoticeFn>,
}

/// Probe for at-risk files via `git status --porcelain`. Returns `Ok(None)`
/// when dirtiness cannot be determined, so the guard fails open.
async fn probe_uncommitted_changes(
    ops: &dyn BashOperations,
    probe_command: &str,
    cwd: &str,
    env: &HashMap<String, String>,
    signal: Option<AbortSignal>,
    timeout: Option<f64>,
) -> anyhow::Result<Option<Vec<String>>> {
    let output = std::sync::Mutex::new(String::new());
    let result = ops
        .exec(
            probe_command,
            cwd,
            ExecOptions {
                on_data: &|data: &[u8]| {
                    output
                        .lock()
                        .unwrap()
                        .push_str(&String::from_utf8_lossy(data));
                },
                on_spawn: None,
                signal,
                timeout,
                env: Some(env.clone()),
            },
        )
        .await;
    match result {
        Ok(Some(0)) => {}
        Err(err) if err.to_string() == "aborted" => return Err(err),
        Ok(_) | Err(_) => return Ok(None),
    }
    Ok(Some(
        output
            .into_inner()
            .expect("probe output unlocked")
            .split('\n')
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.trim_end_matches('\r').to_string())
            .collect(),
    ))
}

/// Serialize a truncation result with the TS wire shape.
pub fn truncation_to_json(truncation: &TruncationResult) -> serde_json::Value {
    json!({
        "content": truncation.content,
        "truncated": truncation.truncated,
        "truncatedBy": match truncation.truncated_by {
            Some(TruncatedBy::Lines) => json!("lines"),
            Some(TruncatedBy::Bytes) => json!("bytes"),
            None => serde_json::Value::Null,
        },
        "totalLines": truncation.total_lines,
        "totalBytes": truncation.total_bytes,
        "outputLines": truncation.output_lines,
        "outputBytes": truncation.output_bytes,
        "lastLinePartial": truncation.last_line_partial,
        "firstLineExceedsLimit": truncation.first_line_exceeds_limit,
        "maxLines": truncation.max_lines,
        "maxBytes": truncation.max_bytes,
    })
}

struct FormattedOutput {
    text: String,
    details: Option<serde_json::Value>,
}

fn format_output(
    snapshot: &crate::tools::output_accumulator::OutputSnapshot,
    last_line_bytes: usize,
    empty_text: &str,
) -> FormattedOutput {
    let truncation = &snapshot.truncation;
    let mut text = if snapshot.content.is_empty() {
        empty_text.to_string()
    } else {
        snapshot.content.clone()
    };
    let mut details = None;
    if truncation.truncated {
        let mut d = json!({ "truncation": truncation_to_json(truncation) });
        if let Some(path) = &snapshot.full_output_path {
            d["fullOutputPath"] = json!(path);
        }
        details = Some(d);
        let start_line = truncation.total_lines - truncation.output_lines + 1;
        let end_line = truncation.total_lines;
        // A degraded spill has no path; never advertise a missing file.
        let location = snapshot
            .full_output_path
            .as_deref()
            .map(|p| format!(". Full output: {p}"))
            .unwrap_or_default();
        if truncation.last_line_partial {
            let line_size = if last_line_bytes > 0 {
                format!(" (line is {})", format_size(last_line_bytes))
            } else {
                String::new()
            };
            let _ = write!(
                text,
                "\n\n[Showing last {} of line {start_line}{line_size}{location}]",
                format_size(truncation.output_bytes)
            );
        } else if truncation.truncated_by == Some(TruncatedBy::Lines) {
            let _ = write!(
                text,
                "\n\n[Showing lines {start_line}-{end_line} of {}{location}]",
                truncation.total_lines
            );
        } else {
            let _ = write!(
                text,
                "\n\n[Showing lines {start_line}-{end_line} of {} ({} limit){location}]",
                truncation.total_lines,
                format_size(DEFAULT_MAX_BYTES)
            );
        }
    }
    FormattedOutput { text, details }
}

fn append_status(text: &str, status: &str) -> String {
    if text.is_empty() {
        status.to_string()
    } else {
        format!("{text}\n\n{status}")
    }
}

/// Execute one bash tool call. Errors carry the model-facing text.
// The TS-parity parameter list is the tool's contract.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(
    level = "debug",
    name = "tool_bash_execute",
    skip(cwd, options, on_update)
    fields(command),
)]
pub async fn execute_bash(
    tool_call_id: &str,
    cwd: &str,
    options: &BashToolOptions,
    command: &str,
    timeout: Option<f64>,
    allow_destructive_git: Option<bool>,
    signal: Option<AbortSignal>,
    on_update: Option<OnUpdate>,
) -> anyhow::Result<ToolExecutionResult> {
    let ops: Arc<dyn BashOperations> = match &options.operations {
        Some(ops) => Arc::clone(ops),
        None => Arc::new(LocalBashOperations {
            shell_path: options.shell_path.clone(),
        }),
    };
    let command_prefix = options.command_prefix.as_deref();
    let spawn_hook = options.spawn_hook.as_ref();

    let resolved_command = match command_prefix {
        Some(prefix) => format!("{prefix}\n{command}"),
        None => command.to_string(),
    };
    let spawn_context = resolve_spawn_context(&resolved_command, cwd, spawn_hook);

    // Refuse destructive git discard commands while the tree is dirty.
    let discard_indices = find_destructive_git_discard_commands(&resolved_command);
    if !discard_indices.is_empty()
        && allow_destructive_git != Some(true)
        && !is_truthy_env_value(spawn_context.env.get(BASH_DESTRUCTIVE_GIT_BYPASS_ENV))
    {
        let mut probes: Vec<(BashSpawnContext, bool)> = Vec::new();
        let mut seen_probes: HashSet<String> = HashSet::new();
        let user_command_start = command_prefix.map_or(0, |p| p.len() + 1);
        for index in discard_indices {
            let target = resolve_discard_probe_target(&resolved_command, index, user_command_start);
            match target {
                DiscardProbeResolution::Unresolvable => {
                    anyhow::bail!("{RELOCATION_REFUSAL}");
                }
                DiscardProbeResolution::NotRelocated => {
                    let raw_probe = match command_prefix {
                        Some(prefix) => format!("{prefix}\n{GIT_STATUS_PORCELAIN_COMMAND}"),
                        None => GIT_STATUS_PORCELAIN_COMMAND.to_string(),
                    };
                    let context = resolve_spawn_context(&raw_probe, cwd, spawn_hook);
                    let key = format!("{}\u{0}{}", context.command, context.cwd);
                    if seen_probes.insert(key) {
                        probes.push((context, false));
                    }
                }
                DiscardProbeResolution::Target(target) => {
                    let relocation_prefix = target.relocation_prefix.as_deref().unwrap_or("");
                    let raw_probe = match command_prefix {
                        Some(prefix) => {
                            format!("{prefix}\n{relocation_prefix}{}", target.git_status_command)
                        }
                        None => format!("{relocation_prefix}{}", target.git_status_command),
                    };
                    let context = resolve_spawn_context(&raw_probe, cwd, spawn_hook);
                    let key = format!("{}\u{0}{}", context.command, context.cwd);
                    if seen_probes.insert(key) {
                        probes.push((
                            context,
                            target.git_status_command.contains("--ignored=matching"),
                        ));
                    }
                }
            }
        }
        for (context, includes_ignored) in probes {
            let dirty_paths = probe_uncommitted_changes(
                ops.as_ref(),
                &context.command,
                &context.cwd,
                &context.env,
                signal.clone(),
                timeout,
            )
            .await?;
            if let Some(paths) = dirty_paths.filter(|paths| !paths.is_empty()) {
                anyhow::bail!("{}", format_dirty_tree_refusal(&paths, includes_ignored));
            }
        }
    }

    let acc = Arc::new(Mutex::new(OutputAccumulator::new(
        &OutputAccumulatorOptions {
            temp_file_prefix: "pi-bash".to_string(),
            ..OutputAccumulatorOptions::default()
        },
    )));
    let notify = Arc::new(tokio::sync::Notify::new());
    let dirty = Arc::new(AtomicBool::new(false));

    if let Some(on_update) = &on_update {
        on_update(ToolUpdate {
            content: Vec::new(),
            details: None,
        });
    }

    // The promotion window: the agent loop's per-call arm (AgentOptions
    // derived) wins; a direct call falls back to the tool options. A zero
    // window disables promotion.
    let (window, notice) = match pa_agent::auto_background::take(tool_call_id) {
        Some(arm) => (arm.window, arm.notice),
        None => (
            options
                .auto_background_after
                .unwrap_or(Duration::from_millis(
                    pa_agent::auto_background::DEFAULT_AUTO_BACKGROUND_AFTER_MS,
                )),
            options.on_background_completion.clone(),
        ),
    };

    // The command runs in its own task so a promotion can hand it off: on
    // promotion this call returns while the task keeps driving the process,
    // streaming output into the accumulator's disk spill, and the
    // completion notice goes out through the same lane a backgrounded
    // `bash()` job uses. The command KEEPS RUNNING; nothing is killed.
    let on_data = {
        let notify = notify.clone();
        let dirty = dirty.clone();
        let acc = acc.clone();
        Arc::new(move |data: &[u8]| {
            acc.lock().unwrap().append(data);
            dirty.store(true, Ordering::SeqCst);
            notify.notify_one();
        }) as Arc<dyn Fn(&[u8]) + Send + Sync>
    };
    let pid_slot = Arc::new(Mutex::new(None::<u32>));
    let promoted = Arc::new(AtomicBool::new(false));
    let (done_tx, mut done_rx) = tokio::sync::oneshot::channel::<ExecDone>();

    {
        let ops = Arc::clone(&ops);
        let task_command = spawn_context.command.clone();
        // The notice carries the raw command (kernel parity: the prefix is
        // machinery, not the user's command).
        let notice_command = command.to_string();
        let task_cwd = spawn_context.cwd.clone();
        let task_env = spawn_context.env.clone();
        let task_signal = signal.clone();
        let task_on_update = on_update.clone();
        let on_data = Arc::clone(&on_data);
        let pid_slot = Arc::clone(&pid_slot);
        let promoted = Arc::clone(&promoted);
        let acc = Arc::clone(&acc);
        tokio::spawn(async move {
            let on_spawn = {
                let pid_slot = Arc::clone(&pid_slot);
                move |pid: u32| {
                    *pid_slot.lock().unwrap() = Some(pid);
                }
            };
            let exec_fut = ops.exec(
                &task_command,
                &task_cwd,
                ExecOptions {
                    on_data: on_data.as_ref(),
                    on_spawn: Some(&on_spawn),
                    signal: task_signal,
                    timeout,
                    env: Some(task_env),
                },
            );
            tokio::pin!(exec_fut);

            let mut last_update_at: Option<std::time::Instant> = None;
            let exec_result: anyhow::Result<Option<i32>> = loop {
                let deadline = if dirty.load(Ordering::SeqCst) {
                    last_update_at.map(|last| {
                        let elapsed = last.elapsed();
                        if elapsed >= BASH_UPDATE_THROTTLE_MS {
                            std::time::Instant::now()
                        } else {
                            last + BASH_UPDATE_THROTTLE_MS
                        }
                    })
                } else {
                    None
                };
                tokio::select! {
                    () = notify.notified() => {}
                    () = async {
                        match deadline {
                            Some(deadline) => tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await,
                            None => std::future::pending::<()>().await,
                        }
                    }, if deadline.is_some() => {}
                    result = &mut exec_fut => {
                        break result;
                    }
                }
                if dirty.swap(false, Ordering::SeqCst) {
                    last_update_at = Some(std::time::Instant::now());
                    if let Some(on_update) = &task_on_update {
                        let snapshot = acc.lock().unwrap().snapshot();
                        on_update(ToolUpdate {
                            content: vec![ToolContentBlock::text(snapshot.content.clone())],
                            details: Some(json!({
                                "truncation": if snapshot.truncation.truncated {
                                    truncation_to_json(&snapshot.truncation)
                                } else {
                                    serde_json::Value::Null
                                },
                                "fullOutputPath": snapshot.full_output_path,
                            })),
                        });
                    }
                }
            };

            {
                let mut acc = acc.lock().unwrap();
                acc.finish();
            }
            let (snapshot, last_line_bytes) = {
                let mut acc = acc.lock().unwrap();
                acc.close_temp_file_sync();
                let snapshot = acc.snapshot();
                (snapshot, acc.get_last_line_bytes())
            };
            let pid = *pid_slot.lock().unwrap();
            if promoted.load(Ordering::SeqCst) {
                if let Some(notice) = notice {
                    notice(pa_agent::auto_background::AutoBackgroundCompletion {
                        pid,
                        exit_code: exec_result.as_ref().ok().copied().flatten(),
                        command: notice_command,
                        output: snapshot.content.clone(),
                        full_output_path: snapshot.full_output_path,
                    });
                }
            } else {
                let _ = done_tx.send(ExecDone {
                    result: exec_result,
                    snapshot,
                    last_line_bytes,
                });
            }
        });
    }

    enum ExecEvent {
        Done(ExecDone),
        Promoted,
    }
    let exec_event = if window.is_zero() {
        match done_rx.await {
            Ok(done) => ExecEvent::Done(done),
            Err(_) => anyhow::bail!("bash execution task failed"),
        }
    } else {
        tokio::select! {
            done = &mut done_rx => match done {
                Ok(done) => ExecEvent::Done(done),
                Err(_) => anyhow::bail!("bash execution task failed"),
            },
            () = tokio::time::sleep(window) => ExecEvent::Promoted,
        }
    };

    match exec_event {
        ExecEvent::Promoted => {
            promoted.store(true, Ordering::SeqCst);
            let pid = *pid_slot.lock().unwrap();
            let snapshot = {
                let acc = acc.lock().unwrap();
                acc.snapshot()
            };
            let job = match pid {
                Some(pid) => pid.to_string(),
                None => format!("bash-{}", JOB_SEQUENCE.fetch_add(1, Ordering::SeqCst)),
            };
            let window_text = if window.as_millis() % 1000 == 0 {
                format!("{}s", window.as_secs())
            } else {
                format!("{}ms", window.as_millis())
            };
            let mut details = json!({
                "autoBackgrounded": true,
                "pid": pid,
                "command": command,
            });
            if let Some(path) = &snapshot.full_output_path {
                details["fullOutputPath"] = json!(path);
            }
            Ok(ToolExecutionResult {
                content: vec![ToolContentBlock::text(format!(
                    "Command exceeded {window_text}. Auto-backgrounded as job {job}. Output \
streams to disk. The completion notification wakes the turn. Continue other work or end the turn."
                ))],
                details: Some(details),
                is_error: false,
            })
        }
        ExecEvent::Done(ExecDone {
            result: exec_result,
            snapshot,
            last_line_bytes,
        }) => match exec_result {
            Err(err) => {
                // TS catch path: timeouts and aborts keep the partial output and
                // append a status line; other errors surface raw.
                let message = err.to_string();
                if message == "aborted" {
                    let formatted = format_output(&snapshot, last_line_bytes, "");
                    anyhow::bail!(append_status(&formatted.text, "Command aborted"));
                }
                if let Some(secs) = message.strip_prefix("timeout:") {
                    let formatted = format_output(&snapshot, last_line_bytes, "");
                    anyhow::bail!(append_status(
                        &formatted.text,
                        &format!("Command timed out after {secs} seconds")
                    ));
                }
                Err(err)
            }
            Ok(exit_code) => {
                let formatted = format_output(&snapshot, last_line_bytes, "(no output)");
                let text = formatted.text;
                if exit_code.is_some_and(|code| code != 0) {
                    anyhow::bail!(append_status(
                        &text,
                        &format!("Command exited with code {}", exit_code.unwrap())
                    ));
                }
                Ok(ToolExecutionResult {
                    content: vec![ToolContentBlock::text(text)],
                    details: formatted.details,
                    is_error: false,
                })
            }
        },
    }
}

/// The detached exec task's handoff: the process result plus the finished
/// accumulator snapshot.
struct ExecDone {
    result: anyhow::Result<Option<i32>>,
    snapshot: crate::tools::output_accumulator::OutputSnapshot,
    last_line_bytes: usize,
}

/// Job-number fallback for promotion when the operations backend reports no
/// pid.
static JOB_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub fn bash_tool_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "required": ["command"],
        "properties": {
            "command": {
                "type": "string",
                "description": "Bash command to execute"
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (optional, no default timeout)"
            },
            "allowDestructiveGit": {
                "type": "boolean",
                "description": "Skip the dirty-tree guard for destructive git discard commands. Only set when discarding uncommitted work is intentional."
            }
        }
    })
}

pub fn bash_tool_description() -> String {
    format!(
        "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds. Destructive git discard commands (git checkout -- ., git checkout ., git clean -f..., git reset --hard, git restore .) are refused while uncommitted changes exist; retry with allowDestructiveGit: true only when the discard is intentional.",
        DEFAULT_MAX_BYTES / 1024
    )
}

/// The `bash` tool definition: exact name, schema, and description.
#[must_use]
pub fn create_bash_tool_definition(cwd: &str) -> ToolDefinition {
    create_bash_tool_definition_with_options(cwd, &BashToolOptions::default())
}

#[must_use]
pub fn create_bash_tool_definition_with_options(
    cwd: &str,
    options: &BashToolOptions,
) -> ToolDefinition {
    let cwd = cwd.to_string();
    let operations = options.operations.clone();
    let command_prefix = options.command_prefix.clone();
    let shell_path = options.shell_path.clone();
    let spawn_hook = options.spawn_hook.clone();
    let auto_background_after = options.auto_background_after;
    let on_background_completion = options.on_background_completion.clone();
    let execute: crate::tools::tool_definition::ExecuteFn = {
        Arc::new(move |tool_call_id, params, signal, on_update| {
            let cwd = cwd.clone();
            let tool_call_id = tool_call_id.to_string();
            let options = BashToolOptions {
                operations: operations.clone(),
                command_prefix: command_prefix.clone(),
                shell_path: shell_path.clone(),
                spawn_hook: spawn_hook.clone(),
                auto_background_after,
                on_background_completion: on_background_completion.clone(),
            };
            Box::pin(async move {
                let command = params
                    .get("command")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| anyhow::anyhow!("bash tool requires a command string"))?
                    .to_string();
                let timeout = params.get("timeout").and_then(serde_json::Value::as_f64);
                let allow_destructive_git = params
                    .get("allowDestructiveGit")
                    .and_then(serde_json::Value::as_bool);
                execute_bash(
                    &tool_call_id,
                    &cwd,
                    &options,
                    &command,
                    timeout,
                    allow_destructive_git,
                    signal,
                    on_update,
                )
                .await
            })
        })
    };
    ToolDefinition {
        name: "bash".to_string(),
        label: "bash".to_string(),
        description: bash_tool_description(),
        prompt_snippet: "Execute bash commands (ls, grep, find, etc.)".to_string(),
        parameters: bash_tool_schema(),
        execution_mode: None,
        prepare_arguments: None,
        execute,
    }
}

/// Refusal text for discard commands whose target repository cannot be
/// resolved safely (TS: `formatRelocationRefusal`).
const RELOCATION_REFUSAL: &str = "Refusing to run this destructive git command: it changes directory (or repository) first, and the uncommitted changes of the repository it targets cannot be checked safely.\n\nRun the discard as its own command from the target directory, or retry with allowDestructiveGit: true, or set PI_BASH_ALLOW_DESTRUCTIVE_GIT=1.";

#[cfg(test)]
mod auto_background_tests {
    //! The promotion contract: an inline bash call still running past the
    //! window is handed off to the background machinery (the command keeps
    //! running, output keeps streaming, the completion notice fires), the
    //! tool result returns immediately with the synthetic text, and a call
    //! under the window (or with promotion disabled) runs inline unchanged.

    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use super::{execute_bash, BashToolOptions};
    use pa_agent::auto_background::AutoBackgroundCompletion;

    #[tokio::test]
    async fn a_call_past_the_window_promotes_and_notifies() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AutoBackgroundCompletion>();
        let options = BashToolOptions {
            on_background_completion: Some(Arc::new(move |completion| {
                tx.send(completion).ok();
            })),
            auto_background_after: Some(Duration::from_secs(2)),
            ..BashToolOptions::default()
        };

        let started = Instant::now();
        let result = execute_bash(
            "test-promote",
            ".",
            &options,
            "sleep 5; echo promoted-lander",
            None,
            None,
            None,
            None,
        )
        .await
        .expect("promotion returns Ok");
        // (a) The tool result returns by the window, not the command.
        assert!(
            started.elapsed() < Duration::from_secs(4),
            "promotion must return at the window, took {:?}",
            started.elapsed()
        );
        let text = result.content[0].as_text().expect("text block");
        assert!(
            text.starts_with("Command exceeded 2s. Auto-backgrounded as job "),
            "unexpected synthetic text: {text}"
        );
        assert!(text.contains(
            "Output streams to disk. The completion notification wakes the turn. \
Continue other work or end the turn."
        ));
        assert!(!result.is_error);
        let details = result.details.expect("promotion details");
        assert_eq!(details["autoBackgrounded"], serde_json::json!(true));
        let pid = details["pid"].as_u64().expect("pid reported");

        // (a) The process outlives the call; its output lands via the
        // notice and the byte-cursor reads it back.
        let completion = tokio::time::timeout(Duration::from_secs(8), rx.recv())
            .await
            .expect("completion notice wakes within the bound")
            .expect("notice channel alive");
        assert_eq!(completion.pid, Some(u32::try_from(pid).expect("pid fits")));
        assert_eq!(completion.exit_code, Some(0));
        assert_eq!(completion.command, "sleep 5; echo promoted-lander");
        assert!(
            completion.output.contains("promoted-lander"),
            "byte-cursor read: {}",
            completion.output
        );
    }

    #[tokio::test]
    async fn a_call_under_the_window_runs_inline() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AutoBackgroundCompletion>();
        let options = BashToolOptions {
            on_background_completion: Some(Arc::new(move |completion| {
                tx.send(completion).ok();
            })),
            auto_background_after: Some(Duration::from_secs(10)),
            ..BashToolOptions::default()
        };

        let result = execute_bash(
            "test-inline",
            ".",
            &options,
            "echo quick-lander",
            None,
            None,
            None,
            None,
        )
        .await
        .expect("inline call succeeds");
        // (b) No promotion: the normal output returns, no notice fires.
        let text = result.content[0].as_text().expect("text block");
        assert_eq!(text, "quick-lander\n");
        assert!(result.details.is_none());
        assert!(rx.try_recv().is_err(), "no notice for an inline call");
    }

    #[tokio::test]
    async fn a_zero_window_disables_promotion() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AutoBackgroundCompletion>();
        let options = BashToolOptions {
            on_background_completion: Some(Arc::new(move |completion| {
                tx.send(completion).ok();
            })),
            auto_background_after: Some(Duration::ZERO),
            ..BashToolOptions::default()
        };

        let started = Instant::now();
        let result = execute_bash(
            "test-disabled",
            ".",
            &options,
            "sleep 2; echo slow-lander",
            None,
            None,
            None,
            None,
        )
        .await
        .expect("disabled promotion runs inline");
        // (c) T=0 keeps the old contract: the call blocks to completion.
        assert!(
            started.elapsed() >= Duration::from_secs(2),
            "T=0 must not promote, returned after {:?}",
            started.elapsed()
        );
        assert_eq!(
            result.content[0].as_text().expect("text block"),
            "slow-lander\n"
        );
        assert!(rx.try_recv().is_err(), "no notice when promotion is off");
    }
}
