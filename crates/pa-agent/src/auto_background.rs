//! Inline-bash auto-backgrounding (the promotion seam): the agent loop
//! arms a per-call watchdog for `bash` tool calls, and the bash dispatch
//! (pa-core) consumes the arm to promote a long-running command to a
//! detached job instead of holding the turn. The registry here is the one
//! hand-off channel between the loop (which knows the configured window)
//! and the tool (which owns the running process).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

/// The default auto-background window: an inline bash call is promoted to
/// a background job after one minute.
pub const DEFAULT_AUTO_BACKGROUND_AFTER_MS: u64 = 60_000;

/// One promoted bash job's completion, handed to the notice lane.
#[derive(Debug, Clone)]
pub struct AutoBackgroundCompletion {
    /// The promoted command's process id (`None` when the operations
    /// backend does not report pids).
    pub pid: Option<u32>,
    /// The process exit code (`None` when killed by a signal).
    pub exit_code: Option<i32>,
    /// The executed command text.
    pub command: String,
    /// The job's final accumulated output.
    pub output: String,
    /// The spilled full-output file, when output was truncated.
    pub full_output_path: Option<String>,
}

/// The wake contract for a promoted job: the embedding wires this to the
/// same notice lane backgrounded bash jobs already use (the daemon feeds
/// its `[bash-done pid:N exit:M]` completion sink).
pub type AutoBackgroundNoticeFn = Arc<dyn Fn(AutoBackgroundCompletion) + Send + Sync>;

/// The per-call watchdog the loop arms for one inline bash call.
#[derive(Clone)]
pub struct AutoBackgroundArm {
    /// The promotion window; a zero duration disables promotion.
    pub window: Duration,
    /// The completion notice hook (`None` keeps the job's completion
    /// silent — no wake lane).
    pub notice: Option<AutoBackgroundNoticeFn>,
}

/// The loop-level directive derived from [`crate::agent::AgentOptions`]:
/// the resolved window plus the embedding's notice hook.
#[derive(Clone, Default)]
pub struct AutoBackgroundDirective {
    /// Zero disables promotion for every tool call.
    pub window: Duration,
    pub notice: Option<AutoBackgroundNoticeFn>,
}

impl AutoBackgroundDirective {
    /// Whether the directive arms a watchdog for this window.
    #[must_use]
    pub fn armed(&self) -> bool {
        !self.window.is_zero()
    }
}

fn arms() -> &'static Mutex<HashMap<String, AutoBackgroundArm>> {
    static ARMS: OnceLock<Mutex<HashMap<String, AutoBackgroundArm>>> = OnceLock::new();
    ARMS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Arm the watchdog for one tool call. The bash dispatch consumes the arm
/// (see [`take`]); the loop also disarms after the call settles, so an
/// arm for a call that never reached the bash dispatch cannot leak.
pub fn arm(tool_call_id: &str, value: AutoBackgroundArm) {
    arms()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(tool_call_id.to_string(), value);
}

/// Consume the arm for one tool call (the bash dispatch's promotion
/// window and notice lane for this call).
#[must_use]
pub fn take(tool_call_id: &str) -> Option<AutoBackgroundArm> {
    arms()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(tool_call_id)
}

/// Remove an unconsumed arm (the loop's post-call cleanup).
pub fn disarm(tool_call_id: &str) {
    arms()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(tool_call_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_take_and_disarm_round_trip() {
        arm(
            "t1",
            AutoBackgroundArm {
                window: Duration::from_secs(2),
                notice: None,
            },
        );
        let taken = take("t1");
        assert_eq!(taken.expect("arm consumed").window, Duration::from_secs(2));
        assert!(take("t1").is_none(), "arm consumed once");

        arm(
            "t2",
            AutoBackgroundArm {
                window: Duration::from_secs(1),
                notice: None,
            },
        );
        disarm("t2");
        assert!(take("t2").is_none(), "disarmed arm is gone");
    }
}
