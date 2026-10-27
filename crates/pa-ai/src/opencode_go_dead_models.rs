//! Learned-dead `opencode-go` models: entries the gateway settled as retired.
//!
//! The live catalog drops a deprecated entry once the host publishes the
//! change, but a settled inference failure is the earliest signal. The ids are
//! persisted under the agent directory (best-effort) and filtered out of the
//! roster and the catalog cache until an operator removes them.

use std::collections::BTreeSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::opencode_go_spec_generated as spec;

/// The learned-dead model-id list under the agent directory.
pub const DEAD_MODELS_FILE: &str = "opencode-go-dead-models.json";

/// The dead-model list path under `agent_dir`.
#[must_use]
pub fn dead_models_path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(DEAD_MODELS_FILE)
}

/// The learned-dead ids. A missing or unreadable file is an empty set.
#[must_use]
pub fn load(agent_dir: &Path) -> BTreeSet<String> {
    std::fs::read_to_string(dead_models_path(agent_dir))
        .ok()
        .and_then(|content| serde_json::from_str::<BTreeSet<String>>(&content).ok())
        .unwrap_or_default()
}

/// Persist `id` as dead (atomic temp-file rename, mode 0600). Idempotent, and
/// a failed write never fails the caller: the next failure retries it.
pub fn record(agent_dir: &Path, id: &str) {
    if id.is_empty() {
        return;
    }
    let mut dead = load(agent_dir);
    if !dead.insert(id.to_string()) {
        return;
    }
    save(agent_dir, &dead);
}

/// The classification of a settled inference failure: a 410 or a message that
/// names the retirement. A plain 404 is a routing blip, not a retirement.
#[must_use]
pub fn is_dead_failure(status: Option<u16>, message: &str) -> bool {
    if status.is_some_and(|status| spec::DEAD_MODEL_STATUSES.contains(&status)) {
        return true;
    }
    let message = message.to_ascii_lowercase();
    spec::DEAD_MODEL_MARKERS
        .iter()
        .any(|marker| message.contains(&marker.to_ascii_lowercase()))
}

fn save(agent_dir: &Path, dead: &BTreeSet<String>) {
    let Ok(content) = serde_json::to_string_pretty(dead) else {
        return;
    };
    let path = dead_models_path(agent_dir);
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let temporary = path.with_extension("json.tmp");
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let write = options
        .open(&temporary)
        .and_then(|mut file| file.write_all(format!("{content}\n").as_bytes()));
    if write.is_ok() {
        let _ = std::fs::rename(&temporary, &path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_410_or_a_retirement_message_marks_the_model_dead() {
        assert!(is_dead_failure(Some(410), "gone"));
        assert!(is_dead_failure(None, "ModelDeprecated: deepseek-v4"));
        assert!(is_dead_failure(None, "this model has been deprecated"));
        assert!(is_dead_failure(Some(410), ""));
        assert!(
            !is_dead_failure(Some(404), "not found"),
            "a plain 404 is a routing blip, not a retirement"
        );
        assert!(!is_dead_failure(None, "insufficient funds"));
        assert!(!is_dead_failure(Some(429), "rate limited"));
    }

    #[test]
    fn recorded_ids_persist_across_reads_with_0600() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load(dir.path()).is_empty());
        record(dir.path(), "space-bunny");
        record(dir.path(), "space-bunny");
        record(dir.path(), "minimax-m2.7");
        assert_eq!(
            load(dir.path()),
            BTreeSet::from(["space-bunny".to_string(), "minimax-m2.7".to_string()])
        );
        record(dir.path(), "");
        assert_eq!(load(dir.path()).len(), 2, "an empty id is ignored");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(dead_models_path(dir.path()))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn a_filtered_roster_drops_the_learned_dead_models() {
        let dir = tempfile::tempdir().unwrap();
        record(dir.path(), "space-bunny");
        let dead = load(dir.path());
        let roster: Vec<String> = crate::opencode_go_models::models()
            .into_iter()
            .map(|model| model.id)
            .filter(|id| !dead.contains(id))
            .collect();
        assert!(!roster.contains(&"space-bunny".to_string()));
        assert_eq!(roster.len(), 30);
    }
}
