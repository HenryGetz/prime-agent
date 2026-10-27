//! The runtime-ready probe concern: the two-layer memo (in-process map + the cross-process on-disk
//! verdict) that skips re-probing a venv nobody damaged.

use super::{collect_python_files, Digest, HashMap, Mutex, Path, PathBuf, Stdio};

/// What probing an interpreter found: whether the runtime-ready check passed, and which of the
/// requested modules failed to import.
pub(crate) struct PythonProbe {
    pub(crate) runtime_ready: bool,
    pub(crate) missing_imports: Vec<String>,
}

/// Read from stdin, so every argument stays one line (Windows batch-file interpreters reject
/// line breaks in arguments). Runs the runtime-ready check (`argv[1]`, skipped when empty), then
/// imports each module named in the remaining arguments, announcing each step on its own
/// flushed marker line before taking it.
const PROBE_SCRIPT: &str = r#"import importlib, os, sys
# Markers go to a private copy of stdout, so an import that redirects stdout can't hide them.
out = os.fdopen(os.dup(1), "w")
def say(*words):
    out.write("\n" + " ".join(("pa-probe",) + words) + "\n")
    out.flush()
if sys.argv[1]:
    try:
        exec(sys.argv[1], {})
        say("ready")
    except BaseException:
        say("not-ready")
for name in sys.argv[2:]:
    say("import", name)
    try:
        importlib.import_module(name)
    except BaseException:
        say("missing", name)
say("done")
"#;

/// Probe `python` with as few interpreter starts as possible: one, unless an import kills the
/// interpreter. That import counts as missing and the probe resumes after it. Blocking: callers
/// on an async runtime run it on the blocking pool.
pub(crate) fn probe_python(python: &str, check_runtime: bool, imports: &[&str]) -> PythonProbe {
    let mut probe = PythonProbe {
        runtime_ready: !check_runtime,
        missing_imports: Vec::new(),
    };
    let mut check = if check_runtime {
        RUNTIME_READY_CHECK
    } else {
        ""
    };
    let mut pending = imports;
    loop {
        let markers = run_probe_script(python, check, pending);
        let marker = |kind: &str| markers.iter().any(|(seen, _)| seen == kind);
        if !check.is_empty() {
            probe.runtime_ready = marker("ready");
        }
        probe.missing_imports.extend(
            markers
                .iter()
                .filter(|(kind, _)| kind == "missing")
                .map(|(_, name)| name.clone()),
        );
        if marker("done") {
            return probe;
        }
        // The interpreter died. Charge the import it was taking and resume after it; one that
        // died before any import (or never started) fails every import still pending.
        let started = markers.iter().filter(|(kind, _)| kind == "import").count();
        if started == 0 {
            probe
                .missing_imports
                .extend(pending.iter().map(ToString::to_string));
            return probe;
        }
        probe.missing_imports.push(pending[started - 1].to_string());
        pending = &pending[started..];
        check = "";
        if pending.is_empty() {
            return probe;
        }
    }
}

/// One run of [`PROBE_SCRIPT`]: its marker lines as `(kind, argument)`.
fn run_probe_script(python: &str, check: &str, imports: &[&str]) -> Vec<(String, String)> {
    let mut command = std::process::Command::new(python);
    command
        .arg("-")
        .arg(check)
        .args(imports)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    // Hidden window on Windows (TS `spawnHidden`).
    crate::platform::process::set_no_window(&mut command);
    if cfg!(windows) {
        command.env("PYTHONUTF8", "1");
    }
    let Ok(mut child) = command.spawn() else {
        return Vec::new();
    };
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write as _;
        // A failed write leaves the interpreter without a script: no markers, every check fails.
        let _ = stdin.write_all(PROBE_SCRIPT.as_bytes());
    }
    let Ok(output) = child.wait_with_output() else {
        return Vec::new();
    };
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.strip_prefix("pa-probe "))
        .map(|rest| {
            let (kind, argument) = rest.split_once(' ').unwrap_or((rest, ""));
            (kind.to_string(), argument.to_string())
        })
        .collect()
}

/// The runtime-ready assertion from the TS product: a current
/// prime-agent-runtime with the callable RLM surface, harness CRUD, bash
/// handles, and protocol version 3.
const RUNTIME_READY_CHECK: &str = "import inspect; import rlm; from rlm import McpIntegration; import rlm.mcp as mcp; from rlm.harness import HarnessEntry; _harness_methods = ['create_memory', 'update_memory', 'delete_memory', 'create_skill', 'update_skill', 'delete_skill', 'create_subagent', 'update_subagent', 'delete_subagent', 'create_prompt_note', 'update_prompt_note', 'delete_prompt_note', 'record_refinement']; _mcp_discovery_methods = ['list_plugins', 'search_plugins', 'list_connections', 'search_tools', 'describe_tool']; assert callable(mcp.list_tools); assert callable(mcp.call_tool); assert all(callable(getattr(mcp, _m, None)) for _m in _mcp_discovery_methods), \"rlm.mcp is missing MCP discovery methods (list_plugins, search_plugins, list_connections, search_tools, describe_tool); the kernel venv needs a current prime-agent-runtime\"; assert callable(rlm.spawn); assert hasattr(rlm, 'rlm'); assert callable(rlm.rlm.spawn); assert inspect.signature(rlm.spawn).parameters['name'].default is inspect.Parameter.empty; assert not hasattr(rlm, 'run'); assert not hasattr(rlm.rlm, 'run'); assert callable(rlm.host_request); assert callable(rlm.find_models); assert callable(rlm.rlm.find_models); assert callable(rlm.create_session); assert callable(rlm.rlm.create_session); assert callable(rlm.progress_note); assert callable(rlm.rlm.progress_note); assert callable(rlm.rename); assert callable(rlm.rlm.rename); assert hasattr(rlm, 'harness'); assert hasattr(rlm, 'get_harness_state'); assert hasattr(rlm.rlm, 'harness'); assert hasattr(rlm.rlm, 'get_harness_state'); assert all(callable(getattr(_harness, _method, None)) for _harness in (rlm.harness, rlm.rlm.harness) for _method in _harness_methods); assert 'reference' in HarnessEntry.__dataclass_fields__; assert 'scope' in HarnessEntry.__dataclass_fields__; assert 'reference' in inspect.signature(rlm.harness.create_skill).parameters; assert 'reference' in inspect.signature(rlm.harness.update_skill).parameters; assert 'global_' in inspect.signature(rlm.harness.create_memory).parameters; assert 'global_' in inspect.signature(rlm.get_harness_state).parameters; assert not hasattr(rlm, 'background'); assert not hasattr(rlm.rlm, 'background'); from rlm.bash import BashHandle, BashResult; assert callable(rlm.bash); assert {'persistent', 'yield_after'} <= set(inspect.signature(rlm.bash).parameters); assert callable(rlm.persistent_shells); assert callable(rlm._set_completion_notices_wired); assert all(callable(getattr(BashHandle, _m, None)) for _m in ('tail', 'output', 'poll', 'kill')); assert {'exit_code', 'output', 'duration'} <= set(BashResult.__dataclass_fields__); import rlm.repl as _repl; assert callable(_repl.main); assert callable(_repl.emit); assert callable(_repl.host_request); assert callable(_repl.is_active); assert _repl.PROTOCOL_VERSION == 3; assert callable(rlm.emit); assert not hasattr(rlm, 'HOST_COMM_TARGET'); assert not hasattr(mcp, 'install_shutdown_hook')";

/// Process-global memo of a successful runtime-ready probe, tiered above the cross-process on-disk
/// memo ([`super::super::disk_memo`]): the probe is a full interpreter start. Memoized on success
/// only: the key carries every input the probe observes; a failed kernel start drops both layers
/// ([`invalidate_runtime_probe_cache`]).
static RUNTIME_PROBE_MEMO: Mutex<Option<HashMap<String, PathBuf>>> = Mutex::new(None);

pub(super) fn lock_probe_memo() -> std::sync::MutexGuard<'static, Option<HashMap<String, PathBuf>>>
{
    RUNTIME_PROBE_MEMO
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Identity of the runtime as installed in the venv: the interpreter binary's stat plus a content
/// hash of the installed `rlm` package tree. Out-of-band damage changes this identity, so a
/// memoized probe result can never mask a mutated install.
pub(super) fn installed_runtime_identity(python: &Path, venv: &Path) -> String {
    let mut hasher = sha2::Sha256::new();
    match std::fs::metadata(python) {
        Ok(meta) => {
            let modified = meta
                .modified()
                .map_or_else(|_| "no-mtime".to_string(), |time| format!("{time:?}"));
            hasher.update(format!("py:{}:{}:{modified}", python.display(), meta.len()).as_bytes());
        }
        Err(error) => hasher.update(format!("py-error:{}:{error}", python.display()).as_bytes()),
    }
    for package in ["rlm", "dill"] {
        match installed_package_dir(venv, package) {
            Some(dir) => match hash_python_tree(&dir) {
                Ok(hash) => hasher.update(format!("{package}:{hash}").as_bytes()),
                Err(error) => hasher.update(format!("{package}-error:{error}").as_bytes()),
            },
            None => hasher.update(format!("{package}-missing").as_bytes()),
        }
    }
    format!("sha256:{:x}", hasher.finalize())
}

/// The installed `rlm` package under the venv's site-packages: the Windows layout
/// `<venv>/Lib/site-packages/rlm` (no python-version layer) or the Unix layout
/// `<venv>/lib/python*/site-packages/rlm`.
#[cfg(test)]
pub(super) fn installed_rlm_dir(venv: &Path) -> Option<PathBuf> {
    installed_package_dir(venv, "rlm")
}

pub(super) fn installed_package_dir(venv: &Path, package: &str) -> Option<PathBuf> {
    let lib = venv.join("lib");
    let windows_layout = lib.join("site-packages").join(package);
    if windows_layout.is_dir() {
        return Some(windows_layout);
    }
    let entries = std::fs::read_dir(&lib).ok()?;
    for entry in entries.flatten() {
        if !entry.file_type().is_ok_and(|t| t.is_dir()) {
            continue;
        }
        if !entry.file_name().to_string_lossy().starts_with("python") {
            continue;
        }
        let installed = entry.path().join("site-packages").join(package);
        if installed.is_dir() {
            return Some(installed);
        }
    }
    None
}

/// Content hash of a python tree: every `.py` file's relative path and
/// bytes, in sorted order (same witness shape as `hash_runtime_source`).
fn hash_python_tree(dir: &Path) -> anyhow::Result<String> {
    let mut files = Vec::new();
    collect_python_files(dir, &mut files)?;
    files.sort();
    let mut hasher = sha2::Sha256::new();
    for file in &files {
        let relative = file.strip_prefix(dir)?;
        hasher.update(relative.to_string_lossy().as_bytes());
        hasher.update([0]);
        hasher.update(&std::fs::read(file)?);
        hasher.update([0]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// The memo key: every input the runtime-ready probe observes.
pub(super) fn runtime_probe_key(
    python: &str,
    runtime_identity: &str,
    version_raw: &str,
    installed_identity: &str,
) -> String {
    format!(
        "{python}\u{0}{runtime_identity}\u{0}{installed_identity}\u{0}sha256:{:x}",
        sha2::Sha256::digest(version_raw.as_bytes())
    )
}

/// The runtime-ready check, memoized on success across two layers: the process-global map first,
/// then the on-disk cross-process memo. `version_raw` is the raw `.bootstrap-version` text the
/// caller already read. The key is recomputed fresh on every call so a hit skips only the two
/// interpreter probes. A caller-owned `PRIME_AGENT_KERNEL_PYTHON` override never reaches this.
pub(super) fn has_prime_agent_runtime_memoized(
    python: &str,
    runtime_identity: &str,
    version_raw: &str,
    installed_identity: &str,
    venv: &Path,
) -> bool {
    let key = runtime_probe_key(python, runtime_identity, version_raw, installed_identity);
    let memo_path = super::super::disk_memo::disk_memo_path(venv);
    if lock_probe_memo()
        .as_ref()
        .is_some_and(|memo| memo.contains_key(&key))
    {
        return true;
    }
    if super::super::disk_memo::disk_memo_hit(&memo_path, &key) {
        let mut memo = lock_probe_memo();
        let entries = memo.get_or_insert_with(HashMap::new);
        if entries.len() >= 16 {
            entries.clear();
        }
        entries.insert(key, memo_path);
        return true;
    }
    let probe = probe_python(python, true, &["dill"]);
    if !probe.runtime_ready || !probe.missing_imports.is_empty() {
        return false;
    }
    let mut memo = lock_probe_memo();
    let entries = memo.get_or_insert_with(HashMap::new);
    if entries.len() >= 16 {
        entries.clear();
    }
    super::super::disk_memo::disk_memo_write(&memo_path, &key);
    entries.insert(key, memo_path);
    true
}

/// Drop every memoized runtime-ready result, both layers: the in-process map dies with this call.
/// The next kernel start re-runs the probe.
pub fn invalidate_runtime_probe_cache() {
    let tracked: Vec<PathBuf> = lock_probe_memo()
        .take()
        .map(|memo| memo.values().cloned().collect())
        .unwrap_or_default();
    for path in tracked {
        super::super::disk_memo::disk_memo_invalidate(&path);
    }
}

/// Drop only the in-process memo layer, leaving the on-disk layer intact:
/// the fresh-process simulation the disk-memo oracles use. Unix only.
#[cfg(all(test, unix))]
pub(crate) fn clear_in_process_probe_memo_for_tests() {
    *lock_probe_memo() = None;
}
