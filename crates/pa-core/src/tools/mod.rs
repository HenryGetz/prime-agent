//! All modules are `pub(crate)` internals of the tools subsystem; the
//! crate facade (crate root) re-exports only the public tool surface.

pub(crate) mod bash;
pub(crate) mod bash_guard;
pub(crate) mod bash_local;
pub(crate) mod golden_replay;
pub(crate) mod ipython;
pub(crate) mod output_accumulator;
#[allow(dead_code)] // util module; every fn exercised by unit tests
pub(crate) mod path_utils;
pub(crate) mod render_utils;
#[allow(dead_code)] // util module; fully exercised by the bash unit tests
pub(crate) mod shell_utils;
pub(crate) mod tool_definition;
pub(crate) mod truncate;
