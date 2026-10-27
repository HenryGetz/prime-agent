//! Client-side info displays: the `/session`, `/context`, `/system-prompt`,
//! `/logs`, and `/changelog` rows. Row data and render are pure; the
//! session UI owns the daemon fetches, and the read-only info panel owns
//! the paint (the operator's 2026-09-26 directive). Every builder
//! returns one [`ClientLine`] per source line, spans carrying their
//! theme color so the view resolves them at render time.

use std::path::Path;

use serde_json::Value;

use crate::theme::{Theme, ThemeColor};
use crate::{Line, Span};
use ratatui::style::Style;

mod context_tree;
#[cfg(test)]
mod tests;

#[cfg(test)]
use context_tree::truncate_plain;

pub use context_tree::{context_tree_rows, ContextTreeScope};

/// One styled segment of a client info row: text plus its theme color
/// (`None` keeps the default foreground).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientSpan {
    pub text: String,
    pub color: Option<ThemeColor>,
}

impl ClientSpan {
    fn raw(text: impl Into<String>) -> Self {
        ClientSpan {
            text: text.into(),
            color: None,
        }
    }

    fn colored(text: impl Into<String>, color: ThemeColor) -> Self {
        ClientSpan {
            text: text.into(),
            color: Some(color),
        }
    }
}

/// One source line of a client info block; the view wraps each line
/// separately (the TS `Text` component wraps each newline-delimited line).
pub type ClientLine = Vec<ClientSpan>;

fn raw_span(text: impl Into<String>) -> ClientSpan {
    ClientSpan::raw(text)
}

fn dim(text: impl Into<String>) -> ClientSpan {
    ClientSpan::colored(text, ThemeColor::Dim)
}

/// Digits grouped with commas (`toLocaleString` for the en-US locale).
pub(crate) fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// JS `Number.prototype.toFixed(digits)` over a non-negative value: the
/// EXACT decimal expansion of the binary double, rounded half away from
/// zero at the digit. Neither Rust's `{:.n}` (ties half-to-even on the
/// exact expansion) nor a float multiply-then-round (the multiply rounds
/// too) matches, so the rounding runs on the double's exact rational
/// value: `value = mantissa / 2^exponent` reduces to one integer divide
/// with a half-away tie on the remainder.
#[must_use]
pub fn js_to_fixed(value: f64, digits: usize) -> String {
    let bits = value.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i64;
    let (mantissa, exponent) = if biased == 0 {
        (bits & ((1u64 << 52) - 1), -1074i64)
    } else {
        ((bits & ((1u64 << 52) - 1)) | (1u64 << 52), biased - 1075)
    };
    let numerator = u128::from(mantissa) * 5u128.pow(digits as u32);
    // value * 10^digits = numerator * 2^(exponent + digits): a left
    // shift when the exponent absorbs the scale, else one exact divide
    // with a half-away tie on the remainder.
    let shift = exponent + digits as i64;
    let mut scaled = if shift >= 0 {
        // Spend values stay far inside u128; saturating guards the
        // denormal edge without panicking.
        numerator
            .checked_shl(shift as u32)
            .filter(|_| shift <= 100)
            .unwrap_or(u128::MAX)
    } else {
        let shift = (-shift) as u32;
        if shift > 127 {
            0
        } else {
            let denom = 1u128 << shift;
            let quotient = numerator / denom;
            let remainder = numerator % denom;
            // Ties round away from zero (JS rounds half toward the larger n).
            quotient + u128::from(2 * remainder >= denom)
        }
    };
    if value == 0.0 {
        scaled = 0;
    }
    let unit = 10u128.pow(digits as u32);
    let integer = scaled / unit;
    let fraction = scaled % unit;
    if digits == 0 {
        return format!("{integer}");
    }
    format!("{integer}.{fraction:0digits$}")
}

/// The `/session` info rows over the `get_session_stats` shape. A
/// missing `sessionFile` renders as `In-memory`; an unset session name
/// omits the `Name:` row.
pub fn session_info_rows(stats: &Value, session_name: Option<&str>) -> Vec<ClientLine> {
    let count = |field: &str| stats.get(field).and_then(Value::as_u64).unwrap_or_default();
    let session_id = stats
        .get("sessionId")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let session_file = stats
        .get("sessionFile")
        .and_then(Value::as_str)
        .filter(|file| !file.is_empty())
        .unwrap_or("In-memory");
    let mut rows = vec![vec![raw_span("Session Info")], vec![]];
    if let Some(name) = session_name.filter(|name| !name.is_empty()) {
        rows.push(vec![dim("Name:"), raw_span(format!(" {name}"))]);
    }
    rows.push(vec![dim("File:"), raw_span(format!(" {session_file}"))]);
    rows.push(vec![dim("ID:"), raw_span(format!(" {session_id}"))]);
    rows.push(vec![]);
    rows.push(vec![raw_span("Messages")]);
    for (label, value) in [
        ("User:", count("userMessages")),
        ("Assistant:", count("assistantMessages")),
        ("Tool Calls:", count("toolCalls")),
        ("Tool Results:", count("toolResults")),
        ("Total:", count("totalMessages")),
    ] {
        rows.push(vec![dim(label), raw_span(format!(" {value}"))]);
    }
    rows.push(vec![]);
    rows.push(vec![dim(
        "Use /context for token, cost, and context usage.",
    )]);
    rows
}

/// The `/logs` info rows: the logs directory, its files sorted by name
/// with `(N KB)` sizes, and the trailing note.
#[must_use]
pub fn logs_rows(logs_dir: &Path) -> Vec<ClientLine> {
    let mut rows = vec![
        vec![raw_span("Logs")],
        vec![],
        vec![
            dim("Directory:"),
            raw_span(format!(" {}", logs_dir.display())),
        ],
        vec![],
    ];
    // A readdir failure falls through to the empty-state row (the TS catch
    // keeps rendering); dot-prefixed files stay hidden.
    let mut files: Vec<String> = std::fs::read_dir(logs_dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .filter(|name| !name.starts_with('.'))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    if files.is_empty() {
        rows.push(vec![dim("No logs written yet.")]);
    } else {
        for name in files {
            let mut row = vec![dim("\u{2022}"), raw_span(format!(" {name}"))];
            // A file vanishing between readdir and stat loses only its
            // size (the TS catch skips the size, keeps the row).
            if let Ok(metadata) = std::fs::metadata(logs_dir.join(&name)) {
                row.push(raw_span(" "));
                row.push(dim(format!(
                    "({} KB)",
                    js_to_fixed(metadata.len() as f64 / 1024.0, 1)
                )));
            }
            rows.push(row);
        }
    }
    rows.push(vec![]);
    rows.push(vec![dim(
        "Daemon crashes log to <socket>.log; agent-open failures log to client-errors.log.",
    )]);
    rows
}

/// `/usage`: the `OpenCode` Go subscription windows from the persisted routing
/// state plus the stacked accounts' labels from `auth.json`. Offline and
/// pure: the background poll owns every network call.
#[must_use]
pub fn opencode_go_usage_rows(agent_dir: &Path) -> Vec<ClientLine> {
    let now = epoch_ms();
    let routing = read_json(&agent_dir.join("opencode-go-routing.json"));
    let auth = read_json(&agent_dir.join("auth.json"));
    let accounts = routing
        .as_ref()
        .and_then(|state| state.get("accounts"))
        .and_then(Value::as_object);
    let auth_slots = auth.as_ref().and_then(Value::as_object);

    let mut slots: Vec<String> = Vec::new();
    for keys in [
        auth_slots.map(|slots| slots.keys()),
        accounts.map(|slots| slots.keys()),
    ]
    .into_iter()
    .flatten()
    {
        slots.extend(keys.filter(|slot| is_opencode_go_slot(slot)).cloned());
    }
    slots.sort();
    slots.dedup();
    slots.sort_by_key(|slot| opencode_go_slot_key(slot));

    let mut rows = vec![
        vec![raw_span("OpenCode Go usage (local accounting)")],
        vec![],
    ];
    let fetched = routing
        .as_ref()
        .and_then(|state| state.get("usageFetchedAtMs"))
        .and_then(Value::as_i64)
        .filter(|fetched| *fetched > 0)
        .or_else(|| {
            accounts
                .into_iter()
                .flat_map(|accounts| accounts.values())
                .filter_map(|account| account.get("usedPctAtMs").and_then(Value::as_i64))
                .max()
                .filter(|last| *last > 0)
        });
    rows.push(vec![
        dim("last poll:"),
        raw_span(format!(" {}", age_label(now, fetched.unwrap_or(0)))),
    ]);
    rows.push(vec![]);

    if slots.is_empty() {
        rows.push(vec![raw_span(
            "No OpenCode Go usage yet — send a request or /login",
        )]);
        rows.push(vec![]);
        rows.push(vec![dim(
            "Windows are per account; the poll runs in the background.",
        )]);
        return rows;
    }

    for slot in slots {
        let label = auth_slots
            .and_then(|auth| auth.get(&slot))
            .and_then(|credential| {
                credential
                    .get("email")
                    .and_then(Value::as_str)
                    .or_else(|| credential.get("orgName").and_then(Value::as_str))
            })
            .filter(|label| !label.is_empty());
        let account = accounts.and_then(|accounts| accounts.get(&slot));
        let mut header = vec![raw_span(slot.clone())];
        if let Some(label) = label {
            header.push(dim(format!(" — {label}")));
        }
        rows.push(header);

        let windows = account
            .and_then(|account| account.get("windows"))
            .and_then(Value::as_object);
        let mut worst: Option<f64> = None;
        for window in ["rolling", "weekly", "monthly"] {
            let Some(detail) = windows
                .and_then(|windows| windows.get(window))
                .and_then(Value::as_object)
            else {
                continue;
            };
            let Some(percent) = detail.get("percent").and_then(Value::as_f64) else {
                continue;
            };
            let status = detail
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let resets_at = detail
                .get("resetsAt")
                .and_then(Value::as_str)
                .unwrap_or_default();
            worst = Some(worst.map_or(percent, |current| current.max(percent)));
            rows.push(vec![
                dim(format!("  {window}:")),
                raw_span(format!(" {}%", js_to_fixed(percent, 1))),
                dim(format!("  {status}  resets {resets_at}")),
            ]);
        }
        match worst {
            Some(worst) => rows.push(vec![
                dim("  state:"),
                raw_span(format!(
                    " {} ({}%)",
                    window_state(worst),
                    js_to_fixed(worst, 1)
                )),
            ]),
            None => rows.push(vec![dim("  no data — waiting for the first poll")]),
        }
        rows.push(vec![]);
    }
    rows.push(vec![dim(
        "Windows are per account; the poll runs in the background.",
    )]);
    rows
}

/// `opencode-go` or `opencode-go#N` (a stacked Go slot, not another provider).
fn is_opencode_go_slot(slot: &str) -> bool {
    slot == "opencode-go"
        || slot
            .strip_prefix("opencode-go#")
            .is_some_and(|number| number.parse::<u64>().is_ok())
}

/// The base slot first, then `#N` ascending.
fn opencode_go_slot_key(slot: &str) -> (u8, u64) {
    slot.strip_prefix("opencode-go#")
        .and_then(|number| number.parse::<u64>().ok())
        .map_or((0, 0), |number| (1, number))
}

/// The worst-window state: `limited` at 100%, `warning` from 80%, else ready.
fn window_state(worst_percent: f64) -> &'static str {
    if worst_percent >= 100.0 {
        "limited"
    } else if worst_percent >= 80.0 {
        "warning"
    } else {
        "ready"
    }
}

/// `12s ago` / `3m ago` / `2h 5m ago` / `3d ago`; `never` for no poll.
fn age_label(now_ms: i64, then_ms: i64) -> String {
    if then_ms <= 0 {
        return "never".to_string();
    }
    let seconds = now_ms.saturating_sub(then_ms).max(0) / 1_000;
    if seconds < 60 {
        format!("{seconds}s ago")
    } else if seconds < 3_600 {
        format!("{}m ago", seconds / 60)
    } else if seconds < 86_400 {
        format!("{}h {}m ago", seconds / 3_600, (seconds % 3_600) / 60)
    } else {
        format!("{}d ago", seconds / 86_400)
    }
}

fn epoch_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
        })
}

fn read_json(path: &Path) -> Option<Value> {
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// The `/system-prompt` header rows; the char count is the JS string
/// length (UTF-16 code units).
#[must_use]
pub fn system_prompt_header_rows(prompt: &str) -> Vec<ClientLine> {
    let chars = prompt.encode_utf16().count();
    vec![vec![
        raw_span("System Prompt "),
        dim(format!("({chars} chars)")),
    ]]
}

/// The `/system-prompt` body rows: the prompt split into source lines for per-line wrapping.
#[must_use]
pub fn system_prompt_body_rows(prompt: &str) -> Vec<ClientLine> {
    prompt
        .split('\n')
        .map(|line| vec![raw_span(line)])
        .collect()
}

/// The `/changelog` markdown: the CHANGELOG.md entries newest-first
/// joined with a blank line, or the empty-state text.
#[must_use]
pub fn changelog_markdown(changelog_path: &Path) -> String {
    let entries = parse_changelog(changelog_path);
    if entries.is_empty() {
        return "No changelog entries found.".to_string();
    }
    entries
        .iter()
        .rev()
        .cloned()
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Parse the shipped CHANGELOG.md: sections under `## ` headers, each
/// entry the trimmed section text including its header line. A `## `
/// header without a parsable `x.y.z` version resets collection; lines
/// before the first version header stay dropped.
fn parse_changelog(changelog_path: &Path) -> Vec<String> {
    let Ok(content) = std::fs::read_to_string(changelog_path) else {
        return Vec::new();
    };
    let mut entries: Vec<String> = Vec::new();
    let mut current: Option<Vec<String>> = None;
    for line in content.split('\n') {
        if let Some(rest) = line.strip_prefix("## ") {
            if let Some(lines) = current.take() {
                push_entry(&mut entries, &lines);
            }
            if is_version_header(rest) {
                current = Some(vec![line.to_string()]);
            }
        } else if let Some(lines) = current.as_mut() {
            lines.push(line.to_string());
        }
    }
    if let Some(lines) = current {
        push_entry(&mut entries, &lines);
    }
    entries
}

fn push_entry(entries: &mut Vec<String>, lines: &[String]) {
    let trimmed = lines.join("\n").trim().to_string();
    if !trimmed.is_empty() {
        entries.push(trimmed);
    }
}

/// Whether a `## ` header rest carries an `x.y.z` version (TS
/// `/##\s+\[?(\d+)\.(\d+)\.(\d+)\]?/`): optional whitespace, an optional
/// `[`, then major.minor.patch.
fn is_version_header(rest: &str) -> bool {
    let mut rest = rest.trim_start();
    rest = rest.strip_prefix('[').unwrap_or(rest);
    for part in 0..3 {
        let digits = rest.chars().take_while(char::is_ascii_digit).count();
        if digits == 0 {
            return false;
        }
        rest = &rest[digits..];
        if part < 2 {
            rest = rest.strip_prefix('.').unwrap_or_default();
        }
    }
    true
}

/// Resolve one info row to styled spans.
fn styled_spans(row: &[ClientSpan], theme: &Theme) -> Line {
    row.iter()
        .map(|span| match span.color {
            Some(color) => theme.fg(color, span.text.clone()),
            None => Span::raw(span.text.clone()),
        })
        .collect()
}

/// `Spacer(1)` + `Text(info, 1, 0)`: one blank row, then each source
/// line wrapped at `width - 2` with a one-column margin on each side
/// and rows padded to the full width (continuation rows pad inside the
/// open style, the last wrapped row after the segment's reset).
#[must_use]
pub fn render_client_text(rows: &[ClientLine], theme: &Theme, width: usize) -> Vec<Line> {
    let content_width = width.saturating_sub(2).max(1);
    let mut out: Vec<Line> = vec![Vec::new()];
    for row in rows {
        let styled = styled_spans(row, theme);
        let wrapped = crate::width::wrap_line(&styled, content_width);
        let row_count = wrapped.len();
        for (index, mut line) in wrapped.into_iter().enumerate() {
            let padding_style = if index + 1 < row_count {
                line.last().map_or(Style::default(), |span| span.style)
            } else {
                Style::default()
            };
            let mut row: Line = vec![Span::raw(" ")];
            row.append(&mut line);
            out.push(crate::chat::pad_to(row, width, padding_style));
        }
    }
    out
}
