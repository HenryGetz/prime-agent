- OpenCode Go (Console) subscriptions are now a provider: the device-code login
  runs through `/login` → **OpenCode Go**, several accounts stack as
  `opencode-go`, `opencode-go#2`, …, and requests route to the session's pinned
  or warm-cache account with usage-driven spill, workspace attribution
  (`x-opencode-org-id`), harness session identity, and 401/402/403 verdicts
  that rotate to a healthy sibling.
- OpenCode Go models now come from the live catalog
  (`<agent_dir>/opencode-go-catalog.json`, 24 h stale-while-revalidate, the
  embedded roster as the fallback), retired models are learned from settled
  410/deprecation failures, and a settled 429 demotes the serving account
  immediately instead of waiting for the usage poll.
- The OpenCode Go wire contract now comes from the shared `opencode-go-spec` checkout; generated constants keep the omp-go plugin and this provider in lockstep (endpoints, headers, transports, catalog, usage, routing policy).
- The TUI `/usage` command (aliases `/opencode-usage`, `/go-usage`) renders
  per-account rolling/weekly/monthly OpenCode Go windows, statuses, reset
  instants, and the worst-window state from local routing state; it performs no
  network I/O, and `/context` remains token/cost.
- OpenCode session identity headers are applied whenever the provider id is
  `opencode`/`opencode-go` or the model base URL is an OpenCode gateway, so
  custom/alias provider rows no longer fail with
  `400 Request is missing x-opencode-session`.
