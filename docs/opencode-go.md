# OpenCode Go subscriptions

The wire contract (Console/gateway endpoints, OAuth grants, header names,
catalog URL/cache policy, per-`npm` transports, hidden transports/models, usage
windows/statuses, routing TTLs) comes from the shared
`opencode-go-spec/spec/opencode-go.v1.json`. `tools/sync.py` generates
`crates/pa-ai/src/opencode_go_spec_generated.rs`; never hand-edit it. Change the
spec once and regenerate both clients:

```
python3 "$OPENCODE_GO_SPEC_DIR/tools/sync.py" \
  --omp-go /path/to/omp-go --prime-agent .
```

`opencode-go` is the OpenCode Go (Console) subscription provider. It serves the
Go gateway at `https://opencode.ai/zen/go/v1` with the Console access token as
the bearer key; there is no separate exchange.

## Login

`/login` → **OpenCode Go** runs the Console device-code flow
(`crates/pa-ai/src/oauth/opencode.rs`):

1. `POST https://opencode.ai/console/auth/device/code` with
   `{"client_id":"opencode-cli","supports_org_scope":true}`.
2. The panel shows the verification URL and `Enter code: …`; the first token
   poll waits the response's `interval` seconds.
3. `POST /auth/device/token` polls the device grant. Poll bodies are parsed
   regardless of HTTP status (`authorization_pending` keeps waiting,
   `slow_down` adds five seconds).
4. `GET /api/user` and `GET /api/orgs` run in parallel with the fresh token.
   The workspace is the token's `org_id` when present (it must exist in the org
   list), otherwise the first org sorted by name then id.
5. The credential stores the access token, the rotating refresh token, the
   account id/email, and the workspace (`orgId`), with `apiEndpoint` as the
   Console base the login ran against.

Every Console request carries the harness fingerprint (`user-agent:
opencode/latest/2.0.26/cli`, `b3`, `traceparent`). Refresh posts the
`refresh_token` grant with an 8 s timeout and maps `400 {"error":"invalid_grant"}`
to "OpenCode Console session expired; log in again".

## Stacking accounts

The credential is stored under a slot: `opencode-go`, `opencode-go#2`,
`opencode-go#3`, … A login for an account whose `accountId` or `orgId` is
already stored replaces that slot; a different account appends the next free
slot. `/logout` names each stacked row from the stored credential:
**OpenCode Go — <email>** when the login captured an email, otherwise
**OpenCode Go (account N)** (`opencode-go#N` is account N; the base slot is
account 1). `/login` keeps one subscription row for every slot and its status
reads `configured (N accounts)` once more than one slot is stored (the
single-slot label stays `configured`). A successful login names the account
(`Logged in to OpenCode Go as <email> (<workspace>)`) when the Console resolved
the email and workspace, and falls back to the TS status message otherwise.

## Routing and usage

State lives at `<agent_dir>/opencode-go-routing.json` (mode 0600, atomic
write): per-slot `{orgId, lastServedAtMs, usedPct, usedPctAtMs, verdict,
windows}` and top-level `usageFetchedAtMs`; `windows` is a map keyed `rolling`,
`weekly`, `monthly`, each `{percent, status, resetsAt}`, written by the same
`GET /v1/usage` poll that computes `usedPct`. The state also holds per-session
pins `{sessionId: {slot, at}}`.

- A session's pin wins while it is younger than seven days and the slot has no
  fresh verdict (drain in place: the live session keeps its warm prompt cache).
- A new session joins the warm leader: an account with no fresh verdict and
  `usedPct < 1`, preferring accounts below the spill threshold, sorted by
  `lastServedAtMs` descending, then `usedPct`, then slot.
- The spill threshold is a percentage: `PRIME_OPENCODE_GO_SPILL_PCT`, or
  `OMP_GO_SPILL_PCT` (legacy), default 90.
- `GET /v1/usage` runs on a detached thread per account (Bearer +
  `x-opencode-session`, 5 s timeout); the bearer is the OAuth access token or
  a slot's literal API key (a `!`-command key is never polled), and a slot with
  no workspace id is polled without `x-opencode-org-id` so its 401/402/403
  still demotes it. The worst of the rolling/weekly/monthly windows is stored
  as `usedPct`. A slot is polled when its usage is unknown or older than five
  minutes (`USAGE_REFRESH_MS`), so an exhausted account heals once its window
  resets (an attempt inside `PRIME_RETRY_MS` still suppresses the poll).
  Transient failures keep the last state; 401/402/403 record a 30-minute
  verdict. The routed request path never waits on the poll: a first chat
  against a pre-existing `auth.json` may start on unknown percentages until
  the background usage poll lands. A 429 turn retries after the poll lands
  (the poll rotates the account by the next turn); 401/402/403 are recorded as
  verdicts immediately. The `/login` flow waits for the prime after the
  credential write, so a real login starts from real percentages.
- An inference 401/402/403 records the same verdict and drops the pins that
  pointed at the account, so the next turn rotates to a healthy sibling.
- A settled 429 demotes the serving slot immediately: `rateLimitedUntilMs`
  (retry-after, or 60 s), `usedPct = 1`, and its pins dropped, so the next turn
  rotates without waiting for the usage poll. Selection skips a slot while
  `now < rateLimitedUntilMs`; a poll reporting `usedPct < 1` clears it.

## Models

The catalog is live. `crates/pa-ai/src/opencode_go_catalog.rs` fetches
`https://models.opencode.ai/api.json` (unauthenticated, harness fingerprint
headers, 5 s timeout, 8 MiB cap, no redirects), converts the `opencode-go`
entry's non-deprecated models, and caches them at
`<agent_dir>/opencode-go-catalog.json` (`{fetchedAt, models}`, mode 0600,
atomic). The cache is authoritative for 24 h: `ModelRegistry::load_models`
serves it immediately and a stale cache triggers one detached refresh per agent
directory per process (single-flight); a failed fetch keeps the last good
cache. Without a cache the embedded 31-model roster
(`crates/pa-ai/src/opencode_go_models.rs`) is used and a fetch is scheduled.
Custom `models.json` entries still override both.

Conversion follows the embedded roster: provider `opencode-go`; the transport is selected by
`provider.npm`: `@ai-sdk/openai` -> `openai-responses` on
`https://opencode.ai/zen/go/v1`, `@ai-sdk/anthropic` -> `anthropic-messages` on
`https://opencode.ai/zen/go`, `@ai-sdk/google` is hidden (the gateway rejects it on
chat/completions and serves no Gemini route), and a missing or other hint ->
`openai-completions` on `https://opencode.ai/zen/go/v1`. The catalog's per-model inference
`api` path is ignored. Text/image inputs come from `modalities`, context/output from
`limit.context`/`limit.output`, base cost from `cost`, and `thinking_level_map` from the
`reasoning_options` effort rungs intersected with Prime's levels (unsupported rungs are null;
toggle-only models map every rung to null, so no effort rides their requests). Deprecated
entries are skipped; an entry without an id or a limit is skipped; a malformed document or an
absent `opencode-go` entry keeps the cache/roster.

A settled inference failure that is HTTP 410 or mentions `ModelDeprecated` /
"has been deprecated" (case-insensitive, the settled message's `model` field)
is learned as dead: `<agent_dir>/opencode-go-dead-models.json` (JSON array,
0600, atomic, best-effort) and filtered out of the roster and the cache. A
plain 404 is a routing blip, and another provider's 410 marks nothing (the hook
filters on `provider == "opencode-go"`).

Requests carry the harness identity header set — `x-opencode-session`,
`x-opencode-session-id`, `x-session-affinity`, `x-session-id`,
`x-opencode-client: cli`, and the harness User-Agent — when the provider id is
`opencode`/`opencode-go` **or** the model base URL points at an OpenCode
gateway (`opencode.ai/zen/go`, `/inference/go`, `/zen`, `/inference`), so a
custom or alias provider row using an OpenCode gateway still gets fingerprinted.
This fixes `400 Request is missing x-opencode-session`. `x-opencode-session` is
a harness-shaped `ses_…` id derived from the conversation id when it is not
already in that form. The derivation is deterministic — SHA-256 of a fixed
domain prefix plus the conversation id, rendered as 12 lowercase hex chars
followed by 14 base62 chars — so a restarted worker sends the same id for the
same conversation instead of a fresh random one. Requests also carry the
workspace attribution `x-opencode-org-id` of the serving account, and the
Go-gateway compat (`store: false`, `max_completion_tokens`, streaming usage).

## Usage surface

`/usage` (aliases `/opencode-usage`, `/go-usage`) renders a client-side panel
built from `<agent_dir>/opencode-go-routing.json` and
`<agent_dir>/auth.json`. It performs no network I/O: the detached usage poll
described above owns refreshes, and the panel only reports what that poll
stored.

Each stacked slot gets one block, labeled by the account email, else its
`orgName`, else the slot id. Inside a block, `rolling` / `weekly` / `monthly`
rows show the window percent, the server status, and the reset instant; a
worst-window `state:` line reads `ready`, `warning` at >= 80%, or `limited` at
100%, and `no data — waiting for the first poll` when the slot has no windows
yet.

The panel header shows `last poll: <age>`, taken from `usageFetchedAtMs` /
`usedPctAtMs`, or `never` when no poll has landed. `/context` remains the
token/cost/agent-tree surface and does not alias `/usage`.
