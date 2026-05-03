# D-013 OQ-31 Audit

Worktree: `/tmp/d013-oq31-audit/` (HEAD `984cf7aef`)
Date: 2026-05-03

## Step 1 — Activation writers BEFORE `_handle_message_with_agent`

`_handle_message_with_agent` is invoked at `gateway/run.py:3930` from inside
`handle_message`. The `was_auto_reset` detection + the round-7 unconditional
`_clear_session_activation(session_key)` runs at lines 4131–4232 (clear at
**line 4230**), as part of `_handle_message_with_agent`.

A "writer" is anything that calls `record_activation(...)` or assigns
`_session_skill_context[session_key] = …` BEFORE line 4230 executes for the
current turn.

### Writer paths discovered

After exhaustive grep across the repo (`grep -rn "record_activation\|_session_skill_context\["`)
there are exactly **2** writers, both reachable from a single inbound message:

#### Writer 1 — Slash-skill handler (`gateway/run.py:3842–3876`)

Site: inside `handle_message` (caller of `_handle_message_with_agent`).
- Writes `_session_skill_context[_quick_key] = {...}` (line 3852).
- Writes `_ssdb.record_activation(_quick_key, _skill_name, …, source="slash")` (line 3864).

Order of execution for `/skill-foo bar`:
1. handle_message receives event.
2. Slash dispatcher runs at L3805–3911. Slash skill matched.
3. **Activation persisted (cache + DB) — L3842/3864**.
4. `event.text` rewritten to skill invocation prompt.
5. Falls through past dispatcher.
6. L3925: sentinel claim.
7. L3930: `await self._handle_message_with_agent(event, source, _quick_key, _run_generation)`.
8. Inside that handler, L4131 detects `was_auto_reset=True` (because this message is the first after idle/daily/suspended).
9. **L4230: `self._clear_session_activation(session_key)` deletes step 3.**
10. L4253: `_auto = getattr(event, "auto_skill", None)` — slash-skill events do NOT carry `event.auto_skill` (the slash dispatch path rewrites `event.text`, never sets `auto_skill`). So the auto_skill repopulator block (4254–4292) does not fire.
11. Result: `pre_memory_write` reads back empty activation → fail-closed (no project resolution) for this turn.

**Auto-reset clears this writer? YES (this is OQ-31).**
**Severity: HIGH.** The slash skill the user just invoked is the canonical
trusted activation source, and it is silently discarded by the auto-reset
clear. The user's first slash skill after idle/daily/suspended runs without
`active_project`, breaking project-scoped memory writes for that turn. This
is exactly the codex round-7 finding (conf 0.92).

Critical sub-finding: slash-skill events have NO mechanism to repopulate
after the clear. The `event.auto_skill` repopulator block at L4253 only
fires for channel-bound auto skills (resolved by platform adapters from
topic/channel bindings), not for explicit `/skill-foo` invocations. So
once the round-7 clear deletes the writer-1 activation, it stays gone for
the entire current turn.

#### Writer 2 — auto_skill block (`gateway/run.py:4253–4292`)

Site: **inside** `_handle_message_with_agent`, **AFTER** the
`was_auto_reset` block at L4162–4232. Writes:
- `_session_skill_context[session_key] = {...}` (line 4266).
- `_ssdb.record_activation(session_key, _primary, …, source="auto_skill")` (line 4276).

Order of execution for an auto_skill turn:
1. ... → L4131 `was_auto_reset` check.
2. L4230 clears whatever was there.
3. L4253–4292 RE-records the auto_skill activation.

**Auto-reset clears this writer? YES, but it is RE-RECORDED unconditionally on the same turn.**
**Severity: NONE.** The auto_skill path self-heals because the clear and
the (re)record are correctly ordered.

### Channel/topic bindings

Channel-bound skills surface to gateway via `event.auto_skill` (set by
`gateway/platforms/telegram.py`'s `topic_skill` resolution and
`gateway/platforms/discord.py`'s `_resolve_channel_skills`). They
therefore go through writer 2 (the auto_skill repopulator), which is
already correctly ordered. Not a separate writer.

### Other write surfaces searched (NEGATIVE)

- `record_activation` callers: only the two above (slash + auto_skill in
  `gateway/run.py`, plus the helper definition itself in `skill_state_db.py`).
- `_session_skill_context[…] =` assignments: only L3852 (slash) and L4266 (auto_skill).
- `apply_skill_context` (the **read** path at L10128) is downstream of both
  writers and does not itself persist anything.

### Total writers pre-`was_auto_reset`: **1** (slash-skill).

This is below the 4-path hard-stop threshold; design proceeds.

## Step 2 — Design decision

### Approach A: Reorder operations

Move the slash-skill `record_activation` + cache write from L3842–3876 (in
`handle_message`) to **after** the `was_auto_reset` clear (i.e. into
`_handle_message_with_agent`, after L4230).

**Pros:**
- Eliminates the order-of-operations bug structurally — the writer simply
  cannot precede the clear.
- Mirrors the existing auto_skill block's correct ordering (writer 2 is
  already after L4230).

**Cons:**
- Slash-skill detection happens in `handle_message` (the dispatcher), but
  `_skill_name` / `_quick_key` would have to be threaded into
  `_handle_message_with_agent` somehow — either via a new param, an
  attribute on `event`, or a transient member on the runner.
- Threading a new arg through `_handle_message_with_agent` (which already
  takes `event, source, _quick_key, _run_generation`) means a wider blast
  radius of touched code than ideal.
- The `event.text` rewrite is also done by the slash dispatcher; only
  the persist step needs to defer. So the slash dispatcher would do the
  text rewrite and stash skill metadata onto `event` (e.g.
  `event._slash_skill_to_record = (skill_name, channel_id, project)`),
  and `_handle_message_with_agent` would consume and clear that field
  after L4230. Workable but adds a coupling.

### Approach B: Guard pattern ("protect-current-turn activation")

Add a "this activation was set in the current turn" signal so the L4230
clear can skip the writer-1 row.

Concretely: when the slash-skill handler persists at L3842/3864, also
record the `session_key` on a per-turn set, e.g.
`self._slash_activation_this_turn: Set[str] = set()` (initialised in
`__init__`, populated by writer 1, consumed and cleared inside the
auto-reset block at L4230 — "skip clear if session_key in
this_turn_set; then drop session_key from the set").

**Pros:**
- Local, surgical change. No threading of new args. No restructuring of
  the slash dispatcher.
- Plays well with the existing source-level guard test
  (`test_run_py_clears_activation_on_was_auto_reset_path`) which still
  finds `_clear_session_activation` inside the `was_auto_reset` block —
  the guard wraps the call, doesn't move it.
- Naturally extensible: if a future writer-3 emerges, just add it to the
  same set.

**Cons:**
- Introduces shared mutable state on the runner (`_slash_activation_this_turn`).
  Concurrency risk: if two messages from the same `session_key` race
  through the slash handler before either reaches L4230, the set could be
  drained too early. **Mitigation:** the runner already serializes the
  same `session_key` via `_running_agents` sentinel claim at L3925 — a
  second message blocks on the running-agents guard before reaching the
  slash dispatcher branch that writes the activation. So per-session_key
  serialization is already provided structurally.
- The guard set is keyed on session_key; we MUST clear the entry whether
  or not the auto-reset path runs (otherwise stale entries accumulate).
  Easiest cleanup point is in the `finally:` of `handle_message` (L3936).

### Decision: **Approach B (guard).**

Rationale:
- Both approaches are correct, but B is **smaller** and **does not move
  load-bearing logic across function boundaries**. Round-7's mistake was
  a structural-ordering error inside one function, and B fixes the
  structural-ordering error inside that same function — clean root-cause
  fix.
- B keeps the slash-skill activation persist at the slash dispatcher, which
  is the natural site for it (it's where skill identity is known and
  resolved). Approach A would smuggle metadata across functions, which
  is the kind of hidden-coupling regression we are trying to avoid.
- B preserves the existing source-level test guard
  (`test_run_py_clears_activation_on_was_auto_reset_path`) without
  modification — the clear call still appears in the `was_auto_reset`
  block; we just make it conditional.
- Per-session_key serialization is already enforced via the running-agents
  sentinel claim, so the concurrency cost of B is zero in practice.
- Codex round-7's third option ("reapply current slash activation
  immediately after clearing") would require either re-fetching the just-
  written DB row or stashing it on `event` — both equivalent in shape to
  approach A's metadata smuggling.

## Step 3 — Implementation plan

1. Add `self._slash_activation_this_turn: Set[str] = set()` in `__init__`.
2. In the slash-skill handler (L3842/3864), after a successful
   `record_activation`, `self._slash_activation_this_turn.add(_quick_key)`.
3. In `_handle_message_with_agent` at the auto-reset block (L4230 area):
   ```python
   if session_key in self._slash_activation_this_turn:
       # Don't drop the slash-skill activation written by THIS turn —
       # OQ-31. The clear is meant to drop activations from the
       # *expired* session_id, not from a writer that ran milliseconds
       # ago for the fresh session.
       self._slash_activation_this_turn.discard(session_key)
   else:
       self._clear_session_activation(session_key)
   ```
4. In `handle_message`'s `finally:` cleanup (L3936 onwards), discard the
   set entry to handle paths where the slash skill ran but `_handle_message_with_agent` was skipped (e.g. an early return inside the slash branch).
   Actually — re-checking the source: the slash branch FALLS THROUGH to
   `_handle_message_with_agent`, it never returns directly except on the
   "unknown command" case. The unknown-command branch returns BEFORE the
   activation write at L3864, so it can't poison the set. Still safe to
   add a `finally` discard for defense-in-depth.

## Step 4 — Tests

Add to `tests/gateway/test_session_boundary_skill_activation.py`:

1. **Behavioral test (the codex-round-7-required test):** `/skill foo`
   is the first message after idle auto-reset → assert
   `slash_skill_activations` row exists with the new skill's project
   AFTER `_handle_message_with_agent` returns. Mock the agent path so the
   test doesn't depend on a full LLM round-trip.
2. **Same-turn protection unit:** call the helper directly with a
   pre-seeded `_slash_activation_this_turn` set and assert the row is
   preserved.
3. **Negative case:** auto-reset with NO slash activation in the set →
   clear still runs (existing OQ-30 / round-7 behavior preserved).
4. **Idle/daily/suspended variants:** parametrize the auto_reset_reason.

## Step 5 — Hard-stop assessment

- Writer paths discovered: **1** (slash). Below the 4-path threshold.
- Round-7 fix architecture: NOT fundamentally incompatible — slash skills
  CAN be persisted before the auto-reset clear without breaking, provided
  the clear is guarded against same-turn writers. No bigger redesign needed.
- Tests behaving consistently: not yet run; will validate after impl.

Proceed to Step 3.
