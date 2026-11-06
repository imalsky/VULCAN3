---
name: doc-upkeep
description: Use when work in a VULCAN_Project repo (VULCAN-JAX, vulcan-forward, vulcan-retrieval, vulcan-jwst-tool) produces something to record or leaves something open - a bug, stale fixture, unrun check or pending decision found mid-task; a todo item to add, change or close; a measurement, design decision or rejected approach to write down; a behavior change that needs its doc pass; or a request to tidy CLAUDE.md, notes.md or todo.md.
---

# Doc upkeep (CLAUDE.md / notes.md / todo.md)

Each code repo has four root docs. README.md is the maintainer's; never
touch it unasked. The other three have one job each, are tracked, and the
repos are public: write for public reading (no personal names except cited
authors, no emails, mail/thread IDs, account names, keys or billing, no chat
quotes, no judgments of people or sessions). Write each fact in exactly one
place. Point at a section by its heading name, never by a number.

## Where a fact goes

| The fact is... | Goes to | Form |
|---|---|---|
| Open task, owed run, question for the maintainer | `todo.md` | headline + detail section |
| Measured number behind a rule | `notes.md`, the topic section of that rule | tagged [M]/[E]/[I], replace in place |
| Convention, design decision, why X not Y | `notes.md` "Design records" (or its topic section) | short record, no history |
| A retired design or model structure | `notes.md` "Superseded designs" | what, why dropped (measured), commit or tag |
| Approach tried and rejected | `notes.md` "Do not retry" | one line with the measured reason, never deleted while the lesson can recur |
| Idea nobody approved | `notes.md` "Ideas not adopted" | one line, expected gain if known |
| Small known issue, deliberately not fixed | `notes.md` "Known and deliberately untracked" | one line |
| What is true right now (versions, suite counts, production setup) | `notes.md` "Current state" | replace the old line, do not append |
| Rule an agent must follow | `CLAUDE.md` | operative sentence + the notes section name, no numbers |

notes.md points at todo.md; todo.md names the notes section. Neither repeats the other.

## todo.md

Format: one line saying the evidence is in notes.md, then
`N. [tag] plain sentence` in priority order, then one `## N. short name`
section per item with: the notes section holding the evidence, files and
gates, the command to run, what "done" means. Tags: `[bug?]` `[speed]`
`[stale]` `[unverified]` `[decision]` `[blocker]` `[science]` `[upstream]`
`[robustness]` `[feature]` `[low]`. No other tags.

- **Open** an item only for something that clears the root CLAUDE.md bar
  (changes a shipped/published number, a user-visible result or a safety
  path; needs the maintainer's decision; an owed run behind a claim; a
  stale pin that breaks a gate; a blocker; a science gate). Open it in the
  same pass, not "later"; not a chat remark, code comment or "Current
  state" line.
- **Do not open** an item for cosmetic, tiny or nice-to-have findings,
  wishlists, coverage gaps with no consequence, or comment drift: fix them
  in the same pass when cheap, otherwise one line under notes "Known and
  deliberately untracked". Ten-plus items means pare, not add.
- **Update** by replacing the detail section in place; re-tag or re-rank
  the headline if that changed; split into separate items rather than
  nesting.
- **Close** when it lands: delete headline and section, renumber, and file
  the result by the table above. A `[decision]` item closes only after the
  maintainer decides; the decision goes to notes "Design records" first.
- **Cross-repo**: the item lives where the code changes; the consumer's
  todo.md gets a one-line item naming it.
- A repo with nothing open has an empty list. Do not pad.

## Repo-specific triggers

- VULCAN-JAX: a suspected parity defect is `[bug?]` until CONFIRMED with
  file:line on both sides; then notes "Parity ledger" / "Known bugs, all
  repos" get the entry and the item closes.
- vulcan-forward: a physics or RT change that leaves a fixture, figure or
  consumer pin behind opens `[stale]` with the regeneration command.
- vulcan-retrieval: a GPU job closes its item only when its numbers are in
  notes "Runs"; an owed job stays an item with its exact invocation.
- vulcan-jwst-tool: a `_VERSION` or pin bump with unregenerated data opens
  `[stale]` listing every file and its script.

## End-of-task pass (before reporting done)

1. todo.md: every item touched is opened, updated or closed as above.
2. notes.md: new numbers under their rule, decisions in "Design records",
   rejects in "Do not retry", "Current state" current; nothing dated
   except a run's or measurement's code state, nothing appended as a diary.
3. CLAUDE.md: edited only if a rule changed; the pointer names a section.
4. No new top-level `.md` (any other goes in `docs/`); README untouched; nothing a public reader should
   not see.

## Never

Dated diary lines, "in progress" markers, checkboxes, done lists, untagged
items, an item without a detail section, the same fact in two files,
section numbers or register numbers as pointers. Deleting or trimming
VULCAN-JAX notes "Known bugs, all repos" in a cleanup pass (it changes only
when a bug is fixed or re-measured).
