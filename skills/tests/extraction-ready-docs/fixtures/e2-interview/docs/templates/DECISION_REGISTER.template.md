---
type: reference
entity: decision register
status: current
updated: YYYY-MM-DD
lifecycle: living
---
# Decision register

The decision register holds every product, business and team decision, one row each. Rows are never deleted:
a stopped decision becomes `killed` with its reason. Open questions live only here.

| ID | Decision | Status | Why | Trigger to reopen | Owner | Review |
|---|---|---|---|---|---|---|
| D-01 | <what was decided> | proposed | <reason, with a source> | <what would reopen it> | <name> | YYYY-MM-DD |
| Q-01 | <open question> | open | — | — | <name> | YYYY-MM-DD |

Status values: `proposed` (waiting for a decision), `decided` (in force, with its date), `open` (not chosen),
`killed` (stopped; the row stays with its reason).

Facts:
- Decision register → holds → every decision and open question
- Each row → has → status, reason, trigger to reopen, owner, review date
