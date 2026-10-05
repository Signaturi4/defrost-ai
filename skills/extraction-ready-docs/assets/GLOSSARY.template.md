---
type: reference
entity: Glossary
status: current
updated: YYYY-MM-DD
---
# Glossary

## Terms

One row per concept. Use the canonical name everywhere in the docs; aliases are what people may search for.

| canonical name | aliases | definition (one sentence) | code |
|---|---|---|---|
| Example Service | example svc, ES | Handles example requests from the API gateway. | `ExampleService` |

## Relations for Facts blocks

Use only these relations in `Subject → relation → Object` lines. Add a row before using a new one.

| relation | meaning | example |
|---|---|---|
| calls | invokes at runtime | `ApiHandler` → calls → `PaymentService` |
| uses | depends on as a library or component | `Worker` → uses → Redis |
| reads / writes to | data access | `Exporter` → writes to → `exports` table |
| owns | responsible team or module | payments-team → owns → `PaymentService` |
| configures | a setting changes behaviour | `MAX_RETRIES` → configures → webhook retries |
| defaults to | default value | `MAX_RETRIES` → defaults to → 5 |
| requires | precondition | `DeployJob` → requires → `AWS_PROFILE` |
| replaces / deprecated by | lifecycle | `v1 API` → deprecated by → `v2 API` |
| triggers / stops | causes / ends an event | HTTP 410 → stops → retries |
| returns / raises | outputs and errors | `charge()` → raises → `CardDeclined` |
| part of | containment | `RetryPolicy` → part of → billing module |
