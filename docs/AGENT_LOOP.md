# How an agent should use the memory (ReAct loop)

An agent works in a loop: **think → act (call a tool) → observe → think again**. defrost-ai gives that loop four
kinds of tools. Documentation sections come from the memory: hybrid search, then the reranker. The code graph and
the files hold the truth about the code. Working notes carry state across `/clear`, and doc sync keeps the docs
current. The graph is for navigation; the answer always comes from a doc section, the code, or both.

Rendered copies for slides and the README: [architecture](img/agent_loop-1.svg), [loop](img/agent_loop-2.svg).

## 1. Where the system sits

```mermaid
flowchart LR
    subgraph Agent["Agent (Claude Code, Cursor, any MCP client)"]
        T[Thought] --> A[Action: tool call]
        A --> O[Observation]
        O --> T
    end

    subgraph MCP["MCP servers (thin stdio)"]
        M1["defrost:<br/>memory_search · memory_docs_for<br/>memory_docs_plan · memory_handoff · memory_brief<br/>memory_update · memory_domains"]
        M2["graphify:<br/>query_graph · get_node · get_neighbors<br/>shortest_path · god_nodes"]
        F["File tools:<br/>Read · Grep · Glob · Bash"]
    end

    subgraph Service["kev-memory service (resident, 127.0.0.1:8765)"]
        S1["BM25<br/>SQLite FTS5 · ~2 ms"]
        S2["Kev-Ret-B<br/>dense query vector · ~0.1 s"]
        P{"fast policy:<br/>top-1 agrees?"}
        S3["Kev-Rerank v2<br/>MLX fp16 · ~1.5 s<br/>score cache"]
        K["adaptive k (1-5)<br/>+ trust header<br/>+ verify in / ! flags"]
        S1 --> P
        S2 --> P
        P -- "yes: hybrid" --> K
        P -- "no" --> S3 --> K
    end

    subgraph Store["Per-project memory (~/.kev-memory/&lt;domain&gt;)"]
        D1[("knowledge.sqlite<br/>sections · links · FTS")]
        D2[("section_vectors.npz")]
        D3[("code_graph.json<br/>(graphify AST)")]
        D4[("&lt;domain&gt;-notes<br/>handoff notes")]
    end

    subgraph Repo["Your repository"]
        R1["docs/ · README · CLAUDE.md"]
        R2["code: backend (core) · frontend · config/CI"]
        R3["git hooks · Claude hooks"]
    end

    A -->|how / why / what happens| M1
    A -->|structure, callers, paths| M2
    A -->|exact code, verify| F
    M1 --> Service
    S1 -. reads .-> D1
    S2 -. reads .-> D2
    K -->|"doc sections + linked core code"| O
    M2 --> D3
    F --> R1 & R2
    R3 -->|"merge/commit to main:<br/>incremental refresh"| Store
    R1 & R2 -->|"build: sections, vectors,<br/>doc→code links"| Store
    M1 -.->|handoff / brief| D4
```

## 2. The loop, step by step

```mermaid
flowchart TD
    Q(["User task"]) --> B{"New session<br/>after /clear?"}
    B -- yes --> BR["memory_brief:<br/>goal, state, next steps"] --> T1
    B -- no --> T1["Thought: what do I need to know?"]

    T1 --> KIND{"Kind of question"}
    KIND -- "how / why / what happens<br/>(behaviour, process, decisions)" --> MS["memory_search(query, k='auto')"]
    KIND -- "where is X defined / called<br/>(structure)" --> G["graphify: query_graph /<br/>get_neighbors / shortest_path"]
    KIND -- "exact string, flag, error text" --> GR["Grep / memory_search mode=bm25"]

    MS --> OBS["Observation: 1-5 sections<br/>path:Lstart-end + linked code<br/>+ verify in: + ! flags"]
    OBS --> TRUST{"doc trust<br/>(project setting)"}
    TRUST -- "low (fast-changing code)" --> V["Read the verify-in files<br/>(core code first)"]
    TRUST -- "high (reliable docs)" --> FLAG{"! stale or<br/>conflict flag?"}
    FLAG -- yes --> V
    FLAG -- no --> ANS
    V --> CMP{"Docs and code agree?"}
    CMP -- yes --> ANS
    CMP -- no --> CONF["Code wins; note it<br/>under Doc/code conflicts"] --> ANS
    G --> GO["Observation: nodes + edges"] --> V
    GR --> V

    ANS["Thought: enough to answer or act?"] -- "no: refine query<br/>or follow a link" --> T1
    ANS -- yes --> ACT{"Task changes code?"}
    ACT -- no --> OUT(["Answer with citations<br/>path:Lstart-end"])
    ACT -- yes --> EDIT["Edit code"] --> DP["memory_docs_for(changed files)<br/>memory_docs_plan()"]
    DP --> DOC["Update those sections (DOC_RULES),<br/>document new env vars / commands / files"]
    DOC --> COMMIT["git commit on main"] --> HOOK["post-commit hook:<br/>refresh --if-changed (seconds)"]
    HOOK --> OUT
    OUT -.->|"long task / context full"| HO["memory_handoff:<br/>goal, state, decisions, next steps"] -.-> CLR["/clear"] -.-> B
```

## 3. Rules the loop follows

| step | rule | why |
|---|---|---|
| choose a tool | Use memory for questions about behaviour, process and decisions; the graph for structure; grep for exact strings | The memory finds the answering doc 94% of the time; graph queries find it 6% of the time ([E2E_GRAPHIFY.md](E2E_GRAPHIFY.md)) |
| `k="auto"` | One confident section is enough, and five means the retriever is unsure | Fewer tokens for the next thought; low confidence prompts a refined query |
| verify | Read the `verify in:` files: always when trust is `low`, and on a `!` flag when trust is `high` | Docs go stale; in the agent test, memory alone repeated a stale CI claim |
| conflicts | Code wins, and the answer lists each doc/code conflict | Silent stale answers are the costliest failure |
| core first | Linked code from the project core (backend, API, db) is shown first | Fewer jumps into UI or test code |
| after edits | `memory_docs_for` the changed files, then update those sections in the same change | Keeps the memory true for the next loop |
| refresh | Git hooks refresh incrementally on main; the agent never rebuilds by hand | The next search already sees the change |
| long tasks | `memory_handoff`, then `/clear`, then `memory_brief` | Carries the goal and state forward, not the full history |

## 4. What each part costs in one loop step (Apple M5, warm service)

| action | typical latency | context it adds |
|---|---|---|
| `memory_search` (hybrid, top-1 agrees) | ~0.1 s | 1–5 sections, about 300 words each |
| `memory_search` (reranked) | 1.2–1.8 s; repeated query 0.04 s | same |
| graphify query | < 0.1 s | node lists (can be large; use for navigation) |
| Read a `verify in:` file | < 0.1 s | the file or a range of it |
| `memory_docs_for` / `memory_docs_plan` | 0.2–0.4 s | list of sections to update |
| `memory_brief` | 0.04 s | ≤ 350 words |
