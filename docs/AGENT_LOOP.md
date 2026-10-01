# How an agent should use the memory (ReAct loop)

An agent works in a loop: **think → act (call a tool) → observe → think again**. defrost-ai gives that loop four
kinds of tools. Documentation sections come from the memory: hybrid search, then the reranker. The code graph and
the files hold the truth about the code. Working notes carry state across `/clear`, and doc sync keeps the docs
current. The graph is for navigation; the answer always comes from a doc section, the code, or both.

Rendered copies for slides and the README: architecture ([svg](img/agent_loop-1.svg), [png](img/agent_loop-1.png)), loop ([svg](img/agent_loop-2.svg), [png](img/agent_loop-2.png)).

Everything in both diagrams is on `main`: the conflict step (human in the loop), the git-backed context repo, its worktree workers and the PreCompact handoff (opt-in: `defrost setup --handoff-on-compact`).

## 1. Where the system sits

```mermaid
---
config:
  layout: elk
  elk:
    mergeEdges: true
    nodePlacementStrategy: BRANDES_KOEPF
---
flowchart LR
    subgraph Agent["Agent (Claude Code, Cursor, any MCP client): Thought → Action → Observation"]
        direction LR
        T[Thought] --> A[Action: tool call] --> O[Observation] --> T
    end
    U(("You<br/>human in the loop"))

    subgraph MCP["MCP servers (thin stdio)"]
        direction LR
        M1["defrost<br/>search · docs_for<br/>remember · refresh"]
        M2["graphify<br/>query_graph · get_neighbors<br/>shortest_path · god_nodes"]
        F["File tools<br/>Read · Grep · Glob · Bash"]
    end

    subgraph Service["defrost service (resident, 127.0.0.1:8765)"]
        direction LR
        S1["BM25<br/>FTS5 · ~2 ms"] --> P{"accurate mode:<br/>top-1 agrees?"}
        S2["Defrost-Ret-B<br/>query vector · ~0.1 s"] --> P
        P -- "yes, or fast mode" --> K
        P -- "no" --> S3["Defrost-Rerank v2<br/>MLX fp16 · ~1.5 s<br/>score cache"] --> K["adaptive k (1-5)<br/>+ trust header · verify in / ! flags<br/>+ resolved: decisions"]
    end

    subgraph Stores["Local stores"]
        direction LR
        subgraph Store["Project memory · ~/.defrost-ai/&lt;domain&gt;"]
            D1[("knowledge.sqlite<br/>sections · links · FTS")]
            D2[("section_vectors.npz")]
            D3[("code_graph.json<br/>graphify AST")]
        end
        subgraph Ctx["Context repo (git) · &lt;project&gt;/defrost-memory"]
            C0["MEMORY.md map + core files"]
            C1["notes/ handoff notes"]
            C2["decisions/ conflict decisions"]
            C3["pre-commit hook:<br/>depth · size · frontmatter · read_only"]
        end
    end

    subgraph Repo["Your repository"]
        direction LR
        R1["docs/ · README · CLAUDE.md"]
        R2["code: backend (core) · frontend · config/CI"]
        R3["git hooks · Claude hooks<br/>SessionStart · PreCompact · Stop"]
    end

    subgraph Workers["Background workers (git worktrees)"]
        direction LR
        W1["doc-sync worker<br/>branch defrost/docs/&lt;sha&gt; → review"]
        W2["context defrag<br/>indexes · split · archive → fast-forward"]
    end

    A -->|"how / why"| M1
    A -->|"structure"| M2
    A -->|"exact code"| F
    A <-->|"doc ≠ code: AskUserQuestion<br/>→ code right / doc right / no conflict / open"| U
    M1 --> Service
    K -->|"sections + linked core code"| O
    Service -. reads .-> Store
    Service -. "reads (&lt;domain&gt;-context)" .-> Ctx
    M2 --> D3
    M1 -->|"one commit per decision / note"| Ctx
    F --> Repo
    Repo -->|"build + incremental refresh<br/>on merge/commit to main"| Store
    R3 -->|"PreCompact: extractive handoff (no LLM)"| C1
    R3 -->|"commit outside Claude"| W1
    W1 -->|"review, then merge"| R1
    W2 --> Ctx
```

## 2. The loop, step by step

```mermaid
flowchart TD
    Q(["User task"]) --> B{"New session<br/>after /clear or compaction?"}
    B -- yes --> BR["SessionStart hook shows the brief:<br/>context repo map, core files,<br/>latest handoff note"] --> T1
    B -- no --> T1["Thought: what do I need to know?"]

    T1 --> KIND{"Kind of question"}
    KIND -- "how / why / what happens<br/>(behaviour, process, decisions)" --> MS["search(question)"]
    KIND -- "where is X defined / called<br/>(structure)" --> G["graphify: query_graph /<br/>get_neighbors / shortest_path"]
    KIND -- "exact string, flag, error text" --> GR["Grep"]

    MS --> OBS["Observation: 1-5 sections<br/>path:Lstart-end + linked code<br/>+ verify in: + ! flags + resolved:"]
    OBS --> TRUST{"doc trust<br/>(project setting)"}
    TRUST -- "low (fast-changing code)" --> V["Read the verify-in files<br/>(core code first)"]
    TRUST -- "high (reliable docs)" --> FLAG{"! stale or<br/>conflict flag?"}
    FLAG -- yes --> V
    FLAG -- no --> ANS
    V --> CMP{"Docs and code agree?"}
    CMP -- yes --> ANS
    CMP -- no --> RES{"Already decided?<br/>(resolved: line)"}
    RES -- yes --> FOL["Follow the recorded decision"] --> ANS
    RES -- no --> ASK[/"Ask the user (AskUserQuestion):<br/>both sides with citations"/]
    ASK --> REC["remember(kind=decision):<br/>one commit in decisions/"]
    REC --> DEC{"User's decision"}
    DEC -- "code is right" --> FIXDOC["Update the doc section"] --> ANS
    DEC -- "doc is right" --> BUG["Report the code as a bug<br/>(change code only if asked)"] --> ANS
    DEC -- "not a conflict" --> ANS
    DEC -- "not sure" --> OPEN["Open-question note<br/>in the doc section"] --> ANS
    G --> GO["Observation: nodes + edges"] --> V
    GR --> V

    ANS["Thought: enough to answer or act?"] -- "no: refine query<br/>or follow a link" --> T1
    ANS -- yes --> ACT{"Task changes code?"}
    ACT -- no --> OUT(["Answer with citations<br/>path:Lstart-end"])
    ACT -- yes --> EDIT["Edit code"] --> DP["docs_for(change=working)"]
    DP --> DOC["Update those sections (DOC_RULES),<br/>document new env vars / commands / files"]
    DOC --> COMMIT["git commit on main"] --> HOOK["post-commit hook:<br/>refresh --if-changed (seconds)"]
    HOOK --> OUT
    COMMIT -.->|"commit made outside Claude"| WK["doc-sync worker in a worktree:<br/>branch defrost/docs/&lt;sha&gt; for review"]
    OUT -.->|"long task: /handoff"| HO["remember(kind=note): goal, state,<br/>decisions, next steps → notes/"]
    OUT -.->|"context full: PreCompact hook"| XH["extractive handoff from the<br/>transcript (no LLM) → notes/"]
    HO -.-> CLR["/clear"] -.-> B
    XH -.-> B
```

## 3. Rules the loop follows

| step | rule | why |
|---|---|---|
| choose a tool | Use memory for questions about behaviour, process and decisions; the graph for structure; grep for exact strings | The memory finds the answering doc 94% of the time; graph queries find it 6% of the time ([E2E_GRAPHIFY.md](E2E_GRAPHIFY.md)) |
| `k="auto"` | One confident section is enough, and five means the retriever is unsure | Fewer tokens for the next thought; low confidence prompts a refined query |
| verify | Read the `verify in:` files: always when trust is `low`, and on a `!` flag when trust is `high` | Docs go stale; in the agent test, memory alone repeated a stale CI claim |
| conflicts | The user decides (human in the loop): the agent shows both sides, asks, records the decision with `remember(kind="decision")`, then updates the doc, reports a bug, or marks an open question | Neither side is always right: a doc can describe intended behaviour the code broke. Decisions are remembered, so each conflict is asked once |
| core first | Linked code from the project core (backend, API, db) is shown first | Fewer jumps into UI or test code |
| after edits | `docs_for` the change, then update those sections in the same change | Keeps the memory true for the next loop |
| refresh | Git hooks refresh incrementally on main; the agent never rebuilds by hand | The next search already sees the change |
| long tasks | `remember(kind="note")` (or `/handoff`), then `/clear`; the SessionStart hook shows the brief | Carries the goal and state forward, not the full history |

## 4. What each part costs in one loop step (Apple M5, warm service)

| action | typical latency | context it adds |
|---|---|---|
| `search`, `fast` mode or retrievers agree | ~0.1 s | 1–5 sections, about 300 words each |
| `search`, `accurate` mode, reranked | 1.2–1.8 s; repeated query 0.04 s | same |
| graphify query | < 0.1 s | node lists (can be large; use for navigation) |
| Read a `verify in:` file | < 0.1 s | the file or a range of it |
| `docs_for` | 0.2–0.4 s | list of sections to update |
| brief (SessionStart hook) | 0.04 s | ≤ 350 words |
