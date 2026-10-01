"""BKA -- Bidirectional Knowledge Attention, adapted to kev_graph's Stage 1 extractor.

Source: docs/jev_for_graph/research/kg-bilm-main (KG-BiLM).

What the reference release actually contains, verified by reading it rather than its README:
`kg-bilm/models/attn_mask_utils.py` swaps `AttentionMaskConverter(is_causal=True)` for
`is_causal=False`. That is a full causal-mask lift with no hop-distance logic -- the
`hop_threshold` key appears only in the README and nowhere in the package. So
`bidirectional_mask` below reproduces what the release ships, and `hop_bounded_mask` is the
paper's actual mechanism, written here because the release does not contain it.

Why the mask has to be 4D. Measured on transformers 5.17 / Qwen2.5-0.5B: passing a 2D all-ones
`attention_mask` does NOT remove causality -- the hidden state at position 0 stays invariant to
every later token (max|delta| = 0.000000). A 4D additive mask does (max|delta| = 38.28), as does
`is_causal=False`. The naive 2D call is a silent no-op, so every function here returns 4D.
"""

from collections import deque

import torch

NEG_INF = float("-inf")


def bidirectional_mask(seq_len, *, dtype=torch.float32, device=None, batch_size=1):
    """The reference release's BKA: attend everywhere, in both directions.

    Returns a 4D additive mask [B, 1, L, L] of zeros (0 = attend, no blocking). This is the
    cheapest test of "does removing causality help Stage 1 at all?" -- it needs no new weights and
    no graph, so run it before hop_bounded_mask and only keep the complexity if it pays.
    """
    return torch.zeros((batch_size, 1, seq_len, seq_len), dtype=dtype, device=device)


def entity_hop_distances(n_entities, edges):
    """All-pairs hop distance over the mention graph. Returns a list of lists; inf = unreachable.

    edges: iterable of (a, b) index pairs into the entity list. Treated as undirected -- a relation
    between two mentions is a symmetric adjacency for attention purposes even though the relation
    label itself is directed (direction is Stage 2's job; see docs/jev_for_graph/v2/concept+lean.md
    section 3.2).
    """
    adj = [[] for _ in range(n_entities)]
    for a, b in edges:
        if a != b and 0 <= a < n_entities and 0 <= b < n_entities:
            adj[a].append(b)
            adj[b].append(a)

    inf = float("inf")
    dist = [[inf] * n_entities for _ in range(n_entities)]
    for src in range(n_entities):
        dist[src][src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[src][v] == inf:
                    dist[src][v] = dist[src][u] + 1
                    queue.append(v)
    return dist


def hop_bounded_mask(seq_len, spans, edges, k, *, local_ranges=None,
                     dtype=torch.float32, device=None):
    """The paper's BKA: a token may attend across ranges only within `k` hops in the mention graph.

    spans: list of (start, end) inclusive token ranges, one per entity mention.
    edges: list of (a, b) index pairs into `spans`.
    k: hop threshold. k >= n_entities degenerates to fully bidirectional.
    local_ranges: list of (start, end) inclusive ranges where attention is unrestricted -- pass
        sentences. Within a sentence, prose tokens attend to each other as normal text.

    Modelling choice worth stating plainly: KG-BiLM's hop distance is defined over a knowledge
    graph that *is* the input, so every token belongs to an entity. A document is mostly prose
    that belongs to no entity. Gating those tokens by graph distance alone would leave most of the
    document unable to attend to anything, so this adapts the mechanism: attention is unrestricted
    inside a local range, and gated by mention-graph hops only across ranges. If `local_ranges` is
    None the whole sequence is one local range, which makes this identical to
    `bidirectional_mask` -- so a None here is a meaningful choice, not a default.

    Returns a 4D additive mask [1, 1, L, L]: 0 where attention is allowed, -inf where blocked.
    """
    if local_ranges is None:
        return bidirectional_mask(seq_len, dtype=dtype, device=device)

    token_entity = torch.full((seq_len,), -1, dtype=torch.long)
    for e, (start, end) in enumerate(spans):
        lo, hi = max(0, start), min(seq_len - 1, end)
        if lo <= hi:
            token_entity[lo:hi + 1] = e

    # token x token hop distance, gathered from the entity x entity table
    dist = entity_hop_distances(len(spans), edges)
    dist_t = torch.tensor(dist, dtype=torch.float32) if spans else torch.zeros((0, 0))
    idx = token_entity.clamp(min=0)
    if spans:
        token_dist = dist_t[idx.unsqueeze(0), idx.unsqueeze(1)]      # [L, L]
    else:
        token_dist = torch.full((seq_len, seq_len), float("inf"))

    both_entities = (token_entity >= 0).unsqueeze(0) & (token_entity >= 0).unsqueeze(1)
    within_k = token_dist <= k

    allowed = torch.zeros((seq_len, seq_len), dtype=torch.bool)
    for start, end in local_ranges:
        lo, hi = max(0, start), min(seq_len - 1, end)
        if lo <= hi:
            allowed[lo:hi + 1, lo:hi + 1] = True
    allowed |= (both_entities & within_k)
    allowed |= torch.eye(seq_len, dtype=torch.bool)                 # a token always attends to itself

    mask = torch.full((1, 1, seq_len, seq_len), NEG_INF, dtype=dtype, device=device)
    mask[0, 0] = torch.where(allowed, torch.zeros((), dtype=dtype), torch.tensor(NEG_INF, dtype=dtype))
    return mask


def mask_from_record(record, k, *, local_ranges=None, dtype=torch.float32, device=None):
    """Convenience: build a hop-bounded mask from a kev_graph record's own gold annotations.

    record is expected to expose `entities` as (start, end) token ranges and `relations` as
    (head_idx, tail_idx) pairs -- the shape used throughout kev_graph/etg and any_domain/.
    """
    spans = [(e[0], e[1]) for e in record["entities"]]
    edges = [(r[0], r[1]) for r in record["relations"]]
    seq_len = record["seq_len"]
    return hop_bounded_mask(seq_len, spans, edges, k, local_ranges=local_ranges,
                            dtype=dtype, device=device)
