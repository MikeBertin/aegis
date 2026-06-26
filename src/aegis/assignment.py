"""The Hungarian algorithm from scratch — optimal assignment.

Given a cost matrix, find the one-to-one matching of rows to columns that
**minimises total cost** — the globally optimal answer, where greedy "take the
cheapest pair first" can be arbitrarily worse. In tracking this matches
detections to tracks with ``cost = 1 - IoU``.

This is the O(n³) potential (Kuhn–Munkres) formulation. Rectangular matrices are
handled by assigning every row of the smaller dimension. Verified against
``scipy.optimize.linear_sum_assignment`` in the tests.
"""

from __future__ import annotations

from typing import Sequence


def _hungarian_core(cost, n: int, m: int) -> list[tuple[int, int]]:
    """Core solver for ``n <= m`` (rows <= columns). ``cost`` is indexable
    ``cost[i][j]``. Returns ``(row, col)`` pairs, one per row."""
    INF = float("inf")
    u = [0.0] * (n + 1)       # row potentials
    v = [0.0] * (m + 1)       # column potentials
    p = [0] * (m + 1)         # p[j] = row matched to column j (1-indexed; 0 = free)
    way = [0] * (m + 1)       # augmenting-path back-pointers

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:                       # grow the shortest augmenting path
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):        # update potentials by the slack delta
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:                          # apply the augmenting path
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]


def linear_sum_assignment(cost: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Minimum-cost assignment. Returns ``(row, col)`` pairs sorted by row.

    Works for any shape; ``min(rows, cols)`` pairs are returned.
    """
    rows = list(cost)
    n = len(rows)
    if n == 0:
        return []
    m = len(rows[0])
    if m == 0:
        return []

    if n <= m:
        pairs = _hungarian_core(rows, n, m)
    else:
        # Transpose: solve columns-as-rows, then swap each pair back.
        transposed = [[rows[i][j] for i in range(n)] for j in range(m)]
        pairs = [(i, j) for (j, i) in _hungarian_core(transposed, m, n)]

    pairs.sort()
    return pairs
