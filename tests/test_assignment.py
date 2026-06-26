"""Tests for the from-scratch Hungarian algorithm."""

import sys

import pytest

sys.path.insert(0, "src")

from aegis.assignment import linear_sum_assignment  # noqa: E402


def _total(cost, pairs):
    return sum(cost[i][j] for i, j in pairs)


def test_beats_greedy_on_the_classic_2x2():
    # Greedy grabs the global min (T1->D1=1) and is forced into T2->D2=9 (total 10);
    # the optimal swap is T1->D2 + T2->D1 = 4.
    cost = [[1, 2], [2, 9]]
    pairs = linear_sum_assignment(cost)
    assert sorted(pairs) == [(0, 1), (1, 0)]
    assert _total(cost, pairs) == 4


def test_trivial_diagonal():
    cost = [[1, 9, 9], [9, 1, 9], [9, 9, 1]]
    assert linear_sum_assignment(cost) == [(0, 0), (1, 1), (2, 2)]


def test_one_pair_per_row_and_column():
    cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    pairs = linear_sum_assignment(cost)
    rows = [r for r, _ in pairs]
    cols = [c for _, c in pairs]
    assert sorted(rows) == [0, 1, 2] and sorted(cols) == [0, 1, 2]


def test_rectangular_more_columns():
    cost = [[4, 1, 3], [2, 0, 5]]       # 2 rows, 3 cols
    pairs = linear_sum_assignment(cost)
    assert len(pairs) == 2              # every row assigned
    assert len({c for _, c in pairs}) == 2


def test_rectangular_more_rows():
    cost = [[4, 1], [2, 0], [3, 5]]     # 3 rows, 2 cols
    pairs = linear_sum_assignment(cost)
    assert len(pairs) == 2              # every column assigned
    assert len({r for r, _ in pairs}) == 2


def test_empty():
    assert linear_sum_assignment([]) == []
    assert linear_sum_assignment([[]]) == []


def test_matches_scipy_total_cost():
    np = pytest.importorskip("numpy")
    sp = pytest.importorskip("scipy.optimize")
    rng = np.random.RandomState(0)
    for _ in range(20):
        n, m = rng.randint(1, 7), rng.randint(1, 7)
        cost = rng.rand(n, m)
        ours = linear_sum_assignment(cost.tolist())
        r, c = sp.linear_sum_assignment(cost)
        ref = cost[r, c].sum()
        assert abs(_total(cost.tolist(), ours) - ref) < 1e-9
