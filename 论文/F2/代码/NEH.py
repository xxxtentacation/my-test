from __future__ import annotations

import random
from typing import List, Sequence


def wcmax_of_order(order: Sequence[int], a: Sequence[float], b: Sequence[float],
                   w: Sequence[float]) -> float:

    n = len(order)
    wc = 0.0
    for j in range(n):
        best = 0.0
        for i in range(j + 1):
            sa = sum(a[order[h]] for h in range(i + 1))
            sb = sum(b[order[h]] for h in range(i, j + 1))
            best = max(best, sa + sb)
        wc = max(wc, w[order[j]] * best)
    return wc


def _wcmax_fast(order: Sequence[int], a: Sequence[float], b: Sequence[float],
                w: Sequence[float]) -> float:
    """
    Evaluate WCmax of `order` in O(len(order)) time by a single forward
    two-machine pass:
        c1 = cumulative M1 time, c2 = M2 completion time of the current job;
        c2 = max(c1, c2) + b_j,   wcmax = max(wcmax, w_j * c2).
    """
    c1 = 0.0
    c2 = 0.0
    wc = 0.0
    for j in order:
        c1 += a[j]
        c2 = max(c1, c2) + b[j]
        wc = max(wc, w[j] * c2)
    return wc


def _insert_best(sigma: List[int], job: int,
                 a: Sequence[float], b: Sequence[float],
                 w: Sequence[float]) -> None:

    k = len(sigma) + 1
    best_pos = 0
    best_val = float("inf")
    for pos in range(k):
        cand = sigma[:pos] + [job] + sigma[pos:]
        val = _wcmax_fast(cand, a, b, w)  # O(k) per candidate
        if val < best_val:
            best_val = val
            best_pos = pos
    sigma.insert(best_pos, job)


def neh(a: Sequence[float], b: Sequence[float], w: Sequence[float]) -> List[int]:
    n = len(a)
    if n != len(b) or n != len(w):
        raise ValueError("a, b, w must have the same length")

    # Step 1: sort jobs by non-increasing a_j + b_j, ties by non-increasing w_j.
    order = sorted(
        range(n),
        key=lambda j: (-(a[j] + b[j]), -w[j]),
    )

    # Step 2: initialize with the first job.
    sigma: List[int] = [order[0]]

    # Step 3: insert remaining jobs one by one at their best position.
    for k in range(1, n):
        _insert_best(sigma, order[k], a, b, w)

    return sigma


def run_neh(a, b, w) -> dict:
    order = neh(a, b, w)
    return {"obj": wcmax_of_order(order, a, b, w), "order": order}


def main() -> None:  # pragma: no cover
    """Random instance generated from a fixed seed, as in MILP.py."""
    # random-instance parameters
    seed = 42       # random seed
    n =  5          # number of jobs
    a_lo, a_hi = 1, 10   # M1 processing-time range
    b_lo, b_hi = 1, 10   # M2 processing-time range
    w_lo, w_hi = 1, 3     # weight range

    random.seed(seed)
    a = [random.randint(a_lo, a_hi) for _ in range(n)]
    b = [random.randint(b_lo, b_hi) for _ in range(n)]
    w = [random.randint(w_lo, w_hi) for _ in range(n)]

    print("seed   :", seed, "| n =", n)
    print("a      :", a)
    print("b      :", b)
    print("w      :", w)

    res = run_neh(a, b, w)
    print("NEH result:")
    print("  WCmax :", res["obj"])
    print("  order :", [j + 1 for j in res["order"]], "(1-based)")

    # Optional: cross-check optimal value with the DP solver, only for sizes
    # where the pseudo-polynomial DP is tractable.
    if n <= 30:
        try:
            from DP import solve_dp
            dp_res = solve_dp(a, b, w)
            print("  DP opt:", dp_res["obj"])
            print("  gap   : %.4f%%" % ((res["obj"] / dp_res["obj"] - 1) * 100))
        except Exception as exc:  # noqa: BLE001
            print("  DP comparison skipped:", exc)


if __name__ == "__main__":
    main()
