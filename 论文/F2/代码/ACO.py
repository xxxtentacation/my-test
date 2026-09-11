# -*- coding: utf-8 -*-
"""
Ant Colony Optimization (ACO) for F2 || WCmax (two-machine flow shop, minimizing
the maximum weighted completion time), implementing Algorithm (alg:aco) of the
paper (paper_framework.tex, Section 5.2).

A colony of m artificial ants constructs permutation schedules one position at a
time, guided by a pheromone matrix tau_ij (edge information: a large tau_ij
encourages job J_j to be placed right after job J_i) and by the heuristic
information eta_j = w_j / (a_j + b_j + 1).  After all ants have built their
schedules, the pheromone trails are reinforced along the edges of the best
schedule found so far:

    tau_ij <- (1 - rho) * tau_ij + rho * Delta_tau_ij,
    Delta_tau_ij = Q / Z*  if (i, j) lies on sigma*, and 0 otherwise.

Note.  The paper writes the construction probability with a job index
(tau_j) while the update rule uses an edge index (tau_ij).  The two are
reconciled here by taking tau_j to mean tau_{i,j}, where i is the job placed at
the previous position; this matches both the update rule and the motivation
("a larger value encourages job J_j to be scheduled after job J_i").
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence


def wcmax_of_order(order: Sequence[int], a: Sequence[float], b: Sequence[float],
                   w: Sequence[float]) -> float:
    """
    Compute the WCmax of a given permutation via the standard flow-shop recurrence
    (paper eq:completion):
        C_{pi(j)} = max_{1<=i<=j} { sum_{h=1..i} a_{pi(h)} + sum_{h=i..j} b_{pi(h)} },
        WCmax = max_j w_{pi(j)} C_{pi(j)}.
    """
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


def aco(a: Sequence[float], b: Sequence[float], w: Sequence[float],
        m: Optional[int] = None, alpha: float = 1.0, beta: float = 5.0,
        rho: float = 0.1, Q: float = 1.0, max_iter: int = 200,
        seed: Optional[int] = None) -> List[int]:
    """
    ACO for F2 || WCmax  (Algorithm alg:aco).

    Parameters
    ----------
    a, b, w  : length n
        Processing time on M1, processing time on M2, and weight of each job.
    m        : number of ants per iteration (default n).
    alpha    : pheromone exponent.
    beta     : heuristic-information exponent.  The paper does not fix a value;
               a comparatively large beta helps here, because the heuristic
               eta_j = w_j / (a_j + b_j + 1) encodes the "heavy jobs first" bias
               that the bottleneck objective rewards.
    rho      : evaporation rate in (0, 1).
    Q        : pheromone deposit constant.
    max_iter : number of iterations (the "stopping criterion" of Step 4).
    seed     : optional random seed.

    Returns
    -------
    list of job indices (0-based) of the best schedule sigma* found.
    """
    n = len(a)
    if n != len(b) or n != len(w):
        raise ValueError("a, b, w must have the same length")
    if m is None:
        m = n

    rng = random.Random(seed)

    # Step 1: initialize pheromone and the best solution.
    tau = [[1.0 / n] * n for _ in range(n)]
    eta = [w[j] / (a[j] + b[j] + 1.0) for j in range(n)]

    best_order: Optional[List[int]] = None
    best_val = float("inf")

    for _ in range(max_iter):
        # Step 2: each ant constructs a permutation.
        for _ant in range(m):
            remaining = list(range(n))
            sigma: List[int] = []
            prev = -1                      # no predecessor for the first position

            for _pos in range(n):
                # Selection weights tau_{prev,j}^alpha * eta_j^beta.
                if prev < 0:
                    weights = [eta[j] ** beta for j in remaining]
                else:
                    tau_row = tau[prev]
                    weights = [tau_row[j] ** alpha * eta[j] ** beta
                               for j in remaining]
                total = sum(weights)
                if total <= 0.0:           # degenerate: fall back to uniform
                    j = rng.choice(remaining)
                else:
                    j = rng.choices(remaining, weights=weights, k=1)[0]

                remaining.remove(j)
                sigma.append(j)
                prev = j

            z_k = _wcmax_fast(sigma, a, b, w)
            if z_k < best_val:             # Step 2, best-so-far update
                best_val = z_k
                best_order = sigma[:]

        # Step 3: evaporate, then reinforce the edges of sigma*.
        for i in range(n):
            row = tau[i]
            for j in range(n):
                row[j] *= (1.0 - rho)

        deposit = rho * Q / best_val if best_val > 0 else 0.0
        for t in range(n - 1):
            tau[best_order[t]][best_order[t + 1]] += deposit

    return best_order


def run_aco(a, b, w, **kwargs) -> dict:
    """Convenience wrapper returning a result dict."""
    order = aco(a, b, w, **kwargs)
    return {"obj": wcmax_of_order(order, a, b, w), "order": order}


def main() -> None:  # pragma: no cover
    """Random instance generated from a fixed seed, as in MILP.py."""
    # random-instance parameters
    seed = 42       # random seed
    n = 150          # number of jobs
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

    res = run_aco(a, b, w, m=50, max_iter=200, seed=seed)
    print("ACO result:")
    print("  WCmax :", res["obj"])
    print("  order :", [j + 1 for j in res["order"]], "(1-based)")
    # cross-check with the closed-form recurrence
    print("  verify:", wcmax_of_order(res["order"], a, b, w))

    # Optional: compare with NEH when available.
    try:
        from NEH import neh
        neh_order = neh(a, b, w)
        print("  NEH   :", wcmax_of_order(neh_order, a, b, w))
    except Exception as exc:  # noqa: BLE001
        print("  NEH comparison skipped:", exc)


if __name__ == "__main__":
    main()
