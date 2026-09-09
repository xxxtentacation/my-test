# -*- coding: utf-8 -*-
"""
Exact DP solver for F2 || WCmax (two-machine flow shop, minimizing the maximum
weighted completion time), implementing Algorithm 1 and Algorithm 2 of the paper
(paper_framework.tex, Section 4).

Dependencies: `johnson_sequence(a, b)` is provided below; for the `milp` module
it is used only as a cross-check in `main()`.
"""

from __future__ import annotations

import random
from typing import List, Tuple


def johnson_sequence(a: List[float], b: List[float]) -> List[int]:
    """
    Johnson's rule for F2 || Cmax.

    Returns the list of job indices (0-based) ordered by Johnson's rule:
    jobs with a_j <= b_j are sorted by non-decreasing a_j;
    jobs with a_j >  b_j are sorted by non-increasing b_j.
    """
    n = len(a)
    if n != len(b):
        raise ValueError("a and b must have the same length")

    left = sorted(((a[j], j) for j in range(n) if a[j] <= b[j]), key=lambda x: x[0])
    right = sorted(((b[j], j) for j in range(n) if a[j] > b[j]), key=lambda x: x[0], reverse=True)

    return [j for _, j in left] + [j for _, j in right]


def block_concatenation_and_evaluation(
    A: List[float],
    B: List[float],
    C: List[float],
    L: List[float],
) -> float:
    """
    Algorithm 1: Block concatenation and evaluation.

    Parameters
    ----------
    A, B, C, L : length K
        Block aggregates as defined in the paper:
        A_i / B_i : total M1 / M2 processing time of block i
        C_i       : completion time of block i when run in isolation
        L_i       : weight of the last job of block i (0 if empty)

    Returns
    -------
    WCmax value of the schedule obtained by concatenating the K blocks.
    """
    K = len(A)
    if not (len(B) == len(C) == len(L) == K):
        raise ValueError("A, B, C, L must have the same length")

    T_A_prev = 0.0
    T_B_prev = 0.0
    wcmax = 0.0

    for i in range(K):
        T_A_i = T_A_prev + A[i]
        T_B_i = max(T_A_prev + C[i], T_B_prev + B[i])
        if L[i] > 0:
            wcmax = max(wcmax, L[i] * T_B_i)
        T_A_prev = T_A_i
        T_B_prev = T_B_i

    return wcmax


class _State:
    """Internal representation of a DP state for Algorithm 2."""

    __slots__ = ("A", "B", "C", "L", "prev", "block")

    def __init__(
        self,
        A: Tuple[float, ...],
        B: Tuple[float, ...],
        C: Tuple[float, ...],
        L: Tuple[float, ...],
        prev: "_State | None" = None,
        block: int = -1,
    ):
        self.A = A  # length K
        self.B = B  # length K
        self.C = C  # length K
        self.L = L  # length K
        self.prev = prev
        self.block = block  # block into which the current job was placed

    def key(self) -> Tuple[float, ...]:
        """Canonical key used for duplicate elimination (Merge step)."""
        return self.A + self.B + self.C + self.L

    def dominates(self, other: "_State") -> bool:
        """
        Lemma 4.2 (merge-dominance).
        self dominates other iff A/B/L are equal and C_i <= other.C_i for all i.
        """
        if self.A != other.A or self.B != other.B or self.L != other.L:
            return False
        return all(c1 <= c2 for c1, c2 in zip(self.C, other.C))

    def copy_with_job(self, idx: int, aj: float, bj: float, wj: float) -> "_State":
        """
        Place job J_idx into block i (0-based), updating only that block.
        Transition: eq:transition in the paper.
        """
        A = list(self.A)
        B = list(self.B)
        C = list(self.C)
        L = list(self.L)

        A[idx] += aj
        B[idx] += bj
        L[idx] = wj
        C[idx] = max(C[idx] + bj, A[idx] + bj)

        return _State(
            tuple(A), tuple(B), tuple(C), tuple(L),
            prev=self,
            block=idx,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"State(A={self.A}, B={self.B}, C={self.C}, L={self.L})"


def _merge(states: List[_State]) -> List[_State]:
    """
    Merge step of Algorithm 2 (Lemma 4.2).
    Group states by (A, B, L); within each group keep only the nondominated
    states (i.e. the Pareto-minimal C-vectors).
    """
    groups: dict[Tuple[float, ...], List[_State]] = {}
    for s in states:
        key = s.A + s.B + s.L
        groups.setdefault(key, []).append(s)

    merged: List[_State] = []
    for group in groups.values():
        # Keep a state s if no other state in the group has C <= s.C componentwise.
        kept: List[_State] = []
        for s in group:
            dominated = False
            new_kept = []
            for t in kept:
                if t.dominates(s):
                    dominated = True
                    new_kept.append(t)
                elif not s.dominates(t):
                    new_kept.append(t)
            if not dominated:
                kept = new_kept + [s]
        merged.extend(kept)

    return merged


def solve_dp(a: List[float], b: List[float], w: List[float]) -> dict:
    """
    Algorithm 2: DP for F2 || WCmax.

    Parameters
    ----------
    a, b, w : length n
        Processing times on M1 / M2 and job weights.

    Returns
    -------
    dict with keys
        obj       : optimal WCmax value
        blocks    : list of lists; blocks[i] contains 0-based job indices in block i,
                    ordered by Johnson's rule within the block
        order     : final concatenated permutation (0-based job indices)
        wcmax     : same as obj
    """
    n = len(a)
    if n != len(b) or n != len(w):
        raise ValueError("a, b, w must have the same length")

    # --- preprocessing: Johnson order and distinct weights / blocks ---
    johnson = johnson_sequence(a, b)
    distinct_weights = sorted({w[j] for j in johnson}, reverse=True)
    K = len(distinct_weights)
    W = list(distinct_weights)  # W[0] > W[1] > ... > W[K-1]

    # Map weight -> block index (0-based)
    weight_to_block = {wt: idx for idx, wt in enumerate(W)}

    # Jobs are processed in Johnson order; eligible blocks for a job are the prefix
    # of blocks whose weight >= w_j.
    #
    # Initialize: single zero state.
    zero_state = _State(
        tuple([0.0] * K),
        tuple([0.0] * K),
        tuple([0.0] * K),
        tuple([0.0] * K),
    )
    H = [zero_state]

    for t in range(n):
        j = johnson[t]
        wj = w[j]
        max_block = weight_to_block[wj]  # largest i with W[i] >= wj

        candidates: List[_State] = []
        for state in H:
            for i in range(max_block + 1):
                candidates.append(state.copy_with_job(i, a[j], b[j], wj))

        H = _merge(candidates)

    # --- Step 4: feasibility check (L_i must be either 0 or W_i) ---
    feasible = [s for s in H if all(s.L[i] in (0.0, W[i]) for i in range(K))]
    if not feasible:
        return {"obj": float("inf"), "blocks": [], "order": [], "wcmax": float("inf")}

    # --- Step 5: calculate best objective via Algorithm 1 ---
    best_value = float("inf")
    best_state: _State | None = None
    for s in feasible:
        value = block_concatenation_and_evaluation(
            list(s.A), list(s.B), list(s.C), list(s.L)
        )
        if value < best_value:
            best_value = value
            best_state = s

    if best_state is None:  # should not happen
        return {"obj": float("inf"), "blocks": [], "order": [], "wcmax": float("inf")}

    # --- Step 6: backtrack to recover block assignment for every job ---
    job_to_block = [-1] * n
    cur = best_state
    # Walk backwards over the Johnson sequence
    for t in range(n - 1, -1, -1):
        j = johnson[t]
        job_to_block[j] = cur.block
        cur = cur.prev
        if cur is None and t > 0:
            raise RuntimeError("Backtracking failed: chain shorter than expected")

    # Build blocks and sequence jobs inside each block by Johnson's rule.
    blocks: List[List[int]] = [[] for _ in range(K)]
    for j in range(n):
        blocks[job_to_block[j]].append(j)

    for i in range(K):
        jobs_in_block = blocks[i]
        seq = johnson_sequence(
            [a[j] for j in jobs_in_block], [b[j] for j in jobs_in_block]
        )
        blocks[i] = [jobs_in_block[p] for p in seq]

    order: List[int] = []
    for i in range(K):
        order.extend(blocks[i])

    return {
        "obj": best_value,
        "wcmax": best_value,
        "blocks": blocks,
        "order": order,
    }


def wcmax_of_order(order: List[int], a: List[float], b: List[float], w: List[float]) -> float:
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


def main() -> None:  # pragma: no cover
    """Random instance generated from a fixed seed, as in MILP.py."""
    # random-instance parameters
    seed = 42       # random seed
    n = 6           # number of jobs (small: the DP is pseudo-polynomial)
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

    res = solve_dp(a, b, w)
    print("DP result:")
    print("  WCmax :", res["obj"])
    print("  blocks:", [[j + 1 for j in blk] for blk in res["blocks"]])
    print("  order :", [j + 1 for j in res["order"]])
    print("  verify:", wcmax_of_order(res["order"], a, b, w))

    # Optional: compare with MILP if gurobipy is installed
    try:
        from MILP import solve_milp, wcmax_of_order as milp_wcmax_of_order
        milp_res = solve_milp(a, b, w, output_flag=0)
        if milp_res["order"] is not None:
            print("\nMILP result:")
            print("  WCmax :", milp_res["obj"])
            print("  order :", [j + 1 for j in milp_res["order"]])
            print("  verify:", milp_wcmax_of_order(milp_res["order"], a, b, w))
    except Exception as exc:  # noqa: BLE001
        print("\nMILP comparison skipped:", exc)


if __name__ == "__main__":
    main()
