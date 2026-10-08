# -*- coding: utf-8 -*-
"""
Exact DP solver for F2 || WCmax (two-machine flow shop, minimizing the maximum
weighted completion time), implementing Algorithm 1 and Algorithm 2 of the paper
(paper_framework.tex, Section 4).

Dependencies: `johnson_sequence(a, b)` is provided below; for the `milp` module
it is used only as a cross-check in `main()`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import random
import statistics
import time
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


def solve_dp(a: List[float], b: List[float], w: List[float],
             time_limit: float | None = None) -> dict:
    """
    Algorithm 2: DP for F2 || WCmax.

    Parameters
    ----------
    a, b, w : length n
        Processing times on M1 / M2 and job weights.
    time_limit : float, optional
        Wall-clock limit in seconds, checked once per job.  When it is exceeded
        the function raises TimeoutError instead of returning; the benchmark
        harness uses this to flag instances the DP cannot finish, so that the
        reported mean covers only the instances actually solved.

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

    deadline = None if time_limit is None else time.time() + time_limit

    for t in range(n):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(
                "solve_dp exceeded the %.1f s time limit at job %d/%d"
                % (time_limit, t, n)
            )
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


# ---------------------------------------------------------------------------
# Experiment harness (paper Section 6.1)
#
# Each configuration is replicated over INSTANCES_PER_CONFIG random instances.
# For every instance the best objective value found by the method and its CPU
# time are recorded, and the two are averaged over the instances to give the
# comparison result (paper Section 6.1.3).
# ---------------------------------------------------------------------------

INSTANCES_PER_CONFIG = 20      # paper Section 6.1: instances per configuration
PROC_LO, PROC_HI = 1, 10       # processing times ~ U{1, ..., 10}

#: name of this module, used to tag the per-instance records
LABEL = Path(__file__).stem.upper()

#: where the per-instance records go when the caller names no file
RECORD_DIR = Path(__file__).resolve().parent / "记录"

#: how a weight setting is written in the records, and in the tables of the
#: paper: by the number of distinct weights it produces.  K is that number for
#: every drawn regime -- the narrow one takes it as the parameter K, and the
#: independent and correlated ones from WIDE_K -- while the unrestricted regime
#: has no such number and is named instead.
def weight_label(weights, K):
    return "all weights" if weights == "free" else "K=%s" % K


#: what the record says about optimality, in the words of the live output of the
#: drivers: True when the method closed the instance, False when it ran into the
#: time limit, and None for a heuristic, which proves nothing about optimality
STATUS = {True: "optimal", False: "time limit", None: "-"}


def record_instance(path, n, weights, seed, idx, total, obj, seconds,
                    K=None, proc_hi=PROC_HI, proved=None, base=None):
    """
    Append one line for a finished instance, and flush it at once.

    The line is written as soon as that instance is done, not when the whole
    benchmark is over, so that a run which is interrupted or killed still holds
    every instance it has already solved.  `path` may be a str or a Path; its
    parent directory is created if it does not exist.  The fields are those of
    the live line of the drivers -- label, n, weight setting, processing-time
    range, instance, value, gap, status and running time -- preceded by the seed
    of the instance and the moment it finished.  The gap is the one the method
    reports on, `base` being the reference it computed for the instance; a
    method that hands none, as the exact ones do, shows a dash there.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("%s  %-5s n=%-4d %-6s %-7s | seed=%-7d | instance %2d/%-2d "
                 "| value %-12s | gap %9s | %-10s %9s\n"
                 % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), LABEL, n,
                    weight_label(weights, K), "[%d,%d]" % (PROC_LO, proc_hi),
                    seed, idx, total,
                    "none" if obj is None else "%.1f" % obj,
                    "-" if obj is None or base is None or base == 0
                    else "%+.2f%%" % (100.0 * (obj - base) / base),
                    STATUS.get(proved, "-"), "%.2f s" % (seconds or 0.0)))


#: instance scales of paper Table 1: name -> (values of n, values of K)
SCALES = {
    "small":  ([6, 8, 10, 12], [2, 3]),
    "medium": ([20, 50, 100], [3, 5]),
    "large":  ([200, 500, 1000], [5, 10, 20]),
}


#: number of distinct weights of the two wide regimes of paper Section 6.1.1; the
#: small scale uses the narrow regime instead, whose count is the parameter K
WIDE_K = {"indep": 3, "corr": 9}

#: weight settings of paper Section 6.1.1
#:   "narrow"  weights uniform on {1, ..., K} with K small, so that few weight
#:             classes keep the exact methods tractable;
#:   "indep"   weights uniform on {1, ..., 3}, independent of the processing times;
#:   "corr"    the jobs split into 9 weight classes by the rank of a_j + b_j, so
#:             that the heaviest jobs are also the longest;
#:   "free"    weights uniform on {1, ..., n}, the unrestricted regime of the
#:             large scale of paper Table 1: as many values as there are jobs.
WEIGHT_SETTINGS = ("narrow", "indep", "corr", "free")


def gen_instance(n, K, rng, geometric=False, weights="narrow", proc_hi=PROC_HI):
    """
    One random instance of the benchmark protocol (paper Section 6).

    a_j, b_j ~ U{PROC_LO, ..., proc_hi}, the upper end of the range being the one
    paper Section 6.1 gives the scale: [1,20] and [1,40] on the small scale,
    where it is varied, and [1,10] on the medium and the large one.  The weights
    follow the setting `weights`: the narrow regime draws them uniformly from
    {1, ..., K}, so that K is the number of values they can take; the
    independent regime draws them uniformly from {1, ..., 3}; the correlated
    regime sorts the jobs by a_j + b_j and cuts that order into 9 equal classes,
    so that the heaviest jobs are also the longest; and the free regime draws
    them uniformly from {1, ..., n}, which places no bound on how many values
    they take.  `geometric` draws them from the ladder {1, 10, ..., 10^(K-1)},
    the rule of the small scale of paper Section 6.1, so that the K weight
    classes are decades apart rather than neighbouring integers; it takes
    precedence over `weights` when set.
    """
    if weights not in WEIGHT_SETTINGS:
        raise ValueError("unknown weight setting: %r" % (weights,))
    a = [rng.randint(PROC_LO, proc_hi) for _ in range(n)]
    b = [rng.randint(PROC_LO, proc_hi) for _ in range(n)]
    if geometric:
        w = [10 ** rng.randrange(K) for _ in range(n)]
    elif weights == "narrow":
        w = [rng.randint(1, K) for _ in range(n)]
    elif weights == "indep":
        w = [rng.randint(1, WIDE_K["indep"]) for _ in range(n)]
    elif weights == "free":
        w = [rng.randint(1, n) for _ in range(n)]
    else:                                   # "corr"
        k = WIDE_K["corr"]
        order = sorted(range(n), key=lambda j: a[j] + b[j])
        w = [0] * n
        for rank, j in enumerate(order):
            w[j] = 1 + rank * k // n
    return a, b, w


def benchmark(n=8, K=2, instances=INSTANCES_PER_CONFIG, seed=42,
              geometric=False, weights="narrow", repeats=1, time_limit=None,
              progress=None, proc_hi=PROC_HI, log_path=None):
    """
    Run the exact DP on `instances` random instances and aggregate the result.

    The DP is deterministic (`repeats` > 1 changes nothing) and exact, so the
    mean objective it reports is the mean optimum of the sampled instances --
    the reference value z_ref of paper Section 6.1.  Its running time
    O(n K V^(3K-3)) is pseudo-polynomial, so the benchmark is meaningful only on
    small configurations; `time_limit` (seconds) aborts an instance that exceeds
    it, and such instances are counted in `failed` rather than dropped.

    Returns
    -------
    dict with keys
        mean_obj  : mean over the solved instances of the optimal objective
                    (comparison result of paper Section 6.1.3)
        best_obj  : best objective value over all instances
        mean_time : mean CPU time of a single run, in seconds
        solved    : number of instances solved within the time limit
        failed    : number of instances aborted by the time limit
        objs, times : the per-instance values
    """
    # Records go to the file the caller names, or to one of our own tagged with
    # the time this benchmark started.
    if log_path is None:
        log_path = (RECORD_DIR / ("%s_%s.txt"
                                  % (LABEL, datetime.now().strftime("%Y%m%d-%H%M%S"))))
    rng = random.Random(seed)
    objs, times = [], []
    failed = 0
    for idx in range(instances):
        a, b, w = gen_instance(n, K, rng, geometric, weights, proc_hi)
        best = float("inf")
        total = 0.0
        for _ in range(repeats):
            t0 = time.perf_counter()
            try:
                res = solve_dp(a, b, w, time_limit=time_limit)
            except TimeoutError:
                failed += 1
                break
            total += time.perf_counter() - t0
            best = min(best, res["obj"])
        objs.append(best if best < float("inf") else None)
        times.append(total / repeats if best < float("inf") else None)
        record_instance(log_path, n, weights, seed, idx + 1, instances,
                        objs[-1], times[-1], K,          # on disk before the next
                        proc_hi=proc_hi, proved=objs[-1] is not None)
        if progress is not None:          # live line for a long-running study
            # the dynamic program is exact, so a value it returns within the
            # time limit is the optimum of the instance
            progress(idx + 1, instances, objs[-1], times[-1],
                     objs[-1] is not None)

    vals = [o for o in objs if o is not None]
    secs = [t for t in times if t is not None]
    return {
        "n": n, "K": K, "instances": instances,
        "mean_obj": statistics.fmean(vals) if vals else float("nan"),
        "best_obj": min(vals) if vals else float("nan"),
        "mean_time": statistics.fmean(secs) if secs else float("nan"),
        "solved": len(vals),
        "failed": failed,
        "objs": objs,
        "times": times,
    }


def report(res, label="DP"):
    """Single-line summary of a benchmark result."""
    return ("%-8s n=%4d K=%2d | solved=%2d/%2d | mean opt=%9.2f | best opt=%9.2f"
            " | mean time=%9.4f s"
            % (label, res["n"], res["K"], res["solved"], res["instances"],
               res["mean_obj"], res["best_obj"], res["mean_time"]))


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

    # ---- benchmark: INSTANCES_PER_CONFIG instances per configuration -------
    # The DP is pseudo-polynomial, so only the small scale is attempted; the
    # time limit flags configurations the state space makes intractable.
    print()
    print("Benchmark, %d instances per configuration (paper Section 6.1):"
          % INSTANCES_PER_CONFIG)
    for n in SCALES["small"][0]:
        for K in SCALES["small"][1]:
            print("  " + report(benchmark(n=n, K=K, seed=1000 * n + K,
                                          time_limit=60.0)))


if __name__ == "__main__":
    main()
