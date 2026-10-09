from __future__ import annotations

from datetime import datetime
from pathlib import Path

import random
import statistics
import time
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


def instance_baseline(a, b, w, order=None):
    """
    The reference value of one instance, the lower bound

        LB = max { LB_1, LB_2, LB_3 }

    of paper Section 4.3.  The jobs are taken in non-increasing order of weight,
    so that no prefix S = {J_1, ..., J_j} of that order holds a weight below w_j,
    and the three bounds read

        LB_1 = max_j w_j * sum_{h in S} a_h,     the M1 work of the prefix,
        LB_2 = max_j w_j * sum_{h in S} b_h,     its M2 counterpart,
        LB_3 = max_j w_j * (a_j + b_j),          a single job on its own.

    Each of them bounds WCmax from below.  For LB_1, the job of S that finishes
    last on M1 completes no earlier than the M1 work of S and carries a weight of
    at least w_j; LB_2 reads the same on M2; and no job completes before its own
    two operations are done.  LB is therefore at most the optimum, which is what
    makes every gap of the benchmark non-negative.

    The value is a property of the instance alone: `order` is accepted for the
    callers that still hand one in, and ignored, so that every method measured on
    an instance is measured against the same reference.
    """
    jobs = sorted(range(len(w)), key=lambda j: -w[j])   # w_1 >= ... >= w_n
    sa = sb = 0.0
    lb1 = lb2 = lb3 = 0.0
    for j in jobs:                                      # prefix S grows by J_j
        sa += a[j]
        sb += b[j]
        lb1 = max(lb1, w[j] * sa)
        lb2 = max(lb2, w[j] * sb)
        lb3 = max(lb3, w[j] * (a[j] + b[j]))
    return max(lb1, lb2, lb3)

#: instance scales: name -> (values of n, values of K).  The small and the large
#: scale are the two of paper Table 1; on the large one the weights are
#: unrestricted, so K is taken as the number of jobs.  The medium scale is the
#: one run_medium.py keeps as a test, the paper reporting no results at it.
SCALES = {
    "small":  ([20, 40, 60], [2, 3]),
    "medium": ([200, 350, 500], [3, 9, 12]),
    "large":  ([500, 600, 700], [500, 600, 700]),
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

    a_j, b_j ~ U{PROC_LO, ..., proc_hi}, the upper end of the range being one of
    the three of paper Section 6.1, [1,20], [1,40] and [1,60], which both scales
    cross with their other factors.  The weights
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




def benchmark(n=50, K=3, instances=INSTANCES_PER_CONFIG, seed=42,
              geometric=False, weights="narrow", repeats=1, progress=None,
              proc_hi=PROC_HI, log_path=None):
    """
    Run the method on `instances` random instances and aggregate the result.

    For every instance the method is run `repeats` times; the best objective
    value found is recorded together with the mean CPU time of a single run.
    NEH is deterministic, so `repeats` > 1 changes nothing but the timing.

    Returns
    -------
    dict with keys
        mean_obj  : mean over instances of the best objective value (comparison
                    result of paper Section 6.1.3)
        best_obj  : best objective value over all instances
        mean_time : mean CPU time of a single run, in seconds
        objs, times : the per-instance values
    """
    # Records go to the file the caller names, or to one of our own tagged with
    # the time this benchmark started.
    if log_path is None:
        log_path = (RECORD_DIR / ("%s_%s.txt"
                                  % (LABEL, datetime.now().strftime("%Y%m%d-%H%M%S"))))
    rng = random.Random(seed)
    objs, times, bases = [], [], []
    for idx in range(instances):
        a, b, w = gen_instance(n, K, rng, geometric, weights, proc_hi)
        best = float("inf")
        best_order = None
        total = 0.0
        for _ in range(repeats):
            t0 = time.perf_counter()
            order = neh(a, b, w)                      # method under test
            total += time.perf_counter() - t0
            obj = wcmax_of_order(order, a, b, w)
            if obj < best:
                best, best_order = obj, order
        # the reference value of this instance: the lower bound LB of paper
        # Section 4.3, which the instance alone determines
        bases.append(None if best_order is None
                     else instance_baseline(a, b, w, best_order))
        objs.append(best)
        times.append(total / repeats)
        record_instance(log_path, n, weights, seed, idx + 1, instances,
                        objs[-1], times[-1], K,          # on disk before the next
                        proc_hi=proc_hi, base=bases[-1])
        if progress is not None:          # live line for a long-running study
            # a heuristic proves nothing about optimality, so the flag is None;
            # the reference is the one the driver estimated for the instance
            progress(idx + 1, instances, objs[-1], times[-1], None, bases[-1])

    return {
        "n": n, "K": K, "instances": instances,
        "mean_obj": statistics.fmean(objs),
        "best_obj": min(objs),
        "mean_time": statistics.fmean(times),
        "objs": objs,
        "times": times,
        "bases": bases,
    }


def report(res, label="NEH"):
    """Single-line summary of a benchmark result."""
    return ("%-8s n=%4d K=%2d | instances=%2d | mean obj=%9.2f | best obj=%9.2f"
            " | mean time=%9.4f s"
            % (label, res["n"], res["K"], res["instances"],
               res["mean_obj"], res["best_obj"], res["mean_time"]))


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

    # ---- benchmark: INSTANCES_PER_CONFIG instances per configuration -------
    print()
    print("Benchmark, %d instances per configuration (paper Section 6.1):"
          % INSTANCES_PER_CONFIG)
    for n, K in [(20, 3), (50, 3), (100, 3)]:
        print("  " + report(benchmark(n=n, K=K, seed=1000 * n + K)))
    print("  other scales: loop over SCALES, e.g. benchmark(n=n, K=K)"
          " for n in SCALES['large'][0]")



if __name__ == "__main__":
    main()
