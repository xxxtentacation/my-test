from __future__ import annotations

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

#: instance scales of paper Table 1: name -> (values of n, values of K)
SCALES = {
    "small":  ([6, 8, 10, 12], [2, 3]),
    "medium": ([20, 50, 100], [3, 5]),
    "large":  ([200, 500, 1000], [5, 10, 20]),
}


def gen_instance(n, K, rng, geometric=False):
    """
    One random instance of the benchmark protocol (paper Section 6.1.1).

    a_j, b_j ~ U{PROC_LO, ..., PROC_HI}; the weights are either uniform on
    {1, ..., K}, or the geometrically spaced ladder {1, 2, 4, ..., 2^(K-1)}
    which stresses the weight-rounding argument of Section 4.4.2.
    """
    a = [rng.randint(PROC_LO, PROC_HI) for _ in range(n)]
    b = [rng.randint(PROC_LO, PROC_HI) for _ in range(n)]
    if geometric:
        w = [1 << rng.randrange(K) for _ in range(n)]
    else:
        w = [rng.randint(1, K) for _ in range(n)]
    return a, b, w


def benchmark(n=50, K=3, instances=INSTANCES_PER_CONFIG, seed=42,
              geometric=False, repeats=1):
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
    rng = random.Random(seed)
    objs, times = [], []
    for _ in range(instances):
        a, b, w = gen_instance(n, K, rng, geometric)
        best = float("inf")
        total = 0.0
        for _ in range(repeats):
            t0 = time.perf_counter()
            order = neh(a, b, w)                      # method under test
            total += time.perf_counter() - t0
            best = min(best, wcmax_of_order(order, a, b, w))
        objs.append(best)
        times.append(total / repeats)

    return {
        "n": n, "K": K, "instances": instances,
        "mean_obj": statistics.fmean(objs),
        "best_obj": min(objs),
        "mean_time": statistics.fmean(times),
        "objs": objs,
        "times": times,
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
