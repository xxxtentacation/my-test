# -*- coding: utf-8 -*-
"""
Genetic Algorithm (GA) for F2 || WCmax (two-machine flow shop, minimizing the
maximum weighted completion time), implementing Section 5.3 of the paper
(paper_framework.tex): Algorithm (alg:pmx) partially mapped crossover, Algorithm
(alg:cm) critical-job mutation, and Algorithm (alg:ga) the genetic algorithm.

Chromosome.  A permutation sigma = (sigma(1), ..., sigma(n)) of the job indices;
the gene at position t is the job scheduled t-th.  Permutation encoding is used
because every permutation is a feasible schedule, so crossover and mutation need
no repair operator.  Decoding evaluates

    WCmax(sigma) = max_t w_{sigma(t)} C_{sigma(t)}

in O(n) time with a single forward pass over the two machines (paper
eq:completion).  Fitness is 1 / WCmax; only the ranking of fitness values is ever
used, so its absolute scale is irrelevant.

Initial population.  N individuals: the three structured seeds, which already
carry the two ingredients of a good schedule (small makespan, heavy jobs early),

    sigma^J     Johnson's rule, minimizes the makespan
    sigma^W     non-increasing weight, ties broken by Johnson's rule
    sigma^NEH   the NEH heuristic (NEH.py / Algorithm alg:neh)

plus N - 3 uniform random permutations supplying diversity.

Operators.
    PMX (Algorithm alg:pmx) copies a contiguous block of jobs from one parent and
    fills the remaining positions from the other, resolving conflicts through the
    induced mapping so that the child is always a valid permutation.
    Critical-job mutation (Algorithm alg:cm) locates the job attaining WCmax and
    tries to move it to every earlier position, keeping the best insertion; if no
    insertion improves the value, it swaps two jobs from different weight classes
    with probability p_m.

Note on p_m.  Algorithm alg:ga applies Algorithm alg:cm to each offspring with
probability p_m, and Algorithm alg:cm itself uses p_m for its fallback random
swap.  Both uses are implemented as written in the paper.
"""

from __future__ import annotations

import random
import statistics
import time
from typing import List, Optional, Sequence

from NEH import neh as neh_heuristic


def wcmax_of_order(order: Sequence[int], a: Sequence[float], b: Sequence[float],
                   w: Sequence[float]) -> float:
    """
    Compute the WCmax of a given permutation via the closed form recurrence
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


def _critical_job(sigma: Sequence[int], a: Sequence[float], b: Sequence[float],
                  w: Sequence[float]) -> tuple:
    """
    Return (position, value) of a job attaining WCmax(sigma), i.e. the critical
    job of paper Section 5.3.  Standard flow-shop forward pass; ties go to the
    earliest such job.
    """
    c1 = 0.0
    c2 = 0.0
    best_val = -1.0
    best_pos = 0
    for t, j in enumerate(sigma):
        c1 += a[j]
        c2 = max(c1, c2) + b[j]
        val = w[j] * c2
        if val > best_val:
            best_val = val
            best_pos = t
    return best_pos, best_val


def johnson_order(a: Sequence[float], b: Sequence[float]) -> List[int]:
    """
    sigma^J: Johnson's rule (paper Section 4.1).  Jobs with a_j <= b_j are
    scheduled first in non-decreasing order of a_j; the remaining jobs follow in
    non-increasing order of b_j.  This minimizes the makespan.
    """
    n = len(a)
    early = sorted((j for j in range(n) if a[j] <= b[j]), key=lambda j: a[j])
    late = sorted((j for j in range(n) if a[j] > b[j]), key=lambda j: -b[j])
    return early + late


def weight_descending_order(a: Sequence[float], b: Sequence[float],
                            w: Sequence[float]) -> List[int]:
    """
    sigma^W: order the jobs by non-increasing weight w_j, breaking ties by
    Johnson's rule.  Heavy jobs thereby receive the earliest positions, which
    attacks the bottleneck objective directly.
    """
    johnson_rank = {j: t for t, j in enumerate(johnson_order(a, b))}
    return sorted(range(len(w)), key=lambda j: (-w[j], johnson_rank[j]))


def pmx(p1: Sequence[int], p2: Sequence[int], cut1: int, cut2: int) -> List[int]:
    """
    Algorithm alg:pmx: partially mapped crossover.

    Step 1  copy the middle segment p1[cut1..cut2] verbatim into the child;
    Step 2  fill every other position t with p2[t]; whenever that job already
            occurs in the child, replace it through the mapping induced by the
            segment, j <- p2[ p1^{-1}(j) ], until an unused job is reached.

    The mapping is a permutation, so the chain terminates; a guard is kept only
    as a safety net.  Swapping the roles of p1 and p2 yields the second child.
    """
    n = len(p1)
    lo, hi = (cut1, cut2) if cut1 <= cut2 else (cut2, cut1)
    pos_in_p1 = {job: t for t, job in enumerate(p1)}

    child: List[Optional[int]] = [None] * n
    used = set()
    for t in range(lo, hi + 1):
        child[t] = p1[t]
        used.add(p1[t])

    for t in list(range(0, lo)) + list(range(hi + 1, n)):
        j = p2[t]
        guard = 0
        while j in used:
            j = p2[pos_in_p1[j]]          # follow the PMX mapping
            guard += 1
            if guard > n:                 # cannot happen for a valid permutation
                break
        child[t] = j
        used.add(j)

    return [int(x) for x in child]


def _random_cross_class_swap(sigma: Sequence[int], w: Sequence[float],
                             rng: random.Random) -> List[int]:
    """
    Swap two jobs of `sigma` lying in different weight classes (Algorithm
    alg:cm, Step 3 fallback).  If every job has the same weight there is no such
    pair, and the schedule is returned unchanged.
    """
    n = len(sigma)
    p = rng.randrange(n)
    wp = w[sigma[p]]
    others = [q for q in range(n) if q != p and w[sigma[q]] != wp]
    if not others:
        return list(sigma)
    q = rng.choice(others)
    out = list(sigma)
    out[p], out[q] = out[q], out[p]
    return out


def critical_job_mutation(sigma: Sequence[int], a: Sequence[float],
                          b: Sequence[float], w: Sequence[float],
                          pm: float, rng: random.Random) -> List[int]:
    """
    Algorithm alg:cm: critical-job mutation.

    Step 1  locate a critical job j* attaining WCmax and its position t*;
    Step 2  move j* to every earlier position and evaluate the result;
    Step 3  keep the best insertion if it improves WCmax, otherwise perform a
            random cross-weight-class swap with probability pm.

    Moving j* earlier can only decrease its own completion time (its machine-2
    work is unchanged while it may start earlier), so this is a cheap targeted
    hill-climbing step on the job that currently decides the objective.
    """
    n = len(sigma)
    if n < 2:
        return list(sigma)

    cur = _wcmax_fast(sigma, a, b, w)

    # Step 1: the critical job and its position.
    t_star, _ = _critical_job(sigma, a, b, w)
    j_star = sigma[t_star]

    # Step 2: try every insertion of j* into an earlier position r < t*.
    best_val = cur
    best_sigma: Optional[List[int]] = None
    for r in range(t_star):
        cand = list(sigma[:r]) + [j_star] + list(sigma[r:t_star]) \
            + list(sigma[t_star + 1:])
        val = _wcmax_fast(cand, a, b, w)
        if val < best_val:
            best_val = val
            best_sigma = cand

    # Step 3.
    if best_sigma is not None:
        return best_sigma
    if rng.random() < pm:
        return _random_cross_class_swap(sigma, w, rng)
    return list(sigma)


def ga(a: Sequence[float], b: Sequence[float], w: Sequence[float],
       N: Optional[int] = None, G: int = 100, pc: float = 0.9, pm: float = 0.1,
       lam: Optional[int] = None, seed: Optional[int] = None) -> List[int]:
    """
    Genetic algorithm for F2 || WCmax  (Algorithm alg:ga).

    Parameters
    ----------
    a, b, w : length n
        Processing time on M1, processing time on M2, and weight of each job.
    N       : population size (default 2n; the paper sets N = Theta(n)).
    G       : number of generations.
    pc      : crossover probability; with probability 1 - pc the two parents are
              copied into the offspring set unchanged.
    pm      : mutation probability.  Used twice, as in the paper: Algorithm
              alg:cm is applied to each offspring with probability pm, and its
              Step 3 fallback swap also uses pm.
    lam     : elitism size (default N // 10): the best lam individuals of the
              previous population survive into the next one.
    seed    : optional random seed.

    Returns
    -------
    list of job indices (0-based) of the best schedule sigma* found.
    """
    n = len(a)
    if n != len(b) or n != len(w):
        raise ValueError("a, b, w must have the same length")
    if n == 0:
        return []
    if N is None:
        N = max(8, 2 * n)
    if lam is None:
        lam = max(1, N // 10)
    lam = max(0, min(lam, N))

    rng = random.Random(seed)

    # ---- Step 1: initialization -------------------------------------------
    # The three structured seeds, then N - 3 uniform random permutations.
    pop: List[List[int]] = [
        johnson_order(a, b),                       # sigma^J
        weight_descending_order(a, b, w),          # sigma^W
        list(neh_heuristic(a, b, w)),              # sigma^NEH
    ]
    while len(pop) < N:
        perm = list(range(n))
        rng.shuffle(perm)
        pop.append(perm)
    pop = pop[:N]

    fit = [_wcmax_fast(s, a, b, w) for s in pop]   # smaller is better
    best_idx = min(range(len(pop)), key=lambda i: fit[i])
    sigma_star: List[int] = list(pop[best_idx])
    best_val = fit[best_idx]

    # ---- Step 2: evolution -------------------------------------------------
    for _g in range(G):
        # Selection: binary tournament, pool of size N.  Only the ranking of
        # fitness matters, so the pool stays selective as the population
        # converges and the WCmax values of near-optimal schedules get close.
        pool: List[List[int]] = []
        for _ in range(N):
            i1 = rng.randrange(len(pop))
            i2 = rng.randrange(len(pop))
            pool.append(pop[i1] if fit[i1] <= fit[i2] else pop[i2])

        # Crossover.
        offspring: List[List[int]] = []
        while len(offspring) < N:
            p1 = pool[rng.randrange(len(pool))]
            p2 = pool[rng.randrange(len(pool))]
            if n >= 2 and rng.random() < pc:
                cut1 = rng.randrange(n)
                cut2 = rng.randrange(n)
                kids = [pmx(p1, p2, cut1, cut2), pmx(p2, p1, cut1, cut2)]
            else:
                kids = [list(p1), list(p2)]
            for kid in kids:
                if len(offspring) < N:
                    offspring.append(kid)

        # Mutation: Algorithm alg:cm on each offspring with probability pm.
        for k in range(len(offspring)):
            if rng.random() < pm:
                offspring[k] = critical_job_mutation(
                    offspring[k], a, b, w, pm, rng)
        off_fit = [_wcmax_fast(s, a, b, w) for s in offspring]

        # Replacement: the best lam of P_{g-1} plus the N - lam best offspring.
        prev_best = sorted(range(len(pop)), key=lambda i: fit[i])[:lam]
        off_best = sorted(range(len(offspring)), key=lambda i: off_fit[i])
        new_pop = [list(pop[i]) for i in prev_best]
        new_fit = [fit[i] for i in prev_best]
        for i in off_best[:max(0, N - lam)]:
            new_pop.append(offspring[i])
            new_fit.append(off_fit[i])
        pop, fit = new_pop, new_fit

        # Update the incumbent.
        best_idx = min(range(len(pop)), key=lambda i: fit[i])
        if fit[best_idx] < best_val:
            best_val = fit[best_idx]
            sigma_star = list(pop[best_idx])

    # ---- Step 3: local search ---------------------------------------------
    # Apply Algorithm alg:cm to sigma* until it yields no improvement.  Passing
    # pm = 0 disables the fallback random swap, so the loop is exactly the
    # "keep inserting while it helps" descent.
    while True:
        cand = critical_job_mutation(sigma_star, a, b, w, 0.0, rng)
        val = _wcmax_fast(cand, a, b, w)
        if val < best_val:
            sigma_star, best_val = cand, val
        else:
            break

    return sigma_star


def run_ga(a, b, w, **kwargs) -> dict:
    """Convenience wrapper returning a result dict."""
    order = ga(a, b, w, **kwargs)
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
              geometric=False, repeats=1, method="ga", **kwargs):
    """
    Run one method on `instances` random instances and aggregate the result.

    The stochastic method (the genetic algorithm) is run `repeats` times per
    instance from independent seeds; the best objective value found is recorded
    together with the mean CPU time of a single run, and both are averaged over
    the instances.  Any extra keyword argument is forwarded to `ga`
    (N, G, pc, pm, lam).

    Parameters
    ----------
    method : "ga"        the genetic algorithm of Section 5.3
             "johnson"   Johnson's rule, the weight-blind baseline
             "2-approx"  the weight-descending schedule of Section 4.3
                         (the 2-approximation algorithm)

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
            if method == "ga":
                order = ga(a, b, w, seed=rng.randrange(1 << 30), **kwargs)
            elif method == "johnson":
                order = johnson_order(a, b)
            elif method in ("2-approx", "weight"):
                order = weight_descending_order(a, b, w)
            else:
                raise ValueError("unknown method: %r" % (method,))
            total += time.perf_counter() - t0
            best = min(best, _wcmax_fast(order, a, b, w))
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


def report(res, label="GA"):
    """Single-line summary of a benchmark result."""
    return ("%-8s n=%4d K=%2d | instances=%2d | mean obj=%9.2f | best obj=%9.2f"
            " | mean time=%9.4f s"
            % (label, res["n"], res["K"], res["instances"],
               res["mean_obj"], res["best_obj"], res["mean_time"]))


def main() -> None:  # pragma: no cover
    """Random instance generated from a fixed seed, as in MILP.py."""
    # random-instance parameters
    seed = 42       # random seed
    n = 50          # number of jobs
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

    res = run_ga(a, b, w, G=100, pc=0.9, pm=0.1, seed=seed)
    print("GA result:")
    print("  WCmax :", res["obj"])
    print("  order :", [j + 1 for j in res["order"]], "(1-based)")
    # cross-check with the closed-form recurrence
    print("  verify:", wcmax_of_order(res["order"], a, b, w))

    # Optional: compare with the seeds and with the other heuristics.
    for name, order in (("Johnson", johnson_order(a, b)),
                        ("Weight ", weight_descending_order(a, b, w)),
                        ("NEH    ", neh_heuristic(a, b, w))):
        print(f"  {name}:", wcmax_of_order(order, a, b, w))
    try:
        from ACO import aco
        aco_order = aco(a, b, w, m=20, max_iter=50, seed=seed)
        print("  ACO    :", wcmax_of_order(aco_order, a, b, w))
    except Exception as exc:  # noqa: BLE001
        print("  ACO comparison skipped:", exc)

    # ---- benchmark: INSTANCES_PER_CONFIG instances per configuration -------
    print()
    print("Benchmark, %d instances per configuration (paper Section 6.1):"
          % INSTANCES_PER_CONFIG)
    for n, K in [(20, 3), (50, 3), (100, 3)]:
        for method in ("2-approx", "johnson", "ga"):
            print("  " + report(benchmark(n=n, K=K, seed=1000 * n + K,
                                          method=method), label=method))
    print("  other scales: loop over SCALES, e.g. benchmark(n=n, K=K)"
          " for n in SCALES['large'][0]")


if __name__ == "__main__":
    main()
