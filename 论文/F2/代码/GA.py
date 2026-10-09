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
eq:completion).

Objectives.  A schedule is judged by the pair

    ( C_max(sigma), w_last(sigma) ),

its makespan and the weight of the job that closes it, both to be minimized.
The pair bounds the criterion from below, WCmax >= w_last * C_max, so the
trade-off between the two is exactly the one the criterion rewards, and the
population is carried forward by the non-dominated sorting and crowding
distance of NSGA-II rather than by a single scalar fitness.  The schedule the
algorithm reports is the one of smallest WCmax over the final population, which
is also what the benchmark of the paper measures.

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

from datetime import datetime
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Bi-objective machinery
#
# A schedule carries two objectives, both to be minimized: its makespan C_max
# and the weight w_last of the job that closes it.  They bound the criterion of
# the paper from below, WCmax = max_j w_j C_j >= w_last * C_max, and the two
# together carry its scale: the criterion is a weighted completion, so a search
# that only pressed the makespan would leave the weights out of the account.
# ---------------------------------------------------------------------------

def objectives(order: Sequence[int], a: Sequence[float], b: Sequence[float],
               w: Sequence[float]):
    """The pair (C_max, w_last): makespan, and weight of the job closing it."""
    c1 = c2 = 0.0
    for j in order:
        c1 += a[j]
        c2 = max(c1, c2) + b[j]
    return c2, w[order[-1]]


def _dominates(p, q) -> bool:
    """True when the pair p dominates q: no worse in both, better in one."""
    return (p[0] <= q[0] and p[1] <= q[1]) and (p[0] < q[0] or p[1] < q[1])


def _pareto_fronts(objs: Sequence[tuple]) -> List[List[int]]:
    """
    Fast non-dominated sorting of `objs`, the best front first.

    Returns the index lists of the Pareto fronts; an individual of a front is
    dominated by no individual of the fronts before it.
    """
    n = len(objs)
    dominated_by = [0] * n
    beats: List[List[int]] = [[] for _ in range(n)]
    first: List[int] = []
    for p in range(n):
        for q in range(p + 1, n):
            if _dominates(objs[p], objs[q]):
                beats[p].append(q)
                dominated_by[q] += 1
            elif _dominates(objs[q], objs[p]):
                beats[q].append(p)
                dominated_by[p] += 1
        if dominated_by[p] == 0:
            first.append(p)
    fronts, cur = [], first
    while cur:
        fronts.append(cur)
        nxt: List[int] = []
        for p in cur:
            for q in beats[p]:
                dominated_by[q] -= 1
                if dominated_by[q] == 0:
                    nxt.append(q)
        cur = nxt
    return fronts


def _crowding(objs: Sequence[tuple], front: Sequence[int]) -> dict:
    """Crowding distance of one front; the extremes of each objective are infinite."""
    dist = {p: 0.0 for p in front}
    if len(front) <= 2:
        return {p: float("inf") for p in front}
    for k in (0, 1):
        line = sorted(front, key=lambda p: objs[p][k])
        lo, hi = objs[line[0]][k], objs[line[-1]][k]
        dist[line[0]] = dist[line[-1]] = float("inf")
        if hi <= lo:
            continue
        for t in range(1, len(line) - 1):
            dist[line[t]] += ((objs[line[t + 1]][k] - objs[line[t - 1]][k])
                              / (hi - lo))
    return dist


def _rank_crowding(objs: Sequence[tuple]):
    """Pareto rank and crowding distance of every individual, and the fronts."""
    rank, crowd = {}, {}
    fronts = _pareto_fronts(objs)
    for f, front in enumerate(fronts):
        dist = _crowding(objs, front)
        for p in front:
            rank[p] = f
            crowd[p] = dist[p]
    return rank, crowd, fronts


def _nsga2_survive(objs: Sequence[tuple], N: int) -> List[int]:
    """
    Environmental selection of NSGA-II: the indices of the N survivors.

    Whole fronts are kept while they fit; the front that overflows the budget is
    cut by crowding distance, its least crowded individuals first.
    """
    _rank, crowd, fronts = _rank_crowding(objs)
    keep: List[int] = []
    for front in fronts:
        if len(keep) + len(front) <= N:
            keep.extend(front)
        else:
            need = N - len(keep)
            keep.extend(sorted(front, key=lambda p: -crowd[p])[:need])
            break
    return keep


def _initial_population(a: Sequence[float], b: Sequence[float],
                        w: Sequence[float], N: int,
                        rng: random.Random) -> List[List[int]]:
    """The three structured seeds, then random permutations up to N individuals."""
    n = len(a)
    pop: List[List[int]] = [
        johnson_order(a, b),                       # sigma^J
        weight_descending_order(a, b, w),          # sigma^W
        list(neh_heuristic(a, b, w)),              # sigma^NEH
    ]
    while len(pop) < N:
        perm = list(range(n))
        rng.shuffle(perm)
        pop.append(perm)
    return pop[:N]


def _evolve(pop: List[List[int]], a: Sequence[float], b: Sequence[float],
            w: Sequence[float], N: int, G: int, pc: float, pm: float,
            rng: random.Random):
    """
    The NSGA-II loop: tournament on (rank, crowding), PMX, critical-job mutation,
    and (mu + lambda) survival by non-dominated sorting.

    Returns the population reached after G generations, its objective pairs, and
    the best schedule on the criterion itself -- WCmax -- met along the way.  The
    criterion is not one of the two objectives the survival ranks, so a schedule
    that is excellent on it need not survive; tracking it separately is what lets
    the algorithm report the best WCmax it saw rather than the best of the last
    population.
    """
    n = len(a)
    objs = [objectives(s, a, b, w) for s in pop]
    elite = min(pop, key=lambda s: _wcmax_fast(s, a, b, w))
    elite_val = _wcmax_fast(elite, a, b, w)
    elite = list(elite)

    for _ in range(G):
        rank, crowd, _ = _rank_crowding(objs)

        # Selection: binary tournament; the better (rank, crowding) wins.
        pool: List[List[int]] = []
        for _ in range(N):
            i, j = rng.randrange(N), rng.randrange(N)
            pick = i if (rank[i], -crowd[i]) <= (rank[j], -crowd[j]) else j
            pool.append(pop[pick])

        # Crossover: PMX, or the two parents copied.
        offspring: List[List[int]] = []
        while len(offspring) < N:
            p1 = pool[rng.randrange(len(pool))]
            p2 = pool[rng.randrange(len(pool))]
            if n >= 2 and rng.random() < pc:
                cut1, cut2 = rng.randrange(n), rng.randrange(n)
                kids = [pmx(p1, p2, cut1, cut2), pmx(p2, p1, cut1, cut2)]
            else:
                kids = [list(p1), list(p2)]
            for kid in kids:
                if len(offspring) < N:
                    offspring.append(kid)

        # Mutation: Algorithm alg:cm on each offspring with probability pm.
        for k in range(len(offspring)):
            if rng.random() < pm:
                offspring[k] = critical_job_mutation(offspring[k], a, b, w, pm, rng)

        # The criterion of the paper, tracked on every offspring.
        for s in offspring:
            val = _wcmax_fast(s, a, b, w)
            if val < elite_val:
                elite, elite_val = list(s), val

        # Survival: parents and offspring together, the best N of them.
        combined = pop + offspring
        all_objs = objs + [objectives(s, a, b, w) for s in offspring]
        keep = _nsga2_survive(all_objs, N)
        pop = [combined[i] for i in keep]
        objs = [all_objs[i] for i in keep]
    return pop, objs, elite, elite_val


def ga(a: Sequence[float], b: Sequence[float], w: Sequence[float],
       N: Optional[int] = None, G: int = 30, pc: float = 0.9, pm: float = 0.1,
       lam: Optional[int] = None, seed: Optional[int] = None) -> List[int]:
    """
    Genetic algorithm for F2 || WCmax, second generation (Algorithm alg:ga).

    The search is bi-objective: an individual is the pair made of its makespan
    C_max and the weight w_last of its last job, and the population is carried
    forward by the non-dominated sorting and crowding distance of NSGA-II, so
    that the whole trade-off between the two is explored rather than a single
    weighted value.  The two objectives bound the criterion of the paper from
    below, WCmax >= w_last * C_max, and the schedule the algorithm reports is the
    one minimizing WCmax over the final population, refined by the critical-job
    local search of Step 3.

    Parameters
    ----------
    a, b, w : length n
        Processing time on M1, processing time on M2, and weight of each job.
    N       : population size (default 2n; the paper sets N = Theta(n)).
    G       : number of generations (default 30).  Calibrated in Section 6.1.2:
              raising G to 120 leaves the mean gap unchanged on every tested
              configuration, so 30 generations already reach the fixed point.
    pc      : crossover probability; with probability 1 - pc the two parents are
              copied into the offspring set unchanged.
    pm      : mutation probability.  Used twice, as in the paper: Algorithm
              alg:cm is applied to each offspring with probability pm, and its
              Step 3 fallback swap also uses pm.
    lam     : elitism size.  Kept for the callers of the paper's run; the
              survivors are now chosen by non-dominated sorting, which is
              elitist by construction, so the argument no longer acts.
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
    pop = _initial_population(a, b, w, N, rng)

    # ---- Step 2: evolution -------------------------------------------------
    # NSGA-II on (C_max, w_last): fronts and crowding carry the selection, while
    # the best schedule on the criterion itself is tracked alongside them.
    pop, _objs, sigma_star, best_val = _evolve(pop, a, b, w, N, G, pc, pm, rng)

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
              geometric=False, weights="narrow", repeats=1, method="ga",
              progress=None, proc_hi=PROC_HI, log_path=None, **kwargs):
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
    # Records go to the file the caller names, or to one of our own tagged with
    # the time this benchmark started.
    if log_path is None:
        log_path = (RECORD_DIR / ("%s_%s.txt"
                                  % (LABEL, datetime.now().strftime("%Y%m%d-%H%M%S"))))
    # Two independent streams: one draws the instances, one only draws the seeds
    # of the restarts.  They must be separate, because drawing a solver seed from
    # the instance stream would shift the instances of every later index, and the
    # methods would then be compared on different samples of instances.
    rng = random.Random(seed)
    solver_rng = random.Random("solver-%d" % seed)
    objs, times, bases = [], [], []
    for idx in range(instances):
        a, b, w = gen_instance(n, K, rng, geometric, weights, proc_hi)
        best = float("inf")
        best_order = None
        total = 0.0
        for _ in range(repeats):
            t0 = time.perf_counter()
            if method == "ga":
                order = ga(a, b, w, seed=solver_rng.randrange(1 << 30), **kwargs)
            elif method in ("2-approx", "weight"):
                order = weight_descending_order(a, b, w)
            else:
                raise ValueError("unknown method: %r" % (method,))
            total += time.perf_counter() - t0
            obj = _wcmax_fast(order, a, b, w)
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
        for method in ("2-approx", "ga"):
            print("  " + report(benchmark(n=n, K=K, seed=1000 * n + K,
                                          method=method), label=method))
    print("  other scales: loop over SCALES, e.g. benchmark(n=n, K=K)"
          " for n in SCALES['large'][0]")


if __name__ == "__main__":
    main()
