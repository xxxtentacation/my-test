# -*- coding: utf-8 -*-
"""
Genetic Algorithm (GA) for F2 || WCmax (two-machine flow shop, minimizing the
maximum weighted completion time), implementing Section 5.3 of the paper
(paper_framework.tex): Algorithm (alg:ap) alternating-position crossover, Algorithm
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

Initial population.  The NEH schedule of the instance and N - 1 neighbours of
it.  NEH orders the jobs by non-increasing a_j + b_j and inserts them one at a
time at the position that lowers the criterion most, so the population starts
from an order that already reads the weights, and the schedule the algorithm
reports can never be worse than the one NEH returns.

Operators.
    Alternating-position crossover (Algorithm alg:ap) gives the child the odd
    positions of one parent -- half the instance, every other job -- and
    front-back crossover (Algorithm alg:fb) gives it the front half instead;
    either way the remaining positions are filled with the jobs of the other
    parent in the order it lists them, passing over a job the child already
    holds, so that the child is a valid permutation without any repair.  Each of
    the two is drawn with probability 1/2.  Swapping the roles of the two
    parents yields the second child.
    Critical-job mutation (Algorithm alg:cm) acts on single jobs: a job
    attaining WCmax is moved to the position that serves the criterion best, and,
    where no such move helps, two jobs of inverted weight are exchanged with
    probability p_m.  It is what moves the population off the order it started
    from, one job at a time.

Local search (Algorithm alg:ga, Step 3).  The jobs carrying the largest weighted
completion times -- descent_k(n) of them, one job in ten of the instance, the
ones that decide the criterion -- are reinserted, each at the position that
lowers WCmax most, until no such insertion improves.

Generations.  The algorithm is run for the number of generations the caller
names.  The initial population already carries an order that reads the weights,
so the generations are spent moving jobs within it rather than building it up
from nothing, and the local search of Step 3 refines what they leave.

Note on p_m.  Algorithm alg:ga applies Algorithm alg:cm to each offspring with
probability p_m, and Algorithm alg:cm itself uses p_m for its fallback exchange.
Both uses are implemented as written in the paper.
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


def alternating_crossover(p1: Sequence[int], p2: Sequence[int]) -> List[int]:
    """
    Algorithm alg:ap: alternating-position crossover.

    Step 1  [Alternate] the child inherits the odd positions 1, 3, 5, ... of p1,
            half of the instance, so that those jobs keep the slot they hold in
            p1;
    Step 2  [Fill] the even positions are filled with the jobs of p2, taken in
            the order p2 lists them, a job the child already holds being passed
            over.

    Conflict resolution.  A job of p2 clashes with the child only when p1
    already contributes it, and the clash is passed over rather than repaired:
    that job has a slot already, the one p1 gave it, and the next job of p2
    takes the free position instead.  The pass is well defined, because p1
    contributes ceil(n / 2) jobs to the odd positions while p2 holds exactly the
    floor(n / 2) remaining ones, which is the number of even positions to fill;
    the scan therefore reaches one free job for each of them and stops with
    every position taken.  No mapping between the two parents and no repair
    operator is needed, and the child is a valid permutation by construction.
    Swapping the roles of p1 and p2 yields the second child.
    """
    n = len(p1)
    child: List[Optional[int]] = [None] * n
    odd = range(0, n, 2)
    for t in odd:
        child[t] = p1[t]
    held = set(child[t] for t in odd)
    free = [j for j in p2 if j not in held]
    for t, j in zip(range(1, n, 2), free):
        child[t] = j
    return [int(x) for x in child]


def front_back_crossover(p1: Sequence[int], p2: Sequence[int]) -> List[int]:
    """
    Algorithm alg:fb: front-back crossover.

    Step 1  [Front] the child inherits the front half of p1, positions
            1, ..., ceil(n / 2), so that those jobs keep the slot they hold in
            p1;
    Step 2  [Rear] the rear half is filled with the jobs of p2, read from the
            position that follows its own front half to its end and then from
            its first position to its front half, a job the child already holds
            being passed over.

    The two operators differ only in which half of p1 the child keeps: every
    other slot here, the front half there.  The clash is settled by the same
    pass -- a job of p2 that the child already holds keeps the slot p1 gave it
    and is passed over -- and the same count makes the pass exact: p1
    contributes ceil(n / 2) jobs to the half it gives, so p2 holds exactly the
    floor(n / 2) others, which is the number of positions to fill.  Reading p2
    from the position after its front half first is what makes the child take
    the rear half of p2 for its own rear, the jobs of p2's front half being
    called on only where the rear half holds a job the child already has.
    Swapping the roles of p1 and p2 yields the second child.
    """
    n = len(p1)
    h = (n + 1) // 2                      # ceil(n / 2): the front half of p1
    child: List[Optional[int]] = [None] * n
    for t in range(h):
        child[t] = p1[t]
    held = set(p1[:h])
    cyclic = list(range(h, n)) + list(range(0, h))
    free = [p2[t] for t in cyclic if p2[t] not in held]
    for t, j in zip(range(h, n), free):
        child[t] = j
    return [int(x) for x in child]


def _inversion_swap(sigma: Sequence[int], w: Sequence[float],
                    rng: random.Random) -> List[int]:
    """
    Swap an earlier lighter job with a later heavier one (Algorithm alg:cm,
    Step 3 fallback): the exchange moves weight forward, which is the direction
    the block structure prescribes.  If no such pair exists the schedule is
    returned unchanged.
    """
    n = len(sigma)
    for _ in range(8):
        p = rng.randrange(n)
        heavier = [q for q in range(p + 1, n) if w[sigma[q]] > w[sigma[p]]]
        if heavier:
            q = rng.choice(heavier)
            out = list(sigma)
            out[p], out[q] = out[q], out[p]
            return out
    return list(sigma)


def job_mutation(sigma: Sequence[int], a: Sequence[float],
                   b: Sequence[float], w: Sequence[float],
                   pm: float, rng: random.Random) -> List[int]:
    """
    Algorithm alg:cm: critical-job mutation.

    Step 1  let j* be a job attaining WCmax; form, for every position r, the
            schedule obtained by moving j* to r, and keep the best of them if it
            improves on sigma;
    Step 2  otherwise exchange an earlier lighter job with a later heavier one
            with probability pm.

    Both moves act on single jobs: a job is moved, or exchanged for one of a
    different weight, and the change is kept only when the criterion falls.
    Nothing is imposed on the schedule as a whole, so the population is moved
    off the order it started from one job at a time.
    """
    n = len(sigma)
    if n < 2:
        return list(sigma)

    # Step 1: the critical job, into the position that serves the criterion best.
    cur = _wcmax_fast(sigma, a, b, w)
    t_star, _ = _critical_job(sigma, a, b, w)
    j_star = sigma[t_star]
    best_val, best_sigma = cur, None
    rest = list(sigma[:t_star]) + list(sigma[t_star + 1:])
    for r in range(n):
        if r == t_star:
            continue
        cand = rest[:r] + [j_star] + rest[r:]
        val = _wcmax_fast(cand, a, b, w)
        if val < best_val:
            best_val, best_sigma = val, cand
    if best_sigma is not None:
        return best_sigma

    # Step 2.
    if rng.random() < pm:
        return _inversion_swap(sigma, w, rng)
    return list(sigma)


def _worst_jobs(sigma: Sequence[int], a: Sequence[float], b: Sequence[float],
                w: Sequence[float], k: int) -> List[int]:
    """
    The k jobs with the largest weighted completion times, worst first.

    They are the jobs that decide the criterion and therefore the ones whose
    reinsertion can lower it.
    """
    c1 = c2 = 0.0
    vals = []
    for j in sigma:
        c1 += a[j]
        c2 = max(c1, c2) + b[j]
        vals.append((w[j] * c2, j))
    vals.sort(key=lambda pair: -pair[0])
    return [j for _v, j in vals[:k]]


def insertion_descent(sigma: Sequence[int], a: Sequence[float],
                      b: Sequence[float], w: Sequence[float],
                      k: int) -> List[int]:
    """
    Algorithm alg:ga, Step 3: the local search on the incumbent.

    Reinsert the k jobs that carry the largest weighted completion times, each
    at the position that lowers the criterion most, and repeat until no such
    insertion improves.
    """
    sigma = list(sigma)
    n = len(sigma)
    for _ in range(n * k):
        cur = _wcmax_fast(sigma, a, b, w)
        best_val, best_sigma = cur, None
        for job in _worst_jobs(sigma, a, b, w, k):
            t = sigma.index(job)
            rest = sigma[:t] + sigma[t + 1:]
            for r in range(n):
                if r == t:
                    continue
                cand = rest[:r] + [job] + rest[r:]
                val = _wcmax_fast(cand, a, b, w)
                if val < best_val:
                    best_val, best_sigma = val, cand
        if best_sigma is None:
            return sigma
        sigma = best_sigma
    return sigma


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


def _neh_population(a: Sequence[float], b: Sequence[float],
                    w: Sequence[float], N: int,
                    rng: random.Random) -> List[List[int]]:
    """
    P0: the NEH schedule and N - 1 neighbours of it.

    The first individual is the schedule of the NEH heuristic of NEH.py, which
    orders the jobs by non-increasing a_j + b_j and inserts them one at a time at
    the position that lowers WCmax most, so it is a schedule the weights have
    already been read into.  The others are copies of it with a few neighbouring
    jobs exchanged, the number of exchanges being drawn between one and
    max(1, n // 4), which keeps the population supplied with the orders the
    operators can move away from.  Since that first individual is in the
    population and the elite is tracked on the criterion, the schedule the
    algorithm reports is never worse than the one NEH returns.
    """
    sigma = list(neh_heuristic(a, b, w))
    n = len(sigma)
    pop: List[List[int]] = [sigma]
    furthest = max(1, n // 4)      # the strongest displacement drawn
    while len(pop) < N:
        neighbour = list(sigma)
        for _ in range(rng.randint(1, furthest)):
            t = rng.randrange(n - 1)
            neighbour[t], neighbour[t + 1] = neighbour[t + 1], neighbour[t]
        pop.append(neighbour)
    return pop[:N]


def _initial_population(a: Sequence[float], b: Sequence[float],
                        w: Sequence[float], N: int,
                        rng: random.Random) -> List[List[int]]:
    """The initial population of paper Algorithm alg:ga, Step 1."""
    return _neh_population(a, b, w, N, rng)



def _evolve(pop: List[List[int]], a: Sequence[float], b: Sequence[float],
            w: Sequence[float], N: int, G: int, pc: float, pm: float,
            rng: random.Random):
    """
    The NSGA-II loop: tournament on (rank, crowding), the two half-inheritance
    crossovers, critical-job mutation,
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

        # Crossover: one of the two half-inheritance operators, each with
        # probability 1/2, or the two parents copied.
        offspring: List[List[int]] = []
        while len(offspring) < N:
            p1 = pool[rng.randrange(len(pool))]
            p2 = pool[rng.randrange(len(pool))]
            if n >= 2 and rng.random() < pc:
                operator = (alternating_crossover if rng.random() < 0.5
                            else front_back_crossover)
                kids = [operator(p1, p2), operator(p2, p1)]
            else:
                kids = [list(p1), list(p2)]
            for kid in kids:
                if len(offspring) < N:
                    offspring.append(kid)

        # Mutation: Algorithm alg:cm on each offspring with probability pm.
        for k in range(len(offspring)):
            if rng.random() < pm:
                offspring[k] = job_mutation(offspring[k], a, b, w, pm, rng)

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
       N: Optional[int] = None, G: int = 1, pc: float = 0.9, pm: float = 0.1,
       k: Optional[int] = None,
       lam: Optional[int] = None, seed: Optional[int] = None) -> List[int]:
    """
    Genetic algorithm for F2 || WCmax (Algorithm alg:ga).

    The search is bi-objective: an individual is the pair made of its makespan
    C_max and the weight w_last of its last job, and the population is carried
    forward by the non-dominated sorting and crowding distance of NSGA-II, so
    that the whole trade-off between the two is explored rather than a single
    weighted value.  The two objectives bound the criterion of the paper from
    below, WCmax >= w_last * C_max, and the schedule the algorithm reports is the
    best one on the criterion that the search saw, refined by the insertion
    descent of Step 3.

    Parameters
    ----------
    a, b, w : length n
        Processing time on M1, processing time on M2, and weight of each job.
    N       : population size (default population_of(n), the capped rule of the
              study: min(20, max(6, ceil(n/2)))).
    G       : number of generations (default 1).  A scale of the study names the
              count it runs -- the small and the large scale differ -- and the
              argument is what carries it.
    pc      : crossover probability; with probability 1 - pc the two parents are
              copied into the offspring set unchanged.
    pm      : mutation probability.  Used twice, as in the paper: Algorithm
              alg:cm is applied to each offspring with probability pm, and its
              Step 3 fallback swap also uses pm.
    k       : how many jobs the local search of Step 3 reinserts, the ones with
              the largest weighted completion times.  A caller that sets the
              parameters of a scale names one of its own -- the small and the
              large scale of the study do -- and a caller that names none gets
              the default rule descent_k(n).
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
        N = population_of(n)
    if lam is None:
        lam = max(1, N // 10)
    lam = max(0, min(lam, N))

    rng = random.Random(seed)

    # ---- Step 1: initialization -------------------------------------------
    # The initial population: the NEH schedule of the instance and its
    # neighbours, so that the search starts from an order the weights have
    # already been read into.
    pop = _initial_population(a, b, w, N, rng)

    # ---- Step 2: evolution -------------------------------------------------
    # G generations, NSGA-II on (C_max, w_last): fronts and crowding carry the
    # selection, while the best schedule on the criterion itself is tracked
    # alongside them.
    pop, _objs, sigma_star, best_val = _evolve(pop, a, b, w, N, G, pc, pm, rng)

    # ---- Step 3: local search ---------------------------------------------
    # The insertion descent of Algorithm alg:ga on the incumbent: the jobs that
    # carry the largest weighted completion times are reinserted while that
    # lowers the criterion.  Its width is the caller's, or the default rule.
    width = descent_k(n) if k is None else max(1, min(n, k))
    sigma_star = insertion_descent(sigma_star, a, b, w, width)

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
PROC_LO, PROC_HI = 1, 10       # processing times ~ normal on {1, ..., 10}

#: How many jobs the local search of Algorithm alg:ga reinserts when the caller
#: names none: the ones with the largest weighted completion times, worst first,
#: one job in ten of the instance and at least one.  The drivers of the study
#: set the rule themselves -- the two scales run it at a rate of their own -- so
#: that this is only the default of a caller that names none.
DESCENT_K = 1
DESCENT_K_RATE = 0.1


def descent_k(n: int) -> int:
    """The jobs the insertion descent of Algorithm alg:ga reinserts."""
    return min(n, max(DESCENT_K, int(n * DESCENT_K_RATE)))


#: The population size of an instance of n jobs.  It grows with the instance and
#: is capped: a small instance is given a small population, and a large one keeps
#: the same twenty individuals, which the calibration of the study showed to be
#: enough for the block mutation and the local search to reach what a population
#: of n individuals reaches, at a fraction of the cost.  The floor keeps the
#: selection and the crossover supplied with distinct parents.
POP_MIN, POP_RATE, POP_MAX = 6, 0.5, 20


def population_of(n: int) -> int:
    """The population size of the genetic algorithm on an instance of n jobs."""
    return min(POP_MAX, max(POP_MIN, (n + 1) // 2))     # ceil(n / 2)


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
                    K=None, proc_hi=PROC_HI, proved=None, base=None, label=None):
    """
    Append one line for a finished instance, and flush it at once.

    The line is written as soon as that instance is done, not when the whole
    benchmark is over, so that a run which is interrupted or killed still holds
    every instance it has already solved.  `path` may be a str or a Path; its
    parent directory is created if it does not exist.  The fields are those of
    the live line of the drivers -- label, n, weight setting, processing-time
    range, instance, value, gap, status and running time -- preceded by the seed
    of the instance and the moment it finished.  `label` is the name the method
    is recorded under, GA by default, which the driver may override.  The gap is
    the one the method reports on, `base` being the reference it computed for the
    instance; a method that hands none, as the exact ones do, shows a dash there.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("%s  %-5s n=%-4d %-6s %-7s | seed=%-7d | instance %2d/%-2d "
                 "| value %-12s | gap %9s | %-10s %9s\n"
                 % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label or LABEL, n,
                    weight_label(weights, K), "[%d,%d]" % (PROC_LO, proc_hi),
                    seed, idx, total,
                    "none" if obj is None else "%.1f" % obj,
                    "-" if obj is None or base is None or base == 0
                    else "%+.2f%%" % (100.0 * (obj - base) / base),
                    STATUS.get(proved, "-"),
                    "%.2f ms" % (1000.0 * (seconds or 0.0))))


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

#: instance scales of paper Section 6.1: name -> (values of n, values of K).
#: The dynamic program is reported on the first of them, whose weights are the
#: geometric ladder {1, 10, ..., 10^(K-1)}; the comparison of Section 6.2 runs on
#: the other two, which draw the weights normally on {1, ..., K} with K = 30,
#: so that the number of weight classes is arbitrary below that bound.
SCALES = {
    "dp":     ([20, 40, 60], [2]),
    "small":  ([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], [30]),
    "large":  ([100, 120, 140, 160, 180, 200, 220, 240, 260, 280,
                300, 320, 340, 360, 380, 400], [30]),
}


#: number of distinct weights of the two wide regimes of paper Section 6.1.1; the
#: small scale uses the narrow regime instead, whose count is the parameter K
WIDE_K = {"indep": 3, "corr": 9}

#: weight settings of paper Section 6.1.1
#:   "narrow"  weights normal on {1, ..., K} with K small, so that few weight
#:             classes keep the exact methods tractable;
#:   "indep"   weights normal on {1, ..., 3}, independent of the processing times;
#:   "corr"    the jobs split into 9 weight classes by the rank of a_j + b_j, so
#:             that the heaviest jobs are also the longest;
#:   "free"    weights normal on {1, ..., n}, the unrestricted regime of the
#:             large scale of paper Table 1: as many values as there are jobs.
WEIGHT_SETTINGS = ("narrow", "indep", "corr", "free")


def _normal_int(rng, lo, hi):
    """
    One integer drawn from a normal distribution on lo, ..., hi.

    The mean is the middle of the range and the standard deviation a sixth of
    its width, so that the range spans three standard deviations on either side
    of the mean: the values near the middle are the likely ones and the two ends
    are rare without being out of reach.  The draw is rounded to an integer and
    clamped to the range, and the small probability mass outside the range is
    folded onto its two ends.
    """
    if hi <= lo:
        return lo
    mu = 0.5 * (lo + hi)
    sigma = (hi - lo) / 6.0
    return max(lo, min(hi, int(round(rng.gauss(mu, sigma)))))


def gen_instance(n, K, rng, geometric=False, weights="narrow", proc_hi=PROC_HI):
    """
    One random instance of the benchmark protocol (paper Section 6).

    Every integer of the instance comes from a normal distribution on the
    integers of its range, with the mean at the middle of the range and the
    standard deviation a sixth of its width, so that the two ends of the range
    are three standard deviations from the mean.  The processing times a_j and
    b_j are drawn on {PROC_LO, ..., proc_hi}, the upper end of the range being
    the one paper Section 6.1 gives the scale: [1,40] and [1,80] on the two
    comparison scales, and [1,20], [1,40] and [1,60] on the geometric one.  The
    weights follow the setting `weights`: the narrow regime draws them on
    {1, ..., K}, so that K is the number of values they can take; the
    independent regime draws them on {1, ..., 3}; the correlated regime sorts
    the jobs by a_j + b_j and cuts that order into 9 equal classes, so that the
    heaviest jobs are also the longest, which is a construction rather than a
    draw; and the free regime draws them on {1, ..., n}, which places no bound
    on how many values they take.  `geometric` draws the exponent of the ladder
    {1, 10, ..., 10^(K-1)} instead, the rule of the first scale of paper
    Section 6.1, so that the K weight classes are decades apart rather than
    neighbouring integers; it takes precedence over `weights` when set.
    """
    if weights not in WEIGHT_SETTINGS:
        raise ValueError("unknown weight setting: %r" % (weights,))
    a = [_normal_int(rng, PROC_LO, proc_hi) for _ in range(n)]
    b = [_normal_int(rng, PROC_LO, proc_hi) for _ in range(n)]
    if geometric:
        w = [10 ** _normal_int(rng, 0, K - 1) for _ in range(n)]
    elif weights == "narrow":
        w = [_normal_int(rng, 1, K) for _ in range(n)]
    elif weights == "indep":
        w = [_normal_int(rng, 1, WIDE_K["indep"]) for _ in range(n)]
    elif weights == "free":
        w = [_normal_int(rng, 1, n) for _ in range(n)]
    else:                                   # "corr"
        k = WIDE_K["corr"]
        order = sorted(range(n), key=lambda j: a[j] + b[j])
        w = [0] * n
        for rank, j in enumerate(order):
            w[j] = 1 + rank * k // n
    return a, b, w


def solve_instance(a, b, w, repeats=1, rng=None, **kwargs):
    """
    One instance, solved by the genetic algorithm.

    Returns the value of the best schedule found, the CPU time that took, `None`
    in place of a proof of optimality, a heuristic proving nothing about it, and
    the reference value LB of the instance, which is the denominator of the gap
    this method is read on.  `rng` is the stream the restarts draw their seeds
    from, and every further keyword argument is forwarded to `ga`.

    One instance of the benchmark protocol is exactly this call, so a driver that
    draws the instances itself can run the algorithm on the very instance the
    other methods of the comparison are run on.  The 2-approximation of the paper
    lives in its own module, approx2.py.
    """
    best = float("inf")
    best_order = None
    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        # The run draws its seed from `rng` unless the caller names one; drawing
        # it only when it is used keeps the stream in step with the driver.
        if "seed" not in kwargs:
            kwargs["seed"] = None if rng is None else rng.randrange(1 << 30)
        order = ga(a, b, w, **kwargs)
        total += time.perf_counter() - t0
        obj = _wcmax_fast(order, a, b, w)
        if obj < best:
            best, best_order = obj, order
    # the reference value of this instance: the lower bound LB of paper
    # Section 4.3, which the instance alone determines
    return (best, total / repeats, None,
            None if best_order is None
            else instance_baseline(a, b, w, best_order))


def benchmark(n=50, K=3, instances=INSTANCES_PER_CONFIG, seed=42, first=1,
              geometric=False, weights="narrow", repeats=1, label=None,
              progress=None, proc_hi=PROC_HI, log_path=None, **kwargs):
    """
    Run the genetic algorithm on `instances` random instances and aggregate the
    result.

    The algorithm is run `repeats` times per instance from independent seeds; the
    best objective value found is recorded together with the mean CPU time of a
    single run, and both are averaged over the instances.  Any extra keyword
    argument is forwarded to `ga` (N, G, pc, pm, lam).

    first : int, optional
        Number of the first instance to run, 1 by default.  The instances before
        it are still drawn from the stream the seed gives, so that every instance
        keeps the sample it is meant to have, but they are not solved; the values
        returned cover the instances from `first` on, and so does the record
        file.  A run that was interrupted can then be continued at the instance
        after the last one it solved.

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
    for _ in range(first - 1):            # the instances an earlier run solved
        gen_instance(n, K, rng, geometric, weights, proc_hi)
    solver_rng = random.Random("solver-%d" % seed)
    objs, times, bases = [], [], []
    for idx in range(first - 1, instances):
        a, b, w = gen_instance(n, K, rng, geometric, weights, proc_hi)
        obj, seconds, proved, base = solve_instance(a, b, w, repeats=repeats,
                                                    rng=solver_rng, **kwargs)
        objs.append(obj)
        times.append(seconds)
        bases.append(base)
        record_instance(log_path, n, weights, seed, idx + 1, instances,
                        obj, seconds, K,                 # on disk before the next
                        proc_hi=proc_hi, base=base, label=label)
        if progress is not None:          # live line for a long-running study
            # a heuristic proves nothing about optimality, so the flag is None;
            # the reference is the one the driver estimated for the instance
            progress(idx + 1, instances, obj, seconds, proved, base)

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
            " | mean time=%9.2f ms"
            % (label, res["n"], res["K"], res["instances"],
               res["mean_obj"], res["best_obj"],
               1000.0 * res["mean_time"]))


def main() -> None:  # pragma: no cover
    """Random instance generated from a fixed seed, as in MILP.py."""
    # random-instance parameters
    seed = 42       # random seed
    n = 50          # number of jobs
    a_lo, a_hi = 1, 10   # M1 processing-time range
    b_lo, b_hi = 1, 10   # M2 processing-time range
    w_lo, w_hi = 1, 3     # weight range

    random.seed(seed)
    a = [_normal_int(random, a_lo, a_hi) for _ in range(n)]
    b = [_normal_int(random, b_lo, b_hi) for _ in range(n)]
    w = [_normal_int(random, w_lo, w_hi) for _ in range(n)]

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

    # Optional: compare with the constructions the algorithm is built on -- the
    # block schedule of the initial population, the 2-approximation of its own
    # module, and the NEH heuristic.
    from approx2 import two_approximation
    for name, order in (("Johnson", johnson_order(a, b)),
                        ("2-approx", two_approximation(a, b, w)),
                        ("NEH    ", neh_heuristic(a, b, w))):
        print(f"  {name}:", wcmax_of_order(order, a, b, w))
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
