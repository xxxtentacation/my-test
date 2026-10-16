# -*- coding: utf-8 -*-
"""
Genetic Algorithm (GA) for F2 || WCmax (two-machine flow shop, minimizing the
maximum weighted completion time), implementing Section 5.3 of the paper
(paper_framework.tex): Algorithm (alg:pmx) partially mapped crossover, Algorithm
(alg:cm) block mutation, and Algorithm (alg:ga) the genetic algorithm.

Chromosome.  A permutation sigma = (sigma(1), ..., sigma(n)) of the job indices;
the gene at position t is the job scheduled t-th.  Permutation encoding is used
because every permutation is a feasible schedule, so crossover and mutation need
no repair operator.  Decoding evaluates

    WCmax(sigma) = max_t w_{sigma(t)} C_{sigma(t)}

in O(n) time with a single forward pass over the two machines (paper
eq:completion).

Objectives.  A schedule is judged by the triple

    ( C_max(sigma), w_last(sigma), WCmax(sigma) ),

its makespan, the weight of the job that closes it, and the criterion itself,
all to be minimized.  The first two bound the criterion from below,
WCmax >= w_last * C_max, and carry its scale; they are not the criterion, so it
is ranked with them rather than tracked beside them, and the population is
carried forward by the non-dominated sorting and crowding distance of NSGA-II
rather than by a single scalar fitness.  The best schedule the search sees on
the criterion is also carried into the next generation, so that an improvement
is inherited by the generation that follows and not merely recorded.

Initial population.  Johnson's schedule and N - 1 neighbours of it.  Johnson's
rule orders the jobs by their two processing times alone, so the population
starts from the order that is shortest on the two machines and carries nothing
of the weights; the criterion is brought in by the mutation below.

Operators.
    PMX (Algorithm alg:pmx) copies a contiguous segment of one parent and fills
    the remaining positions from the other parent, resolving a gene that the
    segment already holds through the mapping the segment induces, so that the
    child is always a valid permutation.  Swapping the roles of the two parents
    yields the second child.
    Block mutation (Algorithm alg:cm) moves a block of the schedule: the jobs of
    one weight leave it together, keeping the order they hold, and come back as
    one run at the position that serves the criterion best.  The block is the one
    a job attaining WCmax belongs to, and, with probability one half, one drawn
    at random.  It is the move that holds the block structure of
    Lemma (lem:block-johnson): an optimal schedule carries one block per weight,
    in non-increasing order of it, and a move of a whole block puts the jobs of a
    weight where the criterion wants them all at once, which a move of a single
    job cannot.

Local search (Algorithm alg:ga, Step 3).  The jobs carrying the largest weighted
completion times -- descent_k(n) of them, one job in ten of the instance, the
ones that decide the criterion -- are reinserted, each at the position that
lowers WCmax most, until no such insertion improves.

Generations.  The algorithm is run for the number of generations the caller
names.  The initial population carries nothing of the weights, so the
generations are spent building the criterion into the schedule rather than
moving an order that already has it, and the local search of Step 3 refines what
they leave.

Note on p_m.  Algorithm alg:ga applies Algorithm alg:cm to each offspring with
probability p_m, and nothing else in the algorithm draws on p_m.
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


def block_mutation(sigma: Sequence[int], a: Sequence[float],
                   b: Sequence[float], w: Sequence[float],
                   rng: random.Random) -> List[int]:
    """
    Algorithm alg:cm: block mutation.

    Step 1  take the jobs of one weight out of the schedule, keeping the order
            they hold in it, and put the run back at the position that lowers
            W C_max most;
    Step 2  the weight is the one of a job attaining W C_max, and, with
            probability one half, one drawn at random instead.

    A weight is not a single job: it is a block of the schedule, and
    \\Cref{lem:block-johnson} says that an optimal schedule holds one such block
    per weight, in non-increasing order of it.  Moving a block therefore does
    what a single job cannot: it puts every job of that weight where the
    criterion wants it at once, and it is the move that builds the block order
    up in as many moves as there are weights rather than as many as there are
    jobs.  Since moving a block forward delays the jobs it passes, the move is
    proposed whether or not it lowers the criterion -- there are orders the
    criterion rewards that no single accepted move reaches -- and the survival
    of the generation decides what becomes of it.
    """
    n = len(sigma)
    if n < 2:
        return list(sigma)

    # Step 1: the block, and where it goes.
    if rng.random() < 0.5:
        t_star, _ = _critical_job(sigma, a, b, w)
        weight = w[sigma[t_star]]
    else:
        weight = w[rng.choice(sigma)]
    members = [j for j in sigma if w[j] == weight]
    if len(members) == n:
        return list(sigma)                # the whole schedule: nothing to move
    rest = [j for j in sigma if w[j] != weight]
    best_val, best = None, list(sigma)
    for r in range(len(rest) + 1):
        cand = rest[:r] + members + rest[r:]
        val = _wcmax_fast(cand, a, b, w)
        if best_val is None or val < best_val:
            best_val, best = val, cand
    return best


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
# A schedule carries three objectives, all to be minimized: its makespan C_max,
# the weight w_last of the job that closes it, and the criterion WCmax itself.
# The first two bound the criterion from below, WCmax = max_j w_j C_j >=
# w_last * C_max, and carry its scale: it is a weighted completion, so a search
# that only pressed the makespan would leave the weights out of the account.
# They are not the criterion, however -- a schedule can end early and light and
# still be a poor one -- so the criterion is ranked with them rather than
# tracked beside them: what the search selects for is what it is measured on.
# ---------------------------------------------------------------------------

def objectives(order: Sequence[int], a: Sequence[float], b: Sequence[float],
               w: Sequence[float]):
    """
    The triple (C_max, w_last, WCmax) the survival ranks.

    All three are read in the one forward pass over the two machines, so
    ranking the criterion as well costs no more than the two bounds alone.
    """
    c1 = c2 = 0.0
    wc = 0.0
    for j in order:
        c1 += a[j]
        c2 = max(c1, c2) + b[j]
        wc = max(wc, w[j] * c2)
    return c2, w[order[-1]], wc


def _dominates(p, q) -> bool:
    """True when p dominates q: no worse on every objective, better on one."""
    return (all(x <= y for x, y in zip(p, q))
            and any(x < y for x, y in zip(p, q)))


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
    for k in range(len(objs[front[0]])):
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


def _johnson_population(a: Sequence[float], b: Sequence[float], N: int,
                        rng: random.Random) -> List[List[int]]:
    """
    P0: Johnson's schedule and N - 1 neighbours of it.

    The first individual is the schedule of Johnson's rule, which minimizes the
    makespan, and the others are copies of it with a few neighbouring jobs
    exchanged, the number of exchanges being drawn between one and
    max(1, n // 4).  Johnson's rule orders the jobs by their processing times
    alone, so the population starts from the order that is shortest on the two
    machines together and carries no information about the weights, which the
    mutation below then brings in one job at a time.
    """
    sigma = johnson_order(a, b)
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
    return _johnson_population(a, b, N, rng)



def _evolve(pop: List[List[int]], a: Sequence[float], b: Sequence[float],
            w: Sequence[float], N: int, lam: int, G: int, pc: float, pm: float,
            rng: random.Random):
    """
    The NSGA-II loop: tournament on (rank, crowding), PMX, the block mutation,
    and (mu + lambda) survival by non-dominated sorting on the three objectives.

    A generation produces `lam` offspring from a population of N parents, so the
    survival ranks N + lam schedules and keeps the best N of them.  More
    offspring than parents is what gives one generation several recombinations
    and mutations of the same parents to choose from, and the cost of the
    generation is set by lam rather than by N.

    Returns the population reached after G generations, its objectives, and the
    best schedule on the criterion met along the way.  The criterion is one of
    the three objectives the survival ranks, so a good schedule is under
    pressure to survive; the elite is tracked on the criterion alone as well,
    and is carried into the next generation, so that nothing the search finds is
    lost when the other two objectives of that schedule are poor.
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
        while len(offspring) < lam:
            p1 = pool[rng.randrange(len(pool))]
            p2 = pool[rng.randrange(len(pool))]
            if n >= 2 and rng.random() < pc:
                # two distinct cut points, so that p1 < p2 as alg:pmx states
                cut1, cut2 = rng.sample(range(n), 2)
                kids = [pmx(p1, p2, cut1, cut2), pmx(p2, p1, cut1, cut2)]
            else:
                kids = [list(p1), list(p2)]
            for kid in kids:
                if len(offspring) < lam:
                    offspring.append(kid)

        # Mutation: Algorithm alg:cm on each offspring with probability pm.
        for k in range(len(offspring)):
            if rng.random() < pm:
                offspring[k] = block_mutation(offspring[k], a, b, w, rng)

        # The criterion, tracked on every offspring.
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

        # The elite is carried into the next generation: it takes the place of
        # the survivor the ranking leaves worst, and is therefore a parent of
        # the generation that follows.  Without this the improvements the search
        # finds would be recorded but never inherited, since what the survival
        # ranks is the criterion together with the two bounds and not the best
        # schedule of the generation.
        if elite not in pop:
            pop[-1] = elite
            objs[-1] = objectives(elite, a, b, w)
    return pop, objs, elite, elite_val


def ga(a: Sequence[float], b: Sequence[float], w: Sequence[float],
       N: Optional[int] = None, G: int = 1, pc: float = 0.9, pm: float = 0.1,
       k: Optional[int] = None,
       lam: Optional[int] = None, seed: Optional[int] = None) -> List[int]:
    """
    Genetic algorithm for F2 || WCmax (Algorithm alg:ga).

    The search is three-objective: an individual carries its makespan C_max, the
    weight w_last of its last job and the criterion WCmax itself, and the
    population is carried forward by the non-dominated sorting and crowding
    distance of NSGA-II.  The first two bound the criterion from below,
    WCmax >= w_last * C_max, and ranking the criterion with them is what makes
    the selection press what the method is measured on; the best schedule the
    search sees is also carried into the next generation, so that what one
    generation finds is inherited by the next rather than only recorded.  The
    schedule the algorithm reports is that best schedule, refined by the
    insertion descent of Step 3.

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
    pm      : mutation probability.  Algorithm alg:cm, the move of a block of
              one weight, is applied to each offspring with probability pm.
    k       : how many jobs the local search of Step 3 reinserts, the ones with
              the largest weighted completion times.  A caller that sets the
              parameters of a scale names one of its own -- the small and the
              large scale of the study do -- and a caller that names none gets
              the default rule descent_k(n).
    lam     : offspring per generation, the lambda of the (mu + lambda) survival.
              The generation runs the crossover and the mutation until lam
              children have been produced, and the survival keeps the best N of
              the N + lam schedules it then holds.  A caller that names none gets
              the default rule offspring_of(n), which rises linearly with the
              instance, and the scales of the study name their own.
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
        lam = offspring_of(n)
    lam = max(1, int(lam))

    rng = random.Random(seed)

    # ---- Step 1: initialization -------------------------------------------
    # The initial population: Johnson's schedule of the instance and its
    # neighbours.  Johnson's rule reads the two processing times alone, so the
    # criterion has to be built into the schedule by the operators below.
    pop = _initial_population(a, b, w, N, rng)

    # ---- Step 2: evolution -------------------------------------------------
    # G generations, NSGA-II on (C_max, w_last): fronts and crowding carry the
    # selection, while the best schedule on the criterion itself is tracked
    # alongside them.
    pop, _objs, sigma_star, best_val = _evolve(pop, a, b, w, N, lam, G, pc, pm, rng)

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


#: How many offspring one generation produces, by the number of jobs of the
#: instance: linear in n, with a floor and a cap.  Every child is one pass of the
#: crossover and, with probability p_m, one pass of the mutation, so this is how
#: many recombinations and mutations a generation gets before the survival sorts
#: them, and the cost of a generation is set by it rather than by the population.
#: The drivers of the study name their own rule; this is the default of a caller
#: that names none.
OFFSPRING_MIN, OFFSPRING_RATE, OFFSPRING_MAX = 20, 1.0, 120


def offspring_of(n: int) -> int:
    """The offspring one generation produces on an instance of n jobs."""
    return min(OFFSPRING_MAX, OFFSPRING_MIN + int(OFFSPRING_RATE * n))


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
