# -*- coding: utf-8 -*-
"""
The genetic algorithm alone, on the large scale of the computational study.

This is the large scale of run_large.py with a single method: the instances, the
seeds, the parameters of the genetic algorithm, the per-instance records and the
results file are the ones of that script, and the genetic algorithm is the only
method run on them.  It is meant for the runs that look at the genetic algorithm
by itself -- while its parameters are being calibrated, say -- so that the NEH
rule and the 2-approximation are not run again on every instance of the scale.

Everything is imported from run_large.py, so the settings of this file are the
settings of that one: editing GA_SETTINGS, or the rules below it, in
run_large.py changes what this script runs and the two can never drift apart.
The reference value of every instance is the lower bound LB = max{LB_1, LB_2,
LB_3} of paper Section 4.3, the gap of the method is measured against it, and
the records of the run go to the same directory as those of run_large.py, under
a name that carries the time the run started.

The instances differ from those of run_large.py in two points.  Every integer of
an instance -- both processing times of a job and its weight, and the exponent
of the geometric ladder -- is drawn from a normal distribution on the integers
of its range rather than uniformly, with the mean at the middle of the range and
the standard deviation a sixth of its width, the draw rounded to an integer and
clamped to the range.  And the configurations are shorter: one of the two
processing-time ranges is left out, so that a run covers the range [1,40] and
not [1,80], and the sizes it covers are the seven of every fiftieth job,
100, 150, ..., 400, rather than the sixteen of every twentieth.  The rule of
run_large.py, and the generator of NEH.py under it, are left untouched, so the
other drivers of the study keep the uniform instances, both ranges and all
sixteen sizes they were calibrated on.

The algorithm differs from the one of GA.py in the three points the search of a
weight-blind start needs to close its gap as the instance grows: the elite is
fed back into the population at the end of every generation, the criterion is
one of the quantities the survival ranks, and the mutation moves a whole weight
class instead of a single job.  The variant is written out below, where it
replaces the entry point of GA.py for this script only; the algorithm of the
study, of the drivers and of the paper is left as it is.

Usage:
    python run_large_ga.py                    the 32 configurations, 20 each
    python run_large_ga.py --instances 3      a quick check of the pipeline
    python run_large_ga.py --ga-pop 10 --ga-gen 4 --ga-pm 0.5
    python run_large_ga.py --list             the configurations, then exit

Command line (all of it the command line of run_large.py):
    --instances N    instances per configuration (default 20)
    --seed S         base random seed (default 42)
    --ga-pop N       GA: population size
    --ga-gen N       GA: generations
    --ga-pc P        GA: crossover probability
    --ga-pm P        GA: mutation probability
    --out FILE       results file (default: the one of run_large.py)
    --list           list the methods and configurations, then exit
"""
from __future__ import annotations

import random
import sys
from typing import List, Optional, Sequence

import GA
import NEH
import run_large


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
    return max(lo, min(hi, int(round(rng.gauss(0.5 * (lo + hi), (hi - lo) / 6.0)))))


class _NormalStream:
    """
    The random stream of an instance, with every integer drawn normally.

    The generator of NEH.py asks its stream for `randint(lo, hi)` -- the two
    processing times of a job, and its weight -- and for `randrange(k)`, the
    exponent of the geometric ladder.  This proxy answers both from the normal
    distribution on the integers of the range, and passes every other attribute
    through to the stream it wraps, so that the instance keeps the structure and
    the order of draws of the generator of the study and only the distribution
    of the values changes.
    """

    def __init__(self, rng):
        self._rng = rng

    def randint(self, lo, hi):
        return _normal_int(self._rng, lo, hi)

    def randrange(self, k):
        return _normal_int(self._rng, 0, k - 1)

    def __getattr__(self, name):
        return getattr(self._rng, name)


_NEH_GEN_INSTANCE = NEH.gen_instance


def gen_instance(n, K, rng, **kwargs):
    """
    One instance of the scale, every integer of it drawn normally.

    The generator of NEH.py is called with its stream wrapped, so that the
    instance is drawn the way the study draws it and only the distribution of
    the values differs.
    """
    return _NEH_GEN_INSTANCE(n, K, _NormalStream(rng), **kwargs)


# The drivers of the study draw their instances through NEH.gen_instance, which
# run_large.py imports when it comes to build them; the name is therefore the
# one to replace for this script, and the replacement lives here rather than in
# NEH.py so that the other drivers keep the uniform instances.
NEH.gen_instance = gen_instance

#: The scale this script runs is the large one with a single method, so the
#: default method set of that script is reduced to the genetic algorithm here:
#: what its --list prints, and what a run without --algos would take, are then
#: the genetic algorithm alone.
run_large.METHODS_OF_SCALE = ("ga",)

#: The configurations of this test, in place of the ones of run_large.py.
#:
#: Two things are left out, so that a run of the test is short enough to be
#: repeated while the algorithm is being changed.  The [1,80] range is dropped,
#: which halves the configurations, and the instance sizes are every fiftieth
#: job instead of every twentieth, which leaves seven sizes over the scale
#: instead of sixteen.  The sizes kept are drawn from the same stream as before
#: -- the seed of a configuration depends on its own settings alone -- so the
#: instances of n = 100, 200, 300 and 400 are the ones the longer list gives.
#: Nothing else is shared: run_large.py, and every other driver of the study,
#: still runs both ranges over all sixteen sizes.
_LOWER_RANGE = "[1,40]"
_SIZES = (100, 150, 200, 250, 300, 350, 400)
_WEIGHTS = run_large.WEIGHT_SETTINGS[0]        # the weight setting of the scale

run_large.CONFIGS = tuple((n, _WEIGHTS, _LOWER_RANGE) for n in _SIZES)

#: The generations of this test are the generations of run_large.py, read from
#: that file rather than written out again: the two had a rule each while the
#: count was being calibrated, and the calibration settled on the one the scale
#: now carries, so a second copy here could only drift from it.


# ---------------------------------------------------------------------------
# The genetic algorithm of this test
# ---------------------------------------------------------------------------
#
# The algorithm of GA.py is the one the paper describes.  This script runs a
# variant of it, in which the three places that keep a weight-blind start from
# closing its gap as the instance grows are changed:
#
#   1. the elite -- the best schedule the search has seen on the criterion -- is
#      fed back into the population at the end of every generation, so that the
#      improvements of one generation are inherited by the next rather than only
#      recorded;
#   2. the survival ranks the criterion itself as well, so that the selection
#      presses W C_max and not only the pair (C_max, w_last) that bounds it from
#      below;
#   3. the mutation moves a whole weight class rather than a single job: the jobs
#      of one weight leave the schedule together and come back as one run, which
#      is the move that builds the non-increasing weight order of
#      \Cref{lem:block-johnson} up in as many moves as there are classes instead
#      of as many as there are jobs.
#
# The variant lives here, and replaces the entry point of GA.py, so that the
# algorithm of the study, of the drivers and of the paper is left as it is.
# ---------------------------------------------------------------------------

def _objectives(order: Sequence[int], a: Sequence[float], b: Sequence[float],
                w: Sequence[float]):
    """
    The three quantities the survival of this variant ranks, each to be
    minimized: the makespan C_max, the weight w_last of the job that closes the
    schedule, and the criterion W C_max itself.
    """
    c1 = c2 = 0.0
    wc = 0.0
    for j in order:
        c1 += a[j]
        c2 = max(c1, c2) + b[j]
        wc = max(wc, w[j] * c2)
    return c2, w[order[-1]], wc


def _dominates(p: Sequence[float], q: Sequence[float]) -> bool:
    """True when p is no worse than q on every objective and better on one."""
    return (all(x <= y for x, y in zip(p, q))
            and any(x < y for x, y in zip(p, q)))


def _pareto_fronts(objs: Sequence[tuple]) -> List[List[int]]:
    """Fast non-dominated sorting of the objectives, the best front first."""
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
    """Crowding distance of one front, over the three objectives."""
    dist = {p: 0.0 for p in front}
    if len(front) <= 2:
        return {p: float("inf") for p in front}
    for k in range(3):
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


def _survive(objs: Sequence[tuple], N: int) -> List[int]:
    """
    The N survivors of a pooled set, worst last.

    Whole fronts are kept while they fit; the front that overflows the budget is
    cut by crowding distance.  The last index of the list is therefore the
    individual the ranking leaves worst, which is the one the feedback of the
    elite replaces.
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


def block_move(sigma: Sequence[int], a: Sequence[float], b: Sequence[float],
               w: Sequence[float], rng: random.Random) -> List[int]:
    """
    One move of a whole weight class, which is what a generation varies its
    offspring by.

    The jobs of one weight leave the schedule together, keeping the order the
    schedule holds them in, and the run is put back at the position that lowers
    the criterion most.  The class moved is the one of a job attaining
    W C_max -- the class that closes the criterion, and therefore the one whose
    place in the schedule most needs to change -- and, with probability one half,
    a class drawn at random instead, so that the order of the classes is
    explored and not only repaired.  A class is not a single job: moving one
    puts every job of that weight where the criterion wants it and delays the
    jobs it passes, so the move is proposed whether or not it lowers the
    criterion -- there are orders the criterion rewards that no single accepted
    move reaches -- and it is the survival of the generation, and the elite
    carried through it, that decide what becomes of it.
    """
    n = len(sigma)
    if n < 2:
        return list(sigma)
    if rng.random() < 0.5:
        t_star, _ = GA._critical_job(sigma, a, b, w)
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
        val = GA._wcmax_fast(cand, a, b, w)
        if best_val is None or val < best_val:
            best_val, best = val, cand
    return best


def _evolve(pop: List[List[int]], a: Sequence[float], b: Sequence[float],
            w: Sequence[float], N: int, lam: int, G: int, pc: float, pm: float,
            rng: random.Random):
    """
    The loop of the variant: tournament, PMX, the block move of `block_move`,
    the elite carried into the next generation, and (mu + lambda) survival on
    the three objectives.
    """
    n = len(a)
    objs = [_objectives(s, a, b, w) for s in pop]
    elite = min(pop, key=lambda s: GA._wcmax_fast(s, a, b, w))
    elite_val = GA._wcmax_fast(elite, a, b, w)
    elite = list(elite)

    for _ in range(G):
        rank, crowd, _ = _rank_crowding(objs)

        # Selection: binary tournament on the rank and the crowding distance.
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
                cut1, cut2 = rng.sample(range(n), 2)
                kids = [GA.pmx(p1, p2, cut1, cut2), GA.pmx(p2, p1, cut1, cut2)]
            else:
                kids = [list(p1), list(p2)]
            for kid in kids:
                if len(offspring) < lam:
                    offspring.append(kid)

        # Mutation: a move of a whole weight class, with probability pm.
        for i in range(len(offspring)):
            if rng.random() < pm:
                offspring[i] = block_move(offspring[i], a, b, w, rng)

        # The criterion, tracked on every offspring.
        for s in offspring:
            val = GA._wcmax_fast(s, a, b, w)
            if val < elite_val:
                elite, elite_val = list(s), val

        # Survival: the best N of the parents and the offspring together.
        combined = pop + offspring
        all_objs = objs + [_objectives(s, a, b, w) for s in offspring]
        keep = _survive(all_objs, N)
        pop = [combined[i] for i in keep]
        objs = [all_objs[i] for i in keep]

        # The elite is carried into the next generation: it takes the place of
        # the survivor the ranking leaves worst, and is therefore a parent of
        # the generation that follows.
        if elite not in pop:
            pop[-1] = elite
            objs[-1] = _objectives(elite, a, b, w)

    return pop, objs, elite, elite_val


def ga(a: Sequence[float], b: Sequence[float], w: Sequence[float],
       N: Optional[int] = None, G: int = 1, pc: float = 0.9, pm: float = 0.1,
       k: Optional[int] = None, lam: Optional[int] = None,
       seed: Optional[int] = None) -> List[int]:
    """
    The variant of Algorithm alg:ga this script runs, with the arguments of the
    algorithm of the study: the initial population, the descent of Step 3, the
    rules for N and lambda and the pmx of GA.py are the ones of that module.
    """
    n = len(a)
    if n != len(b) or n != len(w):
        raise ValueError("a, b, w must have the same length")
    if n == 0:
        return []
    if N is None:
        N = GA.population_of(n)
    if lam is None:
        lam = GA.offspring_of(n)
    lam = max(1, int(lam))

    rng = random.Random(seed)

    pop = GA._initial_population(a, b, w, N, rng)
    pop, _objs, sigma_star, _best = _evolve(pop, a, b, w, N, lam, G, pc, pm, rng)

    width = GA.descent_k(n) if k is None else max(1, min(n, k))
    sigma_star = GA.insertion_descent(sigma_star, a, b, w, width)
    return sigma_star


# The drivers of the study solve an instance through GA.solve_instance, which
# calls the `ga` of that module by name; replacing the name here is therefore
# what makes this script run the variant, and it lives here rather than in
# GA.py so that the algorithm of the study is left as the paper describes it.
GA.ga = ga


def main(argv=None):
    """
    Run the genetic algorithm on the large scale, and nothing else.

    The command line is handed to run_large.py with `--algos ga` appended, so
    that the method of the run is the genetic algorithm whatever else is asked
    for, while every other argument keeps the meaning it has there.
    """
    tail = list(sys.argv[1:] if argv is None else argv)
    print("instances : every integer drawn from a normal distribution "
          "on the integers of its range (this script)", flush=True)
    print("algorithm : the GA of GA.py with the elite fed back into the "
          "population, the criterion ranked in the survival, and a mutation "
          "that moves a whole weight class (this script)", flush=True)
    print("configs   : the %s range and the sizes %s, both shorter than the "
          "study's (this script)"
          % (_LOWER_RANGE, ", ".join(str(n) for n in _SIZES)), flush=True)
    return run_large.main(tail + ["--algos", "ga"])


if __name__ == "__main__":
    main()
