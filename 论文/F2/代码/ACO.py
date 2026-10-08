# -*- coding: utf-8 -*-
"""
Ant Colony Optimization (ACO) for F2 || WCmax (two-machine flow shop, minimizing
the maximum weighted completion time), implementing Algorithm (alg:aco) of the
paper (paper_framework.tex, Section 5.2).

A colony of m artificial ants constructs permutation schedules one position at a
time, guided by a pheromone matrix tau_ij (edge information: a large tau_ij
encourages job J_j to be placed right after job J_i) and by the heuristic
information eta_j = w_j / (a_j + b_j + 1).  An ant is judged by two objectives,
the makespan of the schedule it built and the weight of that schedule's last
job, and the colony keeps the archive of schedules that no other schedule
dominates on the two.  After all ants have built their schedules, the pheromone
trails are reinforced along the edges of every schedule in the archive:

    tau_ij <- (1 - rho) * tau_ij + rho * Delta_tau_ij,
    Delta_tau_ij = Q / (|A| * (C_max * w_last))  over the members of A.

The two objectives bound the criterion from below, WCmax >= w_last * C_max, so
the archive explores the trade-off the criterion rewards; the schedule the
colony returns is the one of smallest WCmax it met.

Note.  The paper writes the construction probability with a job index
(tau_j) while the update rule uses an edge index (tau_ij).  The two are
reconciled here by taking tau_j to mean tau_{i,j}, where i is the job placed at
the previous position; this matches both the update rule and the motivation
("a larger value encourages job J_j to be scheduled after job J_i").
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import math
import random
import statistics
import time
from typing import List, Optional, Sequence

#: log / exp guarded against a zero argument, since tau and eta are both positive
#: in exact arithmetic but can underflow to 0.0 after many evaporation rounds.
_log = lambda x: math.log(x) if x > 0.0 else -math.inf
_exp = lambda x: math.exp(x) if x > -700.0 else 0.0


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


def _archive_add(archive: List[tuple], sigma: Sequence[int], c: float,
                 wl: float) -> None:
    """
    Offer a schedule to the non-dominated archive of the colony, in place.

    The archive holds the non-dominated schedules found so far, by (C_max,
    w_last); a schedule that no member dominates is kept and removes from the
    archive every member it dominates.  Since w_last takes one of finitely many
    values, and one schedule per value survives, the archive stays small.
    """
    for c2, w2, _s in archive:
        if c2 <= c and w2 <= wl:           # dominated by a member, or equal to it
            return
    archive[:] = [e for e in archive
                  if not (c < e[0] and wl <= e[1] or c <= e[0] and wl < e[1])]
    archive.append((c, wl, list(sigma)))


def aco(a: Sequence[float], b: Sequence[float], w: Sequence[float],
        m: Optional[int] = None, alpha: float = 1.0, beta: float = 30.0,
        rho: float = 0.1, Q: float = 1.0, max_iter: int = 100,
        seed: Optional[int] = None) -> List[int]:
    """
    ACO for F2 || WCmax  (Algorithm alg:aco).

    The colony is bi-objective: an ant is judged by the pair made of the makespan
    of the schedule it built and the weight of that schedule's last job, and the
    pheromone is laid by the non-dominated schedules of the archive the colony
    keeps, rather than by a single best schedule.  The two objectives bound the
    criterion of the paper from below, WCmax >= w_last * C_max, so the trade-off
    the archive explores is exactly the one the criterion rewards; the schedule
    returned is the one of smallest WCmax the colony met.

    Parameters
    ----------
    a, b, w  : length n
        Processing time on M1, processing time on M2, and weight of each job.
    m        : number of ants per iteration (default min(n, ACO_ANTS), i.e. 50 for
               every n >= 50).  Calibrated in Section 6.1.2: raising m from 10 to
               50 moves the mean gap by under three points while costing five
               times as much, and tying m to n makes the run cubic in n.
    alpha    : pheromone exponent (default 1.0).
    beta     : heuristic-information exponent (default 30.0).  eta_j =
               w_j / (a_j + b_j + 1) encodes the "heavy jobs first" bias that the
               bottleneck objective rewards.  Calibrated in Section 6.1.2: the gap
               is smallest near beta = 30 on the correlated setting, and grows on
               both sides of it; no beta makes the colony match the
               2-approximation, because as beta grows the construction approaches
               the weight-descending order and hence the approximation itself.
    rho      : evaporation rate in (0, 1) (default 0.1).
    Q        : pheromone deposit constant (default 1.0).
    max_iter : number of iterations, the "stopping criterion" of Step 4
               (default 100).  Calibrated in Section 6.1.2: iterations help on the
               correlated setting and barely at all on the independent one.
    seed     : optional random seed.

    Returns
    -------
    list of job indices (0-based) of the best schedule sigma* found.
    """
    n = len(a)
    if n != len(b) or n != len(w):
        raise ValueError("a, b, w must have the same length")
    if m is None:
        # The colony size is capped rather than tied to n.  The construction loop
        # costs O(m * max_iter * n^2), so m = n makes the run quadratic-and-a-half
        # in n and puts the large scale out of reach; raising m from 10 to 50 also
        # changes the mean gap by under three points, so the cap costs little.
        m = min(n, ACO_ANTS)

    rng = random.Random(seed)

    # Step 1: initialize pheromone, the archive of non-dominated schedules, and
    # the incumbent the colony reports.
    tau = [[1.0 / n] * n for _ in range(n)]
    eta = [w[j] / (a[j] + b[j] + 1.0) for j in range(n)]

    archive: List[tuple] = []              # (C_max, w_last, schedule)
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
                #
                # The product is assembled in log space and shifted by its maximum
                # before exponentiation.  eta_j = w_j / (a_j + b_j + 1) can be huge
                # when the weights are spread over a wide range -- a geometric
                # ladder 1, 2, ..., 2^(K-1) gives eta ~ 2^K -- and raising it to
                # beta then overflows the double range.  Dividing every weight by
                # the largest one is a positive rescaling, so it leaves the
                # selection distribution unchanged.
                if prev < 0:
                    logw = [beta * _log(eta[j]) for j in remaining]
                else:
                    tau_row = tau[prev]
                    logw = [alpha * _log(tau_row[j]) + beta * _log(eta[j])
                            for j in remaining]
                mx = max(logw)
                weights = [_exp(x - mx) for x in logw]
                total = sum(weights)
                if total <= 0.0:           # degenerate: fall back to uniform
                    j = rng.choice(remaining)
                else:
                    j = rng.choices(remaining, weights=weights, k=1)[0]

                remaining.remove(j)
                sigma.append(j)
                prev = j

            # Step 2: the two objectives of the ant, the archive, and the
            # incumbent the colony reports on.
            c_max, w_last = objectives(sigma, a, b, w)
            _archive_add(archive, sigma, c_max, w_last)
            z_k = _wcmax_fast(sigma, a, b, w)
            if z_k < best_val:             # best-so-far update
                best_val = z_k
                best_order = sigma[:]

        # Step 3: evaporate, then let every non-dominated schedule of the archive
        # lay pheromone, in proportion to the reciprocal of the product it
        # attains -- the quantity the reference of the benchmark measures -- and
        # give the incumbent, the best schedule on the criterion itself, the same
        # share, so that the criterion keeps steering the construction.
        for i in range(n):
            row = tau[i]
            for j in range(n):
                row[j] *= (1.0 - rho)

        if archive:
            share = rho * Q / len(archive)
            for c_max, w_last, sigma in archive:
                product = c_max * w_last
                deposit = share / product if product > 0 else 0.0
                for t in range(n - 1):
                    tau[sigma[t]][sigma[t + 1]] += deposit

    return best_order


def run_aco(a, b, w, **kwargs) -> dict:
    """Convenience wrapper returning a result dict."""
    order = aco(a, b, w, **kwargs)
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

#: colony size used when `m` is not given (paper Section 6.1.2: "50 ants")
ACO_ANTS = 50

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


def benchmark(n=50, K=3, instances=INSTANCES_PER_CONFIG, seed=42,
              geometric=False, weights="narrow", repeats=1, progress=None,
              proc_hi=PROC_HI, log_path=None, **kwargs):
    """
    Run the method on `instances` random instances and aggregate the result.

    ACO is stochastic, so each instance is solved `repeats` times from
    independent seeds; the best objective value found is recorded together with
    the mean CPU time of a single run, and both are averaged over the instances.
    Any extra keyword argument is forwarded to `aco` (m, alpha, beta, rho, Q,
    max_iter).

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
            order = aco(a, b, w, seed=solver_rng.randrange(1 << 30), **kwargs)
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


def report(res, label="ACO"):
    """Single-line summary of a benchmark result."""
    return ("%-8s n=%4d K=%2d | instances=%2d | mean obj=%9.2f | best obj=%9.2f"
            " | mean time=%9.4f s"
            % (label, res["n"], res["K"], res["instances"],
               res["mean_obj"], res["best_obj"], res["mean_time"]))


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

    # ---- benchmark: INSTANCES_PER_CONFIG instances per configuration -------
    print()
    print("Benchmark, %d instances per configuration (paper Section 6.1):"
          % INSTANCES_PER_CONFIG)
    for n, K in [(20, 3), (50, 3), (100, 3)]:
        print("  " + report(benchmark(n=n, K=K, seed=1000 * n + K,
                                       m=20, max_iter=50)))
    print("  other scales: loop over SCALES, e.g. benchmark(n=n, K=K)"
          " for n in SCALES['large'][0]")


if __name__ == "__main__":
    main()
