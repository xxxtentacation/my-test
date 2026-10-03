from datetime import datetime
from pathlib import Path

import random
import statistics
import time

import gurobipy as gp
from gurobipy import GRB


def solve_milp(a, b, w, time_limit=None, threads=None, output_flag=0):
    n = len(a)
    if n != len(b) or n != len(w):
        raise ValueError("a, b, w must have the same length")

    M = 2 * sum(ai + bi for ai, bi in zip(a, b))

    m = gp.Model("F2_WCmax")
    m.setParam("OutputFlag", output_flag)
    if time_limit is not None:
        m.setParam("TimeLimit", time_limit)
    if threads is not None:
        m.setParam("Threads", threads)

    jobs = range(n)            # 0-based job indices
    pos = range(1, n + 1)      # 1-based positions, as in the paper

    # --- decision variables ---
    x = m.addVars(jobs, pos, vtype=GRB.BINARY, name="x")
    C1 = m.addVars(pos, lb=0.0, name="C1")
    C2 = m.addVars(pos, lb=0.0, name="C2")
    C = m.addVars(jobs, lb=0.0, name="C")
    Z = m.addVar(lb=0.0, name="Z")

    # --- objective ---
    m.setObjective(Z, GRB.MINIMIZE)

    # --- assignment (each job to exactly one position, each position one job) ---
    m.addConstrs((x.sum(j, '*') == 1 for j in jobs), name="assign_job")
    m.addConstrs((x.sum('*', k) == 1 for k in pos), name="assign_pos")

    # --- M1 completion times ---
    m.addConstr(C1[1] == gp.quicksum(a[j] * x[j, 1] for j in jobs), name="m1_start")
    for k in range(2, n + 1):
        m.addConstr(
            C1[k] == C1[k - 1] + gp.quicksum(a[j] * x[j, k] for j in jobs),
            name=f"m1_rec_{k}",
        )

    # --- M2 completion times ---
    m.addConstr(
        C2[1] == C1[1] + gp.quicksum(b[j] * x[j, 1] for j in jobs),
        name="m2_start",
    )
    for k in range(2, n + 1):
        m.addConstr(
            C2[k] >= C1[k] + gp.quicksum(b[j] * x[j, k] for j in jobs),
            name=f"m2_a_{k}",
        )
        m.addConstr(
            C2[k] >= C2[k - 1] + gp.quicksum(b[j] * x[j, k] for j in jobs),
            name=f"m2_b_{k}",
        )

    # --- link position-based C2 to job-based C (big-M) ---
    for j in jobs:
        for k in pos:
            m.addConstr(C[j] >= C2[k] - M * (1 - x[j, k]), name=f"link_{j}_{k}")

    # --- maximum weighted completion time ---
    for j in jobs:
        m.addConstr(Z >= w[j] * C[j], name=f"wcmax_{j}")

    m.optimize()

    # --- collect results ---
    res = {
        "status": m.Status,
        "status_str": _status_str(m.Status),
        "obj": None,
        "best_bound": None,
        "order": None,
        "C1": None,
        "C2": None,
        "C": None,
    }

    if m.SolCount > 0:
        res["obj"] = m.ObjVal
        res["best_bound"] = m.ObjBound
        # recover the optimal permutation (job index at each position)
        order = [None] * n
        for j in jobs:
            for k in pos:
                if x[j, k].X > 0.5:
                    order[k - 1] = j
        res["order"] = order
        res["C1"] = [C1[k].X for k in pos]
        res["C2"] = [C2[k].X for k in pos]
        res["C"] = [C[j].X for j in jobs]

    return res


def wcmax_of_order(order, a, b, w):
    n = len(order)
    wc = 0.0
    for j in range(n):
        # completion time of the job at position j (0-based)
        best = 0.0
        for i in range(j + 1):
            sa = sum(a[order[h]] for h in range(i + 1))
            sb = sum(b[order[h]] for h in range(i, j + 1))
            best = max(best, sa + sb)
        wc = max(wc, w[order[j]] * best)
    return wc


def _status_str(status):
    return {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    }.get(status, str(status))


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


def record_instance(path, n, weights, seed, idx, total, obj, seconds,
                    K=None):
    """
    Append one line for a finished instance, and flush it at once.

    The line is written as soon as that instance is done, not when the whole
    benchmark is over, so that a run which is interrupted or killed still holds
    every instance it has already solved.  `path` may be a str or a Path; its
    parent directory is created if it does not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("%s  %-5s n=%-4d %-6s seed=%-7d instance %3d/%-3d  obj %-14s"
                 " time %9.4f s\n"
                 % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), LABEL, n,
                    weight_label(weights, K),
                    seed, idx, total,
                    "none" if obj is None else "%.1f" % obj, seconds or 0.0))


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
    they take.  `geometric`, the ladder {1, 2, 4, ..., 2^(K-1)} used to stress
    the weight-rounding argument of Section 4.4.2, is retained for compatibility
    and takes precedence when set.
    """
    if weights not in WEIGHT_SETTINGS:
        raise ValueError("unknown weight setting: %r" % (weights,))
    a = [rng.randint(PROC_LO, proc_hi) for _ in range(n)]
    b = [rng.randint(PROC_LO, proc_hi) for _ in range(n)]
    if geometric:
        w = [1 << rng.randrange(K) for _ in range(n)]
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
              geometric=False, weights="narrow", repeats=1, time_limit=300.0, threads=1,
              progress=None, proc_hi=PROC_HI, log_path=None):
    """
    Run the MILP model on `instances` random instances and aggregate the result.

    The model is deterministic (`repeats` > 1 changes nothing) and exact, so the
    mean objective it reports is the mean optimum of the sampled instances --
    the reference value z_ref of paper Section 6.1.  On the medium and large
    scales the solver may hit `time_limit`, in which case the instance is
    counted in `failed` when no incumbent was found, and its value is not a
    proven optimum even when one was.

    Returns
    -------
    dict with keys
        mean_obj  : mean over the solved instances of the best objective found
        best_obj  : best objective value over all instances
        mean_time : mean CPU time of a single run, in seconds
        solved    : number of instances with a feasible schedule
        optimal   : number of instances whose optimality was proved
        failed    : number of instances left without a solution
        objs, times : the per-instance values
    """
    # Records go to the file the caller names, or to one of our own tagged with
    # the time this benchmark started.
    if log_path is None:
        log_path = (RECORD_DIR / ("%s_%s.txt"
                                  % (LABEL, datetime.now().strftime("%Y%m%d-%H%M%S"))))
    rng = random.Random(seed)
    objs, times = [], []
    optimal = failed = 0
    for idx in range(instances):
        a, b, w = gen_instance(n, K, rng, geometric, weights, proc_hi)
        best = float("inf")
        total = 0.0
        for _ in range(repeats):
            t0 = time.perf_counter()
            res = solve_milp(a, b, w, time_limit=time_limit, threads=threads)
            total += time.perf_counter() - t0
            if res["obj"] is None:
                failed += 1
                break
            if res["status_str"] == "OPTIMAL":
                optimal += 1
            best = min(best, res["obj"])
        objs.append(best if best < float("inf") else None)
        times.append(total / repeats if best < float("inf") else None)
        record_instance(log_path, n, weights, seed, idx + 1, instances,
                        objs[-1], times[-1], K)   # on disk before the next instance
        if progress is not None:          # live line for a long-running study
            progress(idx + 1, instances, objs[-1], times[-1])

    vals = [o for o in objs if o is not None]
    secs = [t for t in times if t is not None]
    return {
        "n": n, "K": K, "instances": instances,
        "mean_obj": statistics.fmean(vals) if vals else float("nan"),
        "best_obj": min(vals) if vals else float("nan"),
        "mean_time": statistics.fmean(secs) if secs else float("nan"),
        "solved": len(vals),
        "optimal": optimal,
        "failed": failed,
        "objs": objs,
        "times": times,
    }


def report(res, label="MILP"):
    """Single-line summary of a benchmark result."""
    return ("%-8s n=%4d K=%2d | solved=%2d/%2d (optimal %2d) | mean obj=%9.2f"
            " | best obj=%9.2f | mean time=%9.4f s"
            % (label, res["n"], res["K"], res["solved"], res["instances"],
               res["optimal"], res["mean_obj"], res["best_obj"],
               res["mean_time"]))


def main():
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

    res = solve_milp(a, b, w, time_limit=1200, output_flag=1)

    print("status :", res["status_str"])
    if res["order"] is not None:
        print("WCmax  :", res["obj"])
        print("order  :", [j + 1 for j in res["order"]], "(1-based job indices)")
        print("C2     :", res["C2"])
        # cross-check with the closed-form recurrence
        check = wcmax_of_order(res["order"], a, b, w)
        print("verify :", check, "(should equal WCmax)")
    if res["best_bound"] is not None:
        print("best bound :", res["best_bound"])
        if res["obj"] is not None and res["best_bound"] > 0:
            gap = (res["obj"] - res["best_bound"]) / res["best_bound"] * 100
            print("gap        : %.4f%%" % gap)

    # ---- benchmark: INSTANCES_PER_CONFIG instances per configuration -------
    # Only the small scale is run by default; raise n / K for the others, where
    # the solver may hit the time limit.
    print()
    print("Benchmark, %d instances per configuration (paper Section 6.1):"
          % INSTANCES_PER_CONFIG)
    for n in SCALES["small"][0]:
        for K in SCALES["small"][1]:
            print("  " + report(benchmark(n=n, K=K, seed=1000 * n + K,
                                          time_limit=300.0)))


if __name__ == "__main__":
    main()
