# -*- coding: utf-8 -*-
"""
Exact MILP solver for F2 || WCmax (two-machine flow shop, minimizing the maximum
weighted completion time), implemented with Gurobi.

This file implements the mixed-integer linear program of the paper
(paper_framework.tex, Section 2 "Preliminaries").

Decision variables
------------------
  x[j,k] in {0,1}   : job J_j is assigned to position k (1-based positions)
  C1[k]             : completion time of the job at position k on M1
  C2[k]             : completion time of the job at position k on M2
  C[j]              : completion time of job J_j on M2
  Z                 : the maximum weighted completion time WCmax

Model
-----
  min  Z
  s.t. sum_k x[j,k] = 1                    (milp:assign1)
       sum_j x[j,k] = 1                    (milp:assign2)
       C1[1] = sum_j a_j x[j,1]            (milp:m1_start)
       C1[k] = C1[k-1] + sum_j a_j x[j,k]  (milp:m1_rec, k>=2)
       C2[1] = C1[1] + sum_j b_j x[j,1]    (milp:m2_start)
       C2[k] >= C1[k] + sum_j b_j x[j,k]   (milp:m2_a, k>=2)
       C2[k] >= C2[k-1] + sum_j b_j x[j,k] (milp:m2_b, k>=2)
       C[j] >= C2[k] - M(1 - x[j,k])       (milp:link)
       Z >= w_j C[j]                       (milp:wcmax)
       x binary; C1, C2, C >= 0            (milp:x, milp:dom)
where M = 2 * sum_j (a_j + b_j) is a sufficiently large constant.
"""

import gurobipy as gp
from gurobipy import GRB


def solve_milp(a, b, w, time_limit=None, threads=None, output_flag=0):
    """
    Solve F2 || WCmax exactly via the MILP model.

    Parameters
    ----------
    a, b, w : list[float], length n
        Processing time on M1, processing time on M2, and weight of each job.
    time_limit : float or None
        Solver time limit in seconds (None means no limit).
    threads : int or None
        Number of threads (None means Gurobi default).
    output_flag : int
        Gurobi OutputFlag (0 = silent, 1 = verbose).

    Returns
    -------
    dict with keys
        status      : Gurobi status code (gurobipy.GRB.*)
        status_str  : readable status string
        obj         : optimal WCmax value (or incumbent if not proven optimal)
        best_bound  : dual bound (only meaningful if not proven optimal)
        order       : list of job indices (0-based) in the optimal sequence,
                      or None if no feasible solution was found
        C1, C2, C   : completion-time arrays (position-based / job-based)
    """
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

    if m.Status in (GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.USER_OBJ_LIMIT, GRB.NODE_LIMIT):
        res["best_bound"] = m.ObjBound

    return res


def wcmax_of_order(order, a, b, w):
    """
    Compute the WCmax of a given permutation via the standard flow-shop recurrence
    (paper eq:completion):
        C_{pi(j)} = max_{1<=i<=j} { sum_{h=1..i} a_{pi(h)} + sum_{h=i..j} b_{pi(h)} },
        WCmax = max_j w_{pi(j)} C_{pi(j)}.
    """
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


def main():
    """Small self-contained example."""
    # jobs: (a_j, b_j, w_j)
    a = [1, 100, 3, 2]
    b = [100, 1, 2, 3]
    w = [10, 11, 5, 7]

    res = solve_milp(a, b, w, time_limit=60, output_flag=1)

    print("status :", res["status_str"])
    if res["order"] is not None:
        print("WCmax  :", res["obj"])
        print("order  :", [j + 1 for j in res["order"]], "(1-based job indices)")
        print("C2     :", res["C2"])
        # cross-check with the closed-form recurrence
        check = wcmax_of_order(res["order"], a, b, w)
        print("verify :", check, "(should equal WCmax)")


if __name__ == "__main__":
    main()
