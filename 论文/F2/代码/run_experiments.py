# -*- coding: utf-8 -*-
"""
Master driver for the computational study of paper Section 6 (Experiments).

It runs a selected subset of the algorithms implemented in this folder on the
configurations of one instance scale, and records the aggregated results -- the
mean objective value and the mean CPU time per configuration, as specified in
paper Section 6.1.3 -- in a single results file.  By default that file is
written next to this folder, i.e. in the same directory as the algorithm folder::

    论文/F2/experiment_results.txt          <- results file (default)
    论文/F2/代码/run_experiments.py          <- this script
    论文/F2/代码/{MILP,DP,NEH,ACO,GA}.py     <- the algorithms

Every algorithm is driven through the `benchmark()` harness of its own module,
so the protocol is the same whichever subset is selected: each configuration is
replicated over `--instances` random instances, and for every instance the best
objective value found and the CPU time are recorded and then averaged.

Usage
-----
    python run_experiments.py --list
    python run_experiments.py --scale medium --algos neh ga approx2
    python run_experiments.py --scale small --algos milp dp neh --time-limit 300
    python run_experiments.py --scale medium --algos ga --geometric --repeats 5
    python run_experiments.py --scale medium --algos neh ga --out ../res.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: default results file: same directory as the algorithm folder (this folder)
DEFAULT_OUT = HERE.parent / "experiment_results.txt"

#: instance scales of paper Table 1: name -> (values of n, values of K)
SCALES = {
    "small":  ([6, 8, 10, 12], [2, 3]),
    "medium": ([20, 50, 100], [3, 5]),
    "large":  ([200, 500, 1000], [5, 10, 20]),
}

#: scales on which each method is meaningful; "all" means every scale
APPLICABLE = {
    "milp":    "small",
    "dp":      "small",
    "approx2": "all",
    "johnson": "all",
    "neh":     "all",
    "aco":     "all",
    "ga":      "all",
}


# ---------------------------------------------------------------------------
# runners: one thin adapter per algorithm, each calling that module's benchmark
# ---------------------------------------------------------------------------

def _run_milp(n, K, instances, seed, geometric, repeats, opts):
    from MILP import benchmark
    return benchmark(n=n, K=K, instances=instances, seed=seed,
                     geometric=geometric, repeats=repeats,
                     time_limit=opts.time_limit)


def _run_dp(n, K, instances, seed, geometric, repeats, opts):
    from DP import benchmark
    return benchmark(n=n, K=K, instances=instances, seed=seed,
                     geometric=geometric, repeats=repeats,
                     time_limit=opts.time_limit)


def _run_neh(n, K, instances, seed, geometric, repeats, opts):
    from NEH import benchmark
    return benchmark(n=n, K=K, instances=instances, seed=seed,
                     geometric=geometric, repeats=repeats)


def _run_aco(n, K, instances, seed, geometric, repeats, opts):
    from ACO import benchmark
    kwargs = {}
    if opts.aco_ants is not None:
        kwargs["m"] = opts.aco_ants
    if opts.aco_iter is not None:
        kwargs["max_iter"] = opts.aco_iter
    if opts.aco_rho is not None:
        kwargs["rho"] = opts.aco_rho
    return benchmark(n=n, K=K, instances=instances, seed=seed,
                     geometric=geometric, repeats=repeats, **kwargs)


def _run_ga(n, K, instances, seed, geometric, repeats, opts):
    from GA import benchmark
    kwargs = {}
    if opts.ga_pop is not None:
        kwargs["N"] = opts.ga_pop
    if opts.ga_gen is not None:
        kwargs["G"] = opts.ga_gen
    return benchmark(n=n, K=K, instances=instances, seed=seed,
                     geometric=geometric, repeats=repeats, **kwargs)


def _run_johnson(n, K, instances, seed, geometric, repeats, opts):
    from GA import benchmark
    return benchmark(n=n, K=K, instances=instances, seed=seed,
                     geometric=geometric, repeats=repeats, method="johnson")


def _run_approx2(n, K, instances, seed, geometric, repeats, opts):
    from GA import benchmark
    return benchmark(n=n, K=K, instances=instances, seed=seed,
                     geometric=geometric, repeats=repeats, method="2-approx")


#: output order of the methods, and the runner of each
METHODS = {
    "milp":    ("MILP (exact)",      _run_milp),
    "dp":      ("DP (exact)",        _run_dp),
    "approx2": ("2-approximation",   _run_approx2),
    "johnson": ("Johnson's rule",    _run_johnson),
    "neh":     ("NEH",               _run_neh),
    "aco":     ("Ant colony",        _run_aco),
    "ga":      ("Genetic algorithm", _run_ga),
}
DEFAULT_METHODS = ["approx2", "johnson", "neh", "aco", "ga"]


# ---------------------------------------------------------------------------
# experiment driver
# ---------------------------------------------------------------------------

def run_experiments(scale, methods, args, stream=sys.stdout):
    """
    Run every selected method on every configuration of `scale`.

    Returns a list of records, one per (method, n, K):
        algo, n, K, instances, mean_obj, best_obj, mean_time, solved, failed
    """
    ns, Ks = SCALES[scale]
    records = []
    total = len(ns) * len(Ks) * len(methods)
    done = 0

    for n in ns:
        for K in Ks:
            for name in methods:
                label, runner = METHODS[name]
                seed = args.seed + 1000 * n + K
                t0 = time.perf_counter()
                try:
                    res = runner(n, K, args.instances, seed, args.geometric,
                                 args.repeats, args)
                except Exception as exc:                       # noqa: BLE001
                    print("  %-8s n=%4d K=%2d  FAILED: %s" % (name, n, K, exc),
                          file=stream)
                    done += 1
                    continue
                elapsed = time.perf_counter() - t0

                rec = {
                    "algo": name,
                    "n": n,
                    "K": K,
                    "instances": res["instances"],
                    "mean_obj": res["mean_obj"],
                    "best_obj": res["best_obj"],
                    "mean_time": res["mean_time"],
                    "solved": res.get("solved", res["instances"]),
                    "failed": res.get("failed", 0),
                    "wall": elapsed,
                }
                records.append(rec)
                done += 1
                print("  [%2d/%2d] %-8s n=%4d K=%2d | mean obj=%9.2f"
                      " | best obj=%9.2f | mean time=%9.4f s"
                      % (done, total, name, n, K, rec["mean_obj"],
                         rec["best_obj"], rec["mean_time"]), file=stream)

    return records


def summarise(records):
    """Mean objective and mean CPU time of each method, over all configurations."""
    by_algo = {}
    for rec in records:
        by_algo.setdefault(rec["algo"], []).append(rec)
    out = []
    for name in METHODS:
        recs = by_algo.get(name)
        if not recs:
            continue
        out.append({
            "algo": name,
            "label": METHODS[name][0],
            "configurations": len(recs),
            "mean_obj": statistics.fmean(r["mean_obj"] for r in recs),
            "mean_time": statistics.fmean(r["mean_time"] for r in recs),
        })
    return out


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

def _header(scale, methods, args):
    return [
        "F2 || WCmax -- computational study (paper Section 6)",
        "generated     : %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scale         : %s  (n in %s, K in %s)"
        % (scale, SCALES[scale][0], SCALES[scale][1]),
        "methods       : %s" % ", ".join(methods),
        "instances/cfg : %d" % args.instances,
        "repeats       : %d" % args.repeats,
        "weight setting: %s" % ("geometric ladder" if args.geometric
                                else "uniform {1..K}"),
        "seed          : %d" % args.seed,
        "time limit    : %s s" % (args.time_limit if args.time_limit else "none"),
    ]


def write_text(path, records, summary, scale, methods, args):
    lines = ["# " + h for h in _header(scale, methods, args)]
    lines.append("#")
    lines.append("# per-configuration results (paper Section 6.1.3)")
    lines.append("# %-8s %5s %3s %10s %12s %12s %13s %8s"
                 % ("algo", "n", "K", "instances", "mean_obj", "best_obj",
                    "mean_time(s)", "failed"))
    for r in records:
        lines.append("  %-8s %5d %3d %10d %12.2f %12.2f %13.6f %8d"
                     % (r["algo"], r["n"], r["K"], r["instances"],
                        r["mean_obj"], r["best_obj"], r["mean_time"],
                        r["failed"]))
    lines.append("#")
    lines.append("# summary, averaged over the configurations of the scale")
    lines.append("# %-20s %10s %12s %14s"
                 % ("method", "configs", "mean_obj", "mean_time(s)"))
    for s in summary:
        lines.append("  %-20s %10d %12.2f %14.6f"
                     % (s["label"], s["configurations"], s["mean_obj"],
                        s["mean_time"]))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path, records, scale, methods, args):
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write("# " + "; ".join(_header(scale, methods, args)) + "\n")
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the F2 || WCmax algorithm comparison and record the "
                    "results in a single file (paper Section 6).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[-1],
    )
    p.add_argument("--scale", choices=sorted(SCALES), default="medium",
                   help="instance scale (default: medium)")
    p.add_argument("--algos", nargs="+", default=None,
                   help="methods to run, space- or comma-separated, or 'all' "
                        "(default: %s)" % ",".join(DEFAULT_METHODS))
    p.add_argument("--instances", type=int, default=20,
                   help="instances per configuration (default: 20, paper 6.1)")
    p.add_argument("--repeats", type=int, default=1,
                   help="runs per instance for the stochastic methods; the best "
                        "value is recorded (default: 1)")
    p.add_argument("--seed", type=int, default=42, help="base random seed")
    p.add_argument("--geometric", action="store_true",
                   help="use the geometrically spaced weight ladder "
                        "{1,2,4,...,2^(K-1)} instead of uniform {1..K}")
    p.add_argument("--time-limit", type=float, default=300.0,
                   help="solver time limit in seconds for MILP / DP "
                        "(default: 300)")
    p.add_argument("--aco-ants", type=int, default=None, help="ACO: number of ants")
    p.add_argument("--aco-iter", type=int, default=None, help="ACO: iterations")
    p.add_argument("--aco-rho", type=float, default=None, help="ACO: evaporation rate")
    p.add_argument("--ga-pop", type=int, default=None, help="GA: population size")
    p.add_argument("--ga-gen", type=int, default=None, help="GA: generations")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help="results file (default: %s; a .csv suffix writes CSV)"
                        % DEFAULT_OUT)
    p.add_argument("--list", action="store_true",
                   help="list the available methods and scales, then exit")
    return p.parse_args(argv)


def resolve_methods(spec):
    if spec is None:
        return list(DEFAULT_METHODS)
    if isinstance(spec, str):
        spec = [spec]
    names = [t for chunk in spec for t in chunk.replace(",", " ").split() if t]
    if len(names) == 1 and names[0].lower() == "all":
        return list(METHODS)
    unknown = [t for t in names if t not in METHODS]
    if unknown:
        raise SystemExit("unknown method(s): %s\navailable: %s"
                         % (", ".join(unknown), ", ".join(METHODS)))
    return names


def main(argv=None):
    args = parse_args(argv)

    if args.list:
        print("methods:")
        for name, (label, _) in METHODS.items():
            print("  %-8s %-20s applicable on: %s"
                  % (name, label, APPLICABLE[name]))
        print("scales:")
        for name, (ns, Ks) in SCALES.items():
            print("  %-8s n in %s, K in %s" % (name, ns, Ks))
        return

    methods = resolve_methods(args.algos)
    out_path = Path(args.out)

    print("F2 || WCmax -- computational study (paper Section 6)")
    print("  scale     : %s" % args.scale)
    print("  methods   : %s" % ", ".join(methods))
    print("  instances : %d per configuration, %d repeat(s) each"
          % (args.instances, args.repeats))
    print("  weights   : %s"
          % ("geometric ladder" if args.geometric else "uniform {1..K}"))
    print("  output    : %s" % out_path)
    print()

    for name in methods:
        if APPLICABLE[name] != "all" and APPLICABLE[name] != args.scale:
            print("  warning: %s is meant for the %s scale, running it on '%s' "
                  "may be intractable" % (name, APPLICABLE[name], args.scale))

    records = run_experiments(args.scale, methods, args)
    if not records:
        raise SystemExit("no results produced")

    summary = summarise(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        write_csv(out_path, records, args.scale, methods, args)
    else:
        write_text(out_path, records, summary, args.scale, methods, args)

    print()
    print("summary (averaged over the configurations of the scale):")
    print("  %-20s %10s %12s %14s"
          % ("method", "configs", "mean_obj", "mean_time(s)"))
    for s in summary:
        print("  %-20s %10d %12.2f %14.6f"
              % (s["label"], s["configurations"], s["mean_obj"], s["mean_time"]))
    print()
    print("results written to %s" % out_path)


if __name__ == "__main__":
    main()
