# -*- coding: utf-8 -*-
"""
The genetic algorithm alone, on the small and the large scale of the study.

This script runs one method -- the genetic algorithm of GA.py -- on the
configurations of run_small.py and run_large.py, and reports what the method
costs and how far from the reference its schedules are.  It is meant for the
work on the algorithm itself: when an operator is changed, or a parameter of the
search is calibrated, only the GA has to be run again, and neither the MILP
model of the small scale nor the 2-approximation and NEH of the large one are
wasted on instances that do not change the answer.

The reference value of an instance differs by scale, as in the study:

    small   the optimum of the MILP model, read from the record of the runs
            that were already made (记录/实例记录_small_*.txt), so that the gap
            is the one of Table 2 and the model is not run again;
    large   the lower bound LB = max{LB_1, LB_2, LB_3} of paper Section 4.3,
            which the instance alone determines.

The small scale therefore follows the record rather than the current setting of
run_small.py: its configurations -- sizes, weight count and processing-time
ranges -- and the seeds of its configurations are the ones of the recorded run,
so that the instances are the ones the recorded optima belong to.  The rest of
what defines a run is imported from the drivers rather than written out again:

    the generator of the instances               NEH.gen_instance
    the seed of a configuration                  run_small.seed_of / run_large.seed_of
    the large-scale configurations               run_large.CONFIGS
    the parameters of the search                 population_of, generations_of, ...

so a configuration of this script is the configuration of the driver of its
scale, drawn on the same seed, and the two cannot drift apart.

Usage:
    python run_ga.py                     both scales, twenty instances each
    python run_ga.py --scale large       the large scale alone
    python run_ga.py --instances 3       a quick check of the pipeline
    python run_ga.py --list              the configurations, then exit
    python run_ga.py --out results.txt   also write the report to a file
    python run_ga.py --small-records 记录/实例记录_small_2026-10-30_04-49-58.txt
"""
from __future__ import annotations

import argparse
import random
import re
import statistics
import sys
import time
from pathlib import Path

import GA
import NEH
import run_large

HERE = Path(__file__).resolve().parent
RECORD_DIR = HERE / "记录"


# ---------------------------------------------------------------------------
# the optima of the small scale, read from the record of a run
# ---------------------------------------------------------------------------
#
# The record holds one line per method and instance:
#
#   MILP  n=10   K=30   [1,40]  | seed=13082   | instance  1/20 | value 2522.0 ...
#
# The optimum of an instance is the value of its MILP line; the instances of a
# configuration are the ones of the run, in the order it took them, and their
# count is the denominator of that line.

_RECORD_LINE = re.compile(
    r"\bMILP\s+n=(\d+)\s+K=(\d+)\s+\[1,(\d+)\]\s+\|\s+seed=(\d+)\s+"
    r"\|\s+instance\s+(\d+)/(\d+)\s+\|\s+value\s+([\d.]+)")


def latest_small_records():
    """The newest record of a small-scale run, or None when there is none."""
    files = sorted(RECORD_DIR.glob("实例记录_small_*.txt"))
    return files[-1] if files else None


def load_small_optima(path):
    """
    The MILP optimum of every instance of a small-scale record.

    Returns {(n, K, proc_hi, seed): {"instances": count, "opt": {index: value}}}.
    A configuration whose instances are not all in the record is kept as it is
    and dropped by the caller, so that a resumed run is not mistaken for a
    complete one.
    """
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _RECORD_LINE.search(line)
            if not m:
                continue
            n, K, proc_hi, seed, idx, total, value = m.groups()
            key = (int(n), int(K), int(proc_hi), int(seed))
            entry = out.setdefault(key, {"instances": int(total), "opt": {}})
            entry["opt"][int(idx)] = float(value)
    return out


# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------
#
# A configuration is a dictionary with the settings of one instance group --
# the scale, the number of jobs, the number of weight classes, the processing-
# time range -- together with the seed the driver of its scale gives it, and,
# on the small scale, the optima of its instances.

def small_configs(records, base_seed: int, stream=sys.stdout):
    """
    The configurations of the small scale, as the recorded run made them.

    Without a record the configurations fall back to the current setting of
    run_small.py, and the reference falls back to the lower bound.
    """
    if records is None:
        import run_small
        K = run_small.WEIGHT_COUNT
        print("# no small-scale record found: configurations of run_small.py, "
              "reference LB", file=stream)
        for n in run_small.SIZES:
            for proc_hi in run_small.RANGES:
                yield {"scale": "small", "n": n, "K": K, "weights": "narrow",
                       "proc_hi": proc_hi, "weight_label": "K=%d" % K,
                       "range_label": "[1,%d]" % proc_hi,
                       "seed": run_small.seed_of(n, K, proc_hi, base_seed),
                       "optima": None}
        return

    optima = load_small_optima(records)
    import run_small
    ks = sorted({key[1] for key in optima})
    if ks and ks != [run_small.WEIGHT_COUNT]:
        print("# note: the record holds K=%s while run_small.py now sets K=%d; "
              "the record is followed" % ("/".join(str(k) for k in ks),
                                          run_small.WEIGHT_COUNT), file=stream)
    for (n, K, proc_hi, seed) in sorted(optima):
        entry = optima[(n, K, proc_hi, seed)]
        if len(entry["opt"]) != entry["instances"]:
            print("# skipping n=%d K=%d [1,%d]: %d of %d instances recorded"
                  % (n, K, proc_hi, len(entry["opt"]), entry["instances"]), file=stream)
            continue
        yield {"scale": "small", "n": n, "K": K, "weights": "narrow",
               "proc_hi": proc_hi, "weight_label": "K=%d" % K,
               "range_label": "[1,%d]" % proc_hi, "seed": seed,
               "optima": [entry["opt"][i] for i in range(1, entry["instances"] + 1)]}


def large_configs(base_seed: int):
    """The configurations of the large scale, as run_large.py runs them."""
    for (n, setting, rng_label) in run_large.CONFIGS:
        K = n if setting == "free" else int(setting[1:])
        yield {"scale": "large", "n": n, "K": K,
               "weights": "free" if setting == "free" else "narrow",
               "proc_hi": run_large.PROC_RANGES[rng_label],
               "weight_label": "K=%d" % K if setting != "free" else "all weights",
               "range_label": rng_label,
               "seed": run_large.seed_of(n, setting, rng_label, base_seed),
               "optima": None}


def configs_of(scale: str, base_seed: int, records=None, stream=sys.stdout):
    """The configurations of one scale, or of both when `scale` is "both"."""
    if scale == "small":
        return list(small_configs(records, base_seed, stream))
    if scale == "large":
        return list(large_configs(base_seed))
    return (list(small_configs(records, base_seed, stream))
            + list(large_configs(base_seed)))


# ---------------------------------------------------------------------------
# the parameters of the search
# ---------------------------------------------------------------------------
#
# The rules of the driver of the scale, read from that module rather than
# written out again: the two scales set the population, the offspring, the
# probabilities and the width of the local search separately.

def ga_kwargs(cfg, args):
    """The arguments of GA.solve_instance on a configuration of `cfg`."""
    n = cfg["n"]
    if cfg["scale"] == "small":
        import run_small as rules
    else:
        rules = run_large
    kwargs = {"N": rules.population_of(n),
              "G": rules.generations_of(n),
              "lam": rules.offspring_of(n),
              "pc": rules.crossover_probability_of(n),
              "pm": rules.mutation_probability_of(n),
              "k": rules.descent_width_of(n)}
    if args.ga_pop is not None:
        kwargs["N"] = args.ga_pop
    if args.ga_gen is not None:
        kwargs["G"] = args.ga_gen
    if args.ga_pc is not None:
        kwargs["pc"] = args.ga_pc
    if args.ga_pm is not None:
        kwargs["pm"] = args.ga_pm
    return kwargs


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run_config(cfg, args):
    """
    Solve the instances of one configuration, and report the gap and the time.

    The instances and the seeds of the genetic algorithm are drawn the way the
    driver of the scale draws them: the instance from the seed of the
    configuration, the seed of each solve from the stream "solver-<seed>", so
    that the instances of this script are the ones the driver solves.

    The reference of instance i is the optimum the record holds for it when the
    configuration comes from a record, and the lower bound otherwise.
    """
    n = cfg["n"]
    rng = random.Random(cfg["seed"])
    solver_rng = random.Random("solver-%d" % cfg["seed"])
    kwargs = ga_kwargs(cfg, args)
    gaps, times = [], []
    for i in range(args.instances):
        a, b, w = NEH.gen_instance(n, cfg["K"], rng, weights=cfg["weights"],
                                   proc_hi=cfg["proc_hi"])
        if cfg["optima"] is not None and i < len(cfg["optima"]):
            ref = cfg["optima"][i]
        else:
            ref = NEH.instance_baseline(a, b, w)
        t0 = time.perf_counter()
        obj, _sigma, _pop, _base = GA.solve_instance(
            a, b, w, seed=solver_rng.randrange(1 << 30), **kwargs)
        times.append(time.perf_counter() - t0)
        gaps.append(100.0 * (obj - ref) / ref if ref > 0 else 0.0)
    return {"gap": statistics.mean(gaps), "gap_min": min(gaps), "gap_max": max(gaps),
            "time": statistics.mean(times), "time_max": max(times), "gaps": gaps}


def report_line(label, cfg, r, instances):
    """One line of the report: a configuration, or a scale, and its figures."""
    ref = "MILP" if cfg["optima"] is not None else "LB"
    return ("%-6s %-4s %-8s %-7s %-5s %4d  gap %6.2f  (%5.2f..%6.2f)  "
            "time %7.2f s  (%7.2f)"
            % (label, cfg["n"], cfg["weight_label"], cfg["range_label"], ref, instances,
               r["gap"], r["gap_min"], r["gap_max"], r["time"], r["time_max"]))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="The genetic algorithm alone, on the scales of the study.")
    parser.add_argument("--scale", choices=("small", "large", "both"), default="both",
                        help="the scale to run (default: both)")
    parser.add_argument("--instances", type=int, default=20,
                        help="instances per configuration (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="base random seed, used where no record is read (default: 42)")
    parser.add_argument("--small-records", default=None,
                        help="record of the small-scale run the optima are read from "
                             "(default: the newest 记录/实例记录_small_*.txt)")
    parser.add_argument("--ga-pop", type=int, default=None, help="GA: population size")
    parser.add_argument("--ga-gen", type=int, default=None, help="GA: generations")
    parser.add_argument("--ga-pc", type=float, default=None, help="GA: crossover probability")
    parser.add_argument("--ga-pm", type=float, default=None, help="GA: mutation probability")
    parser.add_argument("--out", default=None, help="also write the report to this file")
    parser.add_argument("--list", action="store_true",
                        help="list the configurations, then exit")
    args = parser.parse_args(argv)

    records = Path(args.small_records) if args.small_records else latest_small_records()
    if args.scale in ("small", "both") and records is not None:
        print("# small-scale optima read from %s" % records, flush=True)

    todo = configs_of(args.scale, args.seed, records)

    if args.list:
        for cfg in todo:
            ref = "MILP" if cfg["optima"] is not None else "LB"
            print("  %-5s n=%-4d %-6s %-7s  seed %d  ref %s"
                  % (cfg["scale"], cfg["n"], cfg["weight_label"], cfg["range_label"],
                     cfg["seed"], ref))
        return 0

    lines = ["scale  n    weights  range   ref   inst  mean gap (min..max)      "
             "mean time (max)",
             "-" * 96]
    totals = {}
    for cfg in todo:
        r = run_config(cfg, args)
        line = report_line(cfg["scale"], cfg, r, args.instances)
        print(line, flush=True)
        lines.append(line)
        totals.setdefault(cfg["scale"], []).append(r)

    for scale, rows in totals.items():
        gaps = [g for r in rows for g in r["gaps"]]
        mean_gap = statistics.mean(gaps)
        worst = max(gaps)
        mean_time = statistics.mean(r["time"] for r in rows)
        line = ("%-6s %-4s %-8s %-7s %-5s %4d  gap %6.2f  (worst %5.2f)  time %7.2f s"
                % (scale, "all", "-", "-", "-", len(gaps), mean_gap, worst, mean_time))
        print(line, flush=True)
        lines.append(line)

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
        print("report written to %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
