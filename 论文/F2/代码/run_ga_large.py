# -*- coding: utf-8 -*-
"""
The genetic algorithm alone, on the large scale of the study.

This script runs one method -- the genetic algorithm of GA.py -- on the
configurations of run_large.py, and reports what the method costs and how far
from the reference its schedules are.  It is meant for the work on the algorithm
itself: when an operator is changed, or a parameter of the search is calibrated,
only the GA has to be run again, and neither the 2-approximation nor NEH has to
be run on instances that do not change the answer.

The reference value of an instance is the lower bound LB = max{LB_1, LB_2, LB_3}
of paper Section 4.3, which the instance alone determines, so no record is
needed and the gap is non-negative.

The configurations and seeds are those of run_large.py, imported from it rather
than written out again:

    the generator of the instances               NEH.gen_instance
    the seed of a configuration                  run_large.seed_of
    the configurations                           run_large.SCALES
    the parameters of the search                 population_of, generations_of, ...

so a configuration of this script is the configuration of run_large.py, drawn
on the same seed: the instances, and the seed each instance hands the GA, are
the ones the large-scale driver uses, and the two cannot drift apart.

Usage:
    python run_ga_large.py                     every configuration, 20 instances each
    python run_ga_large.py --list              the configurations, then exit
    python run_ga_large.py --instances 3       a quick check of the pipeline
    python run_ga_large.py --out results.txt   also write the report to a file
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

import GA
import NEH
import run_large

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------
#
# A configuration is a dictionary with the settings of one instance group --
# n, K, the weight setting and the processing-time range -- together with the
# seed run_large.py gives it.

def large_configs(base_seed: int):
    """The configurations of the large scale, as run_large.py runs them."""
    for spec in run_large.SCALES.values():
        for (n, setting, rng_label) in spec["configs"]:
            cfg = run_large.config_of(n, setting, rng_label)
            yield {"n": n, "K": cfg["K"], "weights": cfg["weights"],
                   "geometric": False, "proc_hi": cfg["proc_hi"],
                   "weight_label": cfg["weight_label"],
                   "range_label": cfg["range_label"],
                   "seed": run_large.seed_of(n, setting, rng_label, base_seed)}


# ---------------------------------------------------------------------------
# the parameters of the search
# ---------------------------------------------------------------------------
#
# The rules of run_large.py, read from it rather than written out again.

def ga_kwargs(n, args):
    """The arguments of GA.solve_instance on an instance of n jobs."""
    kwargs = {"N": run_large.population_of(n),
              "G": run_large.generations_of(n),
              "lam": run_large.offspring_of(n),
              "pc": run_large.crossover_probability_of(n),
              "pm": run_large.mutation_probability_of(n),
              "k": run_large.descent_width_of(n)}
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

    The instances and the seeds of the genetic algorithm are drawn the way
    run_large.py draws them: the instance from the seed of the configuration,
    the seed of each solve from the stream "solver-<seed>", so that the
    instances of this script are the ones the driver solves, and the GA is run
    on the seeds the driver would have given it.  The reference of an instance
    is the lower bound LB of paper Section 4.3, which the instance determines.
    """
    n = cfg["n"]
    rng = random.Random(cfg["seed"])
    solver_rng = random.Random("solver-%d" % cfg["seed"])
    kwargs = ga_kwargs(n, args)
    data = [NEH.gen_instance(n, cfg["K"], rng, weights=cfg["weights"],
                             proc_hi=cfg["proc_hi"])
            for _ in range(args.instances)]
    gaps, times = [], []
    for a, b, w in data:
        ref = NEH.instance_baseline(a, b, w)
        # the seed of the instance, drawn once from the solver stream exactly as
        # run_large.py draws it, so that the GA solves the instance on the seed
        # the large-scale driver would have given it
        seed = solver_rng.randrange(1 << 30)
        t0 = time.perf_counter()
        obj, _sigma, _pop, _base = GA.solve_instance(
            a, b, w, seed=seed, **kwargs)
        times.append(time.perf_counter() - t0)
        gaps.append(100.0 * (obj - ref) / ref if ref > 0 else 0.0)
    return {"gap": statistics.mean(gaps), "gap_min": min(gaps),
            "gap_max": max(gaps), "time": statistics.mean(times),
            "time_max": max(times), "gaps": gaps, "ref": "LB"}


def report_line(cfg, r, instances):
    """One line of the report: a configuration and its figures."""
    return ("%-4s %-8s %-7s %-5s %4d  gap %6.2f  (%5.2f..%6.2f)  "
            "time %7.2f s  (%7.2f)"
            % (cfg["n"], cfg["weight_label"], cfg["range_label"], r["ref"],
               instances, r["gap"], r["gap_min"], r["gap_max"],
               r["time"], r["time_max"]))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="The genetic algorithm alone, on the large scale of the study.")
    parser.add_argument("--instances", type=int, default=20,
                        help="instances per configuration (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="base random seed of run_large.py: the instances and "
                             "their GA seeds are the ones of the large-scale run "
                             "(default: 42)")
    parser.add_argument("--ga-pop", type=int, default=None, help="GA: population size")
    parser.add_argument("--ga-gen", type=int, default=None, help="GA: generations")
    parser.add_argument("--ga-pc", type=float, default=None,
                        help="GA: crossover probability")
    parser.add_argument("--ga-pm", type=float, default=None,
                        help="GA: mutation probability")
    parser.add_argument("--out", default=None, help="also write the report to this file")
    parser.add_argument("--list", action="store_true",
                        help="list the configurations, then exit")
    args = parser.parse_args(argv)

    todo = list(large_configs(args.seed))

    if args.list:
        for cfg in todo:
            print("  n=%-4d %-6s %-7s  seed %d  ref LB"
                  % (cfg["n"], cfg["weight_label"], cfg["range_label"],
                     cfg["seed"]))
        return 0

    lines = ["n    weights  range   ref   inst  mean gap (min..max)      "
             "mean time (max)",
             "-" * 88]
    rows = []
    for cfg in todo:
        r = run_config(cfg, args)
        line = report_line(cfg, r, args.instances)
        print(line, flush=True)
        lines.append(line)
        rows.append(r)

    gaps = [g for r in rows for g in r["gaps"]]
    line = ("%-4s %-8s %-7s %-5s %4d  gap %6.2f  (worst %5.2f)  time %7.2f s"
            % ("all", "-", "-", "-", len(gaps), statistics.mean(gaps),
               max(gaps), statistics.mean(r["time"] for r in rows)))
    print(line, flush=True)
    lines.append(line)

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
        print("report written to %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
