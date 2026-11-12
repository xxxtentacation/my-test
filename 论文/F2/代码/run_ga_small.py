# -*- coding: utf-8 -*-
"""
The genetic algorithm alone, on the small scale of the study.

This script runs one method -- the genetic algorithm of GA.py -- on the
configurations of run_small.py, and reports what the method costs and how far
from the reference its schedules are.  It is meant for the work on the algorithm
itself: when an operator is changed, or a parameter of the search is calibrated,
only the GA has to be run again, and the MILP model does not have to be solved a
second time.

The reference value of an instance is the optimum of the MILP model, read from
the records of the runs that were already made (记录/实例记录_small*.txt).  A
record is accepted only when its NEH values match the instances the current
generator draws, so the optima of a run made under another weight distribution
are not read on the wrong instances; where no record matches, the configuration
falls back to the lower bound LB = max{LB_1, LB_2, LB_3} of paper Section 4.3.

The configurations and seeds are those of run_small.py, imported from it rather
than written out again:

    the generator of the instances               NEH.gen_instance
    the seed of a configuration                  run_small.seed_of
    the configurations                           run_small.SCALES
    the parameters of the search                 population_of, generations_of, ...

so a configuration of this script is the configuration of run_small.py, drawn
on the same seed: the instances, and the seed each instance hands the GA, are
the ones the small-scale driver uses, and the two cannot drift apart.

Usage:
    python run_ga_small.py                     every configuration, 20 instances each
    python run_ga_small.py --list              the configurations, then exit
    python run_ga_small.py --instances 3       a quick check of the pipeline
    python run_ga_small.py --out results.txt   also write the report to a file
    python run_ga_small.py --small-records 记录/实例记录_small100_2026-11-10_02-24-04.txt
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

#: the NEH line of the same record: NEH is a deterministic function of the
#: instance alone, so a recorded NEH value is a fingerprint of the instance the
#: recorded optimum belongs to.  It is used to reject the optima of an older run
#: whose instances were drawn under another weight distribution.
_NEH_LINE = re.compile(
    r"\bNEH\s+n=(\d+)\s+K=(\d+)\s+\[1,(\d+)\]\s+\|\s+seed=(\d+)\s+"
    r"\|\s+instance\s+(\d+)/(\d+)\s+\|\s+value\s+([\d.]+)")


def default_small_records():
    """
    The records of the small-scale runs the optima are read from, oldest first.

    They are ordered by the timestamp in the file name rather than by the name
    itself, so that a configuration held by several runs is read on a fixed
    order: the first record whose NEH values match the instances the current
    generator draws is the one used.
    """
    stamp = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.txt$")

    def when(path):
        m = stamp.search(path.name)
        return m.group(1) if m else ""

    return sorted(RECORD_DIR.glob("实例记录_small*.txt"), key=when)


def load_small_optima(path):
    """
    The MILP optimum and the NEH value of every instance of a small-scale record.

    Returns {(n, K, proc_hi, seed): {"instances": count, "opt": {}, "neh": {}}}.
    A configuration whose instances are not all in the record keeps the prefix
    it holds, and is dropped by the caller when the prefix is short.
    """
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _RECORD_LINE.search(line)
            if m:
                n, K, proc_hi, seed, idx, total, value = m.groups()
                key = (int(n), int(K), int(proc_hi), int(seed))
                entry = out.setdefault(
                    key, {"instances": int(total), "opt": {}, "neh": {}})
                entry["opt"][int(idx)] = float(value)
                continue
            m = _NEH_LINE.search(line)
            if m:
                n, K, proc_hi, seed, idx, total, value = m.groups()
                key = (int(n), int(K), int(proc_hi), int(seed))
                entry = out.setdefault(
                    key, {"instances": int(total), "opt": {}, "neh": {}})
                entry["neh"][int(idx)] = float(value)
    return out


# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------
#
# A configuration is a dictionary with the settings of one instance group --
# n, K, the processing-time range -- together with the seed run_small.py gives
# it and, when a record holds the same instances, the optima of its instances.

def recorded_optima(entry, n, K, proc_hi, seed):
    """
    The optima of a record, but only when the record holds the same instances.

    NEH is a deterministic function of the instance, so the NEH value a record
    holds is a fingerprint of the instance its optimum belongs to.  The instance
    is redrawn here, on the configuration's own seed, and the record is accepted
    only when every one of its instances carries the fingerprint the record
    wrote for it; an older run of another weight distribution is thus rejected
    rather than read on the wrong denominators.  Returns the list of optima, or
    None when the record does not belong to the instances of the current
    generator.
    """
    count = entry["instances"]
    if any(i not in entry["opt"] or i not in entry["neh"]
           for i in range(1, count + 1)):
        return None
    rng = random.Random(seed)
    for i in range(1, count + 1):
        a, b, w = NEH.gen_instance(n, K, rng, weights="narrow", proc_hi=proc_hi)
        here = NEH.wcmax_of_order(NEH.neh(a, b, w), a, b, w)
        if abs(here - entry["neh"][i]) > 0.5:
            return None
    return [entry["opt"][i] for i in range(1, count + 1)]


def small_configs(records, base_seed: int):
    """
    The configurations of the small scale, in the order run_small.py runs them.

    The seed of a configuration is the one run_small.py gives it, so that the
    instances, and the seed each instance hands the GA, are the ones of the
    small-scale run.  When a record holds the same instances, its MILP values
    are the references of those instances; otherwise the reference falls back
    to the lower bound.
    """
    import run_small
    # A configuration may be held by more than one run, drawn under different
    # weight distributions; every record is kept, and the first whose NEH
    # fingerprint matches the current instances supplies the optima.
    optima = {}
    for path in records or ():
        for key, entry in load_small_optima(path).items():
            optima.setdefault(key, []).append(entry)
    for spec in run_small.SCALES.values():
        for (n, K, proc_hi) in spec["configs"]:
            seed = run_small.seed_of(n, K, proc_hi, base_seed)
            ref = None
            for entry in optima.get((n, K, proc_hi, seed), ()):
                ref = recorded_optima(entry, n, K, proc_hi, seed)
                if ref is not None:
                    break
            yield {"n": n, "K": K, "weights": "narrow", "geometric": False,
                   "proc_hi": proc_hi, "weight_label": "K=%d" % K,
                   "range_label": "[1,%d]" % proc_hi, "seed": seed,
                   "optima": ref}


# ---------------------------------------------------------------------------
# the parameters of the search
# ---------------------------------------------------------------------------
#
# The rules of run_small.py, read from it rather than written out again.

def ga_kwargs(n, args):
    """The arguments of GA.solve_instance on an instance of n jobs."""
    import run_small as rules
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

    The instances and the seeds of the genetic algorithm are drawn the way
    run_small.py draws them: the instance from the seed of the configuration,
    the seed of each solve from the stream "solver-<seed>", so that the
    instances of this script are the ones the driver solves, and the GA is run
    on the seeds the driver would have given it.

    The reference of instance i is the optimum the record holds for it when the
    record belongs to the very instances this run draws -- `recorded_optima`
    has already made that check -- and the lower bound otherwise.
    """
    n = cfg["n"]
    rng = random.Random(cfg["seed"])
    solver_rng = random.Random("solver-%d" % cfg["seed"])
    kwargs = ga_kwargs(n, args)
    data = [NEH.gen_instance(n, cfg["K"], rng,
                             geometric=cfg.get("geometric", False),
                             weights=cfg["weights"], proc_hi=cfg["proc_hi"])
            for _ in range(args.instances)]
    # A configuration shorter than the run takes the lower bound for all of its
    # instances, rather than mixing two references inside one configuration.
    use_optima = (cfg.get("optima") is not None
                  and len(cfg["optima"]) >= args.instances)
    gaps, times = [], []
    for i, (a, b, w) in enumerate(data):
        if use_optima:
            ref = cfg["optima"][i]
        else:
            ref = NEH.instance_baseline(a, b, w)
        # the seed of the instance, drawn once from the solver stream exactly as
        # run_small.py draws it, so that the GA solves the instance on the seed
        # the small-scale driver would have given it
        seed = solver_rng.randrange(1 << 30)
        t0 = time.perf_counter()
        obj, _sigma, _pop, _base = GA.solve_instance(
            a, b, w, seed=seed, **kwargs)
        times.append(time.perf_counter() - t0)
        gaps.append(100.0 * (obj - ref) / ref if ref > 0 else 0.0)
    return {"gap": statistics.mean(gaps), "gap_min": min(gaps),
            "gap_max": max(gaps), "time": statistics.mean(times),
            "time_max": max(times), "gaps": gaps,
            "ref": "MILP" if use_optima else "LB"}


def report_line(cfg, r, instances):
    """One line of the report: a configuration and its figures."""
    return ("%-4s %-8s %-7s %-5s %4d  gap %6.2f  (%5.2f..%6.2f)  "
            "time %7.2f s  (%7.2f)"
            % (cfg["n"], cfg["weight_label"], cfg["range_label"], r["ref"],
               instances, r["gap"], r["gap_min"], r["gap_max"],
               r["time"], r["time_max"]))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="The genetic algorithm alone, on the small scale of the study.")
    parser.add_argument("--instances", type=int, default=20,
                        help="instances per configuration (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="base random seed of run_small.py: the instances and "
                             "their GA seeds are the ones of the small-scale run "
                             "(default: 42)")
    parser.add_argument("--small-records", nargs="+", default=None, metavar="FILE",
                        help="record(s) of the small-scale runs the optima are read "
                             "from (default: every 记录/实例记录_small*.txt)")
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

    records = ([Path(p) for p in args.small_records] if args.small_records
               else default_small_records())
    if records:
        print("# small-scale optima read from %s"
              % ", ".join(str(p) for p in records), flush=True)
    else:
        print("# no small-scale record: references are LB", flush=True)

    todo = list(small_configs(records, args.seed))

    if args.list:
        for cfg in todo:
            ref = "MILP" if cfg["optima"] is not None else "LB"
            print("  n=%-4d %-6s %-7s  seed %d  ref %s"
                  % (cfg["n"], cfg["weight_label"], cfg["range_label"],
                     cfg["seed"], ref))
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
