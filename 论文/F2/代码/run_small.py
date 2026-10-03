# -*- coding: utf-8 -*-
"""
Driver for the **small** scale of the computational study of paper Section 6.

This is the small-scale third of the study: the medium scale is run by
``run_medium.py`` and the large one by ``run_large.py``.  The three scripts share
the driver below and differ only in the configurations and the methods they run,
so a scale can be run, interrupted and repeated on its own.

The script follows the design of paper Section 6 rather than inventing its own:

* **Instances.**  The small scale of paper Table 1 varies two factors.  The
  weights take the values {1, ..., K} with K = 2 and K = 3, which is what the
  tables of the paper label K=2 and K=3, and the processing times are drawn from
  the range [1,20] or the wider [1,40].  Both are crossed with the four instance
  sizes, and each of the 16 configurations is replicated 20 times.

* **Methods.**  MILP and DP, the two exact methods of the paper, which are still
  within reach at this scale.  Neither of them is given a gap: the reference of
  this scale is the value of the model itself, so a method is measured against
  its own answer and there is nothing for a relative figure to say.  What the
  tables of the paper report for them, and what this script records, is the
  number of instances each settles to optimality and its running time.

* **Parameters.**  Those of paper Section 6, all of them the defaults of the
  algorithm modules or the constants below: a time limit of 600 s shared by MILP
  and DP, and Gurobi on a single thread.

* **Optimality count.**  Once both methods of a configuration have run, the best
  value found on each instance is determined, and each method reports the number
  of instances on which it attains that value.  The MILP model is one of the two,
  so that value is the optimum and the count is the number of instances solved
  to optimality.  A method that hits the time limit without producing a value
  counts on none of them and is reported separately.

* **Seed.**  The seed of a configuration depends only on the configuration and
  never on the method, so both methods see literally the same instances.

* **Output.**  The run is reported line by line as it happens: one line per
  instance, carrying the value found, the running time of that instance and the
  time elapsed since the method started on the configuration, and one line per
  configuration once both methods have run.  stdout is put in line-buffered
  mode, so the same holds when the output is redirected to a file.

The run is written to the small-scale results file, and a copy is filed away
under the date, so that running the script again never overwrites the results of
an earlier run::

    论文/F2/代码/记录/结果_small_最新.txt              <- the run that just finished
    论文/F2/代码/记录/结果存档/<date>_small.txt        <- one file per run, kept
    论文/F2/代码/记录/实例记录_small_<date>.txt        <- one line per instance, live
    论文/F2/代码/run_small.py                         <- this script
    论文/F2/代码/{MILP,DP}.py                         <- the two exact methods

Usage
-----
    python run_small.py                       # the whole scale
    python run_small.py --list                # what would be run
    python run_small.py --algos milp          # one method
    python run_small.py --instances 2         # a quick check
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
#: everything the run produces is written here, and nowhere else: the results of
#: the run that has just finished, the kept copy of every earlier run, and the
#: per-instance records
RECORD_DIR = HERE / "记录"
#: default results file of this scale: holds the run that has just finished, and
#: is overwritten by the next one -- the kept copies are the archive below
DEFAULT_OUT = RECORD_DIR / "结果_small_最新.txt"
#: every run is also filed here under its date, so that no run overwrites the
#: results of an earlier run
ARCHIVE_DIR = RECORD_DIR / "结果存档"

#: weight settings of paper Table 1, in the order used to derive the seed.  The
#: key names the number of distinct weights the setting produces, which is how
#: the tables of the paper label their rows; the generator draws the weights
#: uniformly from {1, ..., K}, so the key is also the K the exact methods take.
WEIGHT_SETTINGS = ("K2", "K3")

#: processing-time ranges of the scale: the label of the table row, and the upper
#: end of the discrete uniform range both processing times are drawn from
PROC_RANGES = {"[1,20]": 20, "[1,40]": 40}

#: the scale this script runs, and the only place where it is written down
SCALE = "small"
#: instances per configuration
INSTANCES = 20
#: the methods the paper reports at this scale
METHODS_OF_SCALE = ("milp", "dp")
#: the configurations of paper Section 6 and Table 1: the four instance sizes
#: crossed with the two weight settings and the two processing-time ranges
CONFIGS = tuple((n, wt, rg)
                for n in (40, 50, 60, 70)
                for wt in WEIGHT_SETTINGS
                for rg in ("[1,20]", "[1,40]"))


def config_of(n, weights, rng):
    """
    The arguments of one configuration, and the labels its row is written with.

    The weight setting names the count K, and the generator is asked for the
    narrow regime, which draws the weights uniformly from {1, ..., K}; K is
    passed on to the model and to the dynamic program as well, since both take it
    as the number of weight classes.
    """
    return {
        "n": n,
        "K": int(weights[1:]),
        "weights": "narrow",
        "proc_hi": PROC_RANGES[rng],
        "weight_label": "K=" + weights[1:],
        "range_label": rng,
    }


# ---------------------------------------------------------------------------
# runners: one thin adapter per algorithm, each calling that module's benchmark
# ---------------------------------------------------------------------------

def _run_milp(n, cfg, instances, seed, opts, progress=None):
    from MILP import benchmark
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed,
                     weights=cfg["weights"], proc_hi=cfg["proc_hi"],
                     time_limit=opts.time_limit, progress=progress,
                     log_path=getattr(opts, "record_path", None))


def _run_dp(n, cfg, instances, seed, opts, progress=None):
    from DP import benchmark
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed,
                     weights=cfg["weights"], proc_hi=cfg["proc_hi"],
                     time_limit=opts.time_limit, progress=progress,
                     log_path=getattr(opts, "record_path", None))


#: output order of the methods, and the runner of each
METHODS = {
    "milp": ("MILP", _run_milp),
    "dp":   ("DP",   _run_dp),
}


# ---------------------------------------------------------------------------
# experiment driver
# ---------------------------------------------------------------------------

def seed_of(n, weights, rng, base):
    """Seed of a configuration; depends on the configuration, never on the method."""
    return base + 1000 * n + 100 * WEIGHT_SETTINGS.index(weights) \
        + PROC_RANGES[rng]


def _hms(seconds):
    """Seconds as h:mm:ss, or m:ss below an hour."""
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return ("%d:%02d:%02d" % (h, m, s)) if h else ("%d:%02d" % (m, s))


def _instance_reporter(label, n, weights, stream):
    """
    One line per instance, printed as soon as that instance is done.

    A configuration can take an hour or more -- the model is given ten minutes on
    every instance of the largest size -- so the driver would otherwise be silent
    for the whole of it.  The reporter carries the running time of the instance
    and the time elapsed since the method started on this configuration.
    """
    started = time.perf_counter()

    def report(done, total, obj, seconds):
        print("      %-4s n=%-4d %-6s | instance %2d/%-2d | value %-14s"
              " %9s | elapsed %s"
              % (label, n, weights, done, total,
                 "none" if obj is None else "%.1f" % obj,
                 "%.2f s" % (seconds or 0.0),
                 _hms(time.perf_counter() - started)),
              file=stream, flush=True)

    return report


def run_scale(methods, args, stream=sys.stdout):
    """
    Run every selected method on every configuration of the scale.

    Neither method is given a gap: the reference of this scale is the value of
    the model itself, so what is recorded for each of them is how many instances
    it settles and how long it takes.

    Returns a list of records, one per (method, n, weights, range):
        algo, n, weights, range, instances, missing, mean_time, hit
    """
    instances = args.instances or INSTANCES
    total = len(CONFIGS) * len(methods)
    records, done = [], 0

    for (n, weights, rng) in CONFIGS:
        cfg = config_of(n, weights, rng)
        seed = seed_of(n, weights, rng, args.seed)
        wlabel, rlabel = cfg["weight_label"], cfg["range_label"]

        # The model is run first whatever --algos says: the best value found on
        # an instance is the optimum only when the model is one of the methods
        # compared on it.
        label, runner = METHODS["milp"]
        print("  n=%-4d %-6s %-7s | %s (seed %d)"
              % (n, wlabel, rlabel, label, seed), file=stream, flush=True)
        milp_res = runner(n, cfg, instances, seed, args,
                          _instance_reporter(label, n, wlabel, stream))

        # Every method of the configuration is run first, because the number of
        # instances on which a method finds the best solution cannot be counted
        # until all of them have reported on the same instances.
        results = {}
        for name in methods:
            label, runner = METHODS[name]
            if name == "milp":
                results[name] = base_res
            else:
                results[name] = runner(n, cfg, instances, seed, args,
                                       _instance_reporter(label, n, wlabel, stream))

        # The best value found on each instance by either method.  The model is
        # one of them, so this is the optimum of the instance.
        best_of_instance = []
        for i in range(instances):
            vals = [results[m]["objs"][i] for m in methods
                    if results[m]["objs"][i] is not None]
            best_of_instance.append(min(vals) if vals else None)

        for name in methods:
            label = METHODS[name][0]
            res = results[name]
            missing = sum(1 for o in res["objs"] if o is None)
            hit = sum(1 for i, o in enumerate(res["objs"])
                      if o is not None and best_of_instance[i] is not None
                      and abs(o - best_of_instance[i]) < 1e-9)
            rec = {
                "algo": name,
                "label": label,
                "n": n,
                "K": cfg["K"],
                "weights": wlabel,
                "range": rlabel,
                "instances": instances,
                "missing": missing,
                "mean_time": res["mean_time"],
                "hit": hit,
                # the raw per-instance values, so that the file still holds the
                # run after the driver has moved on: another reading of it can
                # then be applied without repeating the run
                "objs": res["objs"],
                "times": res["times"],
                "best_of_instance": best_of_instance,
            }
            records.append(rec)
            done += 1
            print("  [%2d/%2d] %-4s n=%-4d %-6s %-7s | opt %2d/%-2d "
                  "| %9.4f s | no value %d"
                  % (done, total, label, n, wlabel, rlabel, hit, instances,
                     rec["mean_time"], missing),
                  file=stream, flush=True)

    return records


def summarise(records):
    """Total optimality count and mean CPU time of each method, over the scale."""
    by_algo = {}
    for rec in records:
        by_algo.setdefault(rec["algo"], []).append(rec)
    out = []
    for name in METHODS:
        recs = by_algo.get(name, [])
        if not recs:
            continue
        out.append({
            "algo": name,
            "label": METHODS[name][0],
            "configurations": len(recs),
            "instances": sum(r["instances"] for r in recs),
            "opt": sum(r["hit"] for r in recs),
            "mean_time": statistics.fmean(r["mean_time"] for r in recs),
        })
    return out


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

def _header(methods, args):
    return [
        "F2 || WCmax -- computational study (paper Section 6)",
        "scale         : %s" % SCALE,
        "generated     : %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "methods       : %s" % ", ".join(METHODS[m][0] for m in methods),
        "instances/cfg : %d" % (args.instances or INSTANCES),
        "seed          : %d (+1000*n, + 100*weight setting, + range)" % args.seed,
        "time limit    : %s s" % (args.time_limit if args.time_limit else "none"),
        "records       : %s" % getattr(args, "record_path", "-"),
    ]


def _scale_block(lines, records, summary, args):
    lines.append("")
    lines.append("# " + "=" * 84)
    lines.append("# scale %s  --  configurations: %s"
                 % (SCALE, ", ".join("n=%d/%s/%s"
                                     % (n, config_of(n, wt, rg)["weight_label"], rg)
                                     for n, wt, rg in CONFIGS)))
    lines.append("# " + "=" * 84)
    lines.append("# per-configuration results")
    lines.append("# %-5s %5s %-7s %-7s %9s %6s %11s"
                 % ("algo", "n", "weights", "range", "instances", "opt",
                    "time(s)"))
    for r in records:
        lines.append("  %-5s %5d %-7s %-7s %9d %6d %11.4f"
                     % (r["label"], r["n"], r["weights"], r["range"],
                        r["instances"], r["hit"], r["mean_time"]))
    lines.append("#")
    lines.append("# the column 'opt' counts the instances on which the method attained the")
    lines.append("# best value found on that instance by either method; the model is one of")
    lines.append("# them, so that value is the optimum")
    lines.append("#")
    lines.append("# per-instance objective values and CPU times, in instance order")
    seen = set()
    for r in records:
        key = (r["n"], r["weights"], r["range"])
        if key not in seen:            # the same for every method of the configuration
            seen.add(key)
            lines.append("  %-5s %5d %-7s %-7s best  : %s"
                         % ("-", r["n"], r["weights"], r["range"],
                            " ".join("none" if b is None else "%.1f" % b
                                     for b in r["best_of_instance"])))
        lines.append("  %-5s %5d %-7s %-7s objs  : %s"
                     % (r["label"], r["n"], r["weights"], r["range"],
                        " ".join("none" if o is None else "%.1f" % o
                                 for o in r["objs"])))
        lines.append("  %-5s %5d %-7s %-7s times : %s"
                     % (r["label"], r["n"], r["weights"], r["range"],
                        " ".join("none" if t is None else "%.4f" % t
                                 for t in r["times"])))
    lines.append("#")
    lines.append("# summary, averaged over the configurations of the scale")
    lines.append("# %-8s %14s %6s %13s"
                 % ("method", "configurations", "opt", "mean_time(s)"))
    for s in summary:
        lines.append("  %-8s %14d %6d %13.4f"
                     % (s["label"], s["configurations"], s["opt"],
                        s["mean_time"]))


def write_text(path, by_scale, methods, args):
    """The file holding the scale that was run."""
    lines = ["# " + h for h in _header(methods, args)]
    for records, summary in by_scale.values():
        _scale_block(lines, records, summary, args)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path, by_scale, methods, args):
    rows = []
    for scale, (records, _) in by_scale.items():
        for r in records:
            rows.append(dict(r, scale=scale))
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write("# " + "; ".join(_header(methods, args)) + "\n")
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the small-scale part of the F2 || WCmax algorithm "
                    "comparison of paper Section 6 and record the results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[-1],
    )
    p.add_argument("--algos", nargs="+", default=None,
                   help="methods to run, space- or comma-separated, or 'all' "
                        "(default: both methods the paper reports at this scale, "
                        "which are listed by --list)")
    p.add_argument("--instances", type=int, default=None,
                   help="instances per configuration (default: %d)" % INSTANCES)
    p.add_argument("--seed", type=int, default=42, help="base random seed")
    p.add_argument("--time-limit", type=float, default=600.0,
                   help="solver time limit in seconds for MILP / DP "
                        "(default: 600, the limit of paper Section 6)")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help="results file (default: %s; a .csv suffix writes CSV)"
                        % DEFAULT_OUT)
    p.add_argument("--list", action="store_true",
                   help="list the available methods and configurations, then exit")
    return p.parse_args(argv)


def resolve_methods(spec):
    """
    The methods named on the command line, or None when the scale decides.

    None (no --algos) and "all" both mean "the methods the paper reports at each
    scale".
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = [spec]
    names = [t for chunk in spec for t in chunk.replace(",", " ").split() if t]
    if len(names) == 1 and names[0].lower() == "all":
        return None
    unknown = [t for t in names if t not in METHODS]
    if unknown:
        raise SystemExit("unknown method(s): %s\navailable: %s"
                         % (", ".join(unknown), ", ".join(METHODS)))
    return names


def main(argv=None):
    args = parse_args(argv)

    # The run takes hours and its output is often redirected to a file, where
    # stdout would otherwise be block-buffered and show nothing until the end,
    # so it is put into line-buffered mode: every line appears as it is printed.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):        # older interpreters, closed pipe
        pass

    if args.list:
        print("scale %s:" % SCALE)
        print("methods:")
        for name, (label, _) in METHODS.items():
            print("  %-8s %s" % (name, label))
        print("configurations:")
        for n, wt, rg in CONFIGS:
            print("  n=%-4d %-7s %-7s"
                  % (n, config_of(n, wt, rg)["weight_label"], rg))
        return

    methods = resolve_methods(args.algos) or list(METHODS_OF_SCALE)
    out_path = Path(args.out)

    # The per-instance records of the run go into one file, whose name carries
    # the scale and the time the run started.  Every method appends to it as each
    # of its instances finishes, so a run that is interrupted here leaves behind
    # everything it has already solved.
    started = datetime.now()
    stamp = started.strftime("%Y-%m-%d_%H-%M-%S")
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    args.record_path = RECORD_DIR / ("实例记录_%s_%s.txt" % (SCALE, stamp))

    print("F2 || WCmax -- computational study (paper Section 6)")
    print("  scale     : %s" % SCALE)
    print("  methods   : %s" % ", ".join(METHODS[m][0] for m in methods))
    print("  instances : %d per configuration" % (args.instances or INSTANCES))
    print("  time limit: %s s" % args.time_limit)
    print("  output    : %s" % out_path)
    print("  archive   : %s" % ARCHIVE_DIR)
    print("  records   : %s" % args.record_path)
    print(flush=True)

    print("=== scale %s ===" % SCALE)
    records = run_scale(methods, args)
    if not records:
        raise SystemExit("no results produced")
    by_scale = {SCALE: (records, summarise(records))}

    # Every run is kept: the file named by --out holds this run, and a copy that
    # carries the date and the scale is filed away so that no run overwrites an
    # earlier one.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    # named date first, so that a listing of the folder reads chronologically
    archive_path = ARCHIVE_DIR / ("%s_%s%s"
                                  % (stamp, SCALE, out_path.suffix or ".txt"))
    for path in (out_path, archive_path):
        if path.suffix.lower() == ".csv":
            write_csv(path, by_scale, methods, args)
        else:
            write_text(path, by_scale, methods, args)

    print()
    print("summary for %s (averaged over its configurations):" % SCALE)
    print("  %-8s %14s %6s %13s"
          % ("method", "configurations", "opt", "mean_time(s)"))
    for s in summarise(records):
        print("  %-8s %14d %6d %13.4f"
              % (s["label"], s["configurations"], s["opt"], s["mean_time"]))
    print()
    print("results written to %s" % out_path)
    print("archived as        %s" % archive_path)


if __name__ == "__main__":
    main()
