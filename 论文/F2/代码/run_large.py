# -*- coding: utf-8 -*-
"""
Driver for the **large** scale of the computational study of paper Section 6.

This is the large-scale third of the study: the small scale is run by
``run_small.py`` and the medium one by ``run_medium.py``.  The three scripts share
the driver below and differ only in the configurations and the methods they run,
so a scale can be run, interrupted and repeated on its own.

The script follows the design of paper Section 6 rather than inventing its own:

* **Instances.**  The large scale of paper Table 1 varies the instance size, at
  a single processing-time range of [1,10], and places no bound on the weights:
  they are drawn uniformly from {1, ..., n_jobs}, which is why paper Table 1
  records them as "all weights".  The three instance sizes give 3 configurations,
  each of them replicated 20 times.

* **Methods.**  NEH, ACO and GA -- the three heuristics of the paper, the exact
  methods being out of reach at this size.  The baseline is not a method: it is
  the reference value W_max * sum_j (a_j + b_j), which returns no schedule and is
  never run.  Each algorithm computes it on the instance it has just been handed
  (`instance_baseline()` of the algorithm modules), so nothing extra is run and
  the denominator is always the value of the very instance that was solved.
  Being a lower bound on the optimum, it makes every gap of this scale positive,
  and the smaller the better.

* **Parameters.**  Those of paper Section 6, fixed by the calibration run that
  paragraph reports and held in the two constants below: ACO with 10 ants,
  alpha=1, beta=30, rho=0.1 and 100 iterations; GA with N=n individuals, G=10
  generations, p_c=0.9, p_m=0.1 and an elitism size of N/10.

* **Gap.**  The quantity reported is the relative percentage gap of a method
  against the reference value of the configuration, computed **per instance** and
  then aggregated, so the mean, the minimum and the maximum below are the
  statistics of the same 20 gaps.  A method that produces no value on an
  instance contributes no gap and is counted separately.

* **Best count.**  Once every method of a configuration has run, the best value
  found on each instance by any of them is determined, and each method reports
  the number of instances on which it attains that value.

* **Seed.**  The seed of a configuration depends only on the configuration and
  never on the method, so the three methods see literally the same instances and
  the gaps are paired.

* **Output.**  The run is reported line by line as it happens: one line per
  instance, carrying the value found, the running time of that instance and the
  time elapsed since the method started on the configuration, and one line per
  configuration once its methods have all run.  stdout is put in line-buffered
  mode, so the same holds when the output is redirected to a file.

The run is written to the large-scale results file, and a copy is filed away
under the date, so that running the script again never overwrites the results of
an earlier run::

    论文/F2/代码/记录/结果_large_最新.txt             <- the run that just finished
    论文/F2/代码/记录/结果存档/<date>_large.txt       <- one file per run, kept
    论文/F2/代码/记录/实例记录_large_<date>.txt       <- one line per instance, live
    论文/F2/代码/run_large.py                        <- this script
    论文/F2/代码/{NEH,ACO,GA}.py                      <- the three heuristics

Usage
-----
    python run_large.py                      # the whole scale
    python run_large.py --list               # what would be run
    python run_large.py --algos neh ga       # two of the three
    python run_large.py --instances 2        # a quick check
    python run_large.py --aco-ants 4 --aco-iter 20 --ga-gen 3
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
DEFAULT_OUT = RECORD_DIR / "结果_large_最新.txt"
#: every run is also filed here under its date, so that no run overwrites the
#: results of an earlier one
ARCHIVE_DIR = RECORD_DIR / "结果存档"

#: weight settings of paper Table 1.  A setting named K<k> is the narrow regime,
#: which draws the weights uniformly from {1, ..., k}, so that k is the number of
#: values they can take and the number the tables of the paper label the row
#: with.  "free" is the unrestricted regime of this scale, which draws them
#: uniformly from {1, ..., n_jobs} instead; a setting named K<k> is used by
#: run_medium.py rather than here.
WEIGHT_SETTINGS = ("free",)

#: processing-time ranges of the scale: the label of the table row, and the upper
#: end of the discrete uniform range both processing times are drawn from.  This
#: scale is run at the narrow range only.
PROC_RANGES = {"[1,10]": 10}

#: the baseline of every configuration of this scale is the reference value
#: W_max * sum_j (a_j + b_j), which is a property of the instance rather than a
#: method and is never run
BASELINE = "base"

#: how a baseline is written in the output
BASELINE_LABEL = {"milp": "MILP", "base": "W_max*sum(a_j+b_j)"}

#: the scale this script runs, and the only place where it is written down
SCALE = "large"
#: instances per configuration
INSTANCES = 20
#: the methods the paper reports at this scale
METHODS_OF_SCALE = ("neh", "aco", "ga")
#: the configurations of paper Section 6 and Table 1: the three instance sizes
#: crossed with the three weight settings, at the single range of the scale
CONFIGS = tuple((n, wt, rg)
                for n in (500, 600, 700)
                for wt in WEIGHT_SETTINGS
                for rg in ("[1,10]",))


def config_of(n, weights, rng):
    """
    The arguments of one configuration, and the labels its row is written with.

    A setting named K<k> is the narrow regime, which draws the weights uniformly
    from {1, ..., k}, so that k is the number of values they can take; "free" is
    the unrestricted regime of the large scale, which draws them uniformly from
    {1, ..., n_jobs} and so bounds them by the number of jobs rather than by a
    fixed count.
    """
    if weights == "free":
        return {"n": n, "K": n, "weights": "free",
                "proc_hi": PROC_RANGES[rng],
                "weight_label": "all weights", "range_label": rng}
    return {"n": n, "K": int(weights[1:]), "weights": "narrow",
            "proc_hi": PROC_RANGES[rng],
            "weight_label": "K=" + weights[1:], "range_label": rng}


# ---------------------------------------------------------------------------
# runners: one thin adapter per algorithm, each calling that module's benchmark
# ---------------------------------------------------------------------------

def _run_neh(n, cfg, instances, seed, opts, progress=None):
    from NEH import benchmark
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed,
                     weights=cfg["weights"], proc_hi=cfg["proc_hi"],
                     progress=progress, log_path=getattr(opts, "record_path", None))


def _run_aco(n, cfg, instances, seed, opts, progress=None):
    from ACO import benchmark
    kwargs = dict(ACO_SETTINGS)                 # test mode, or the paper values
    if opts.aco_ants is not None:
        kwargs["m"] = opts.aco_ants
    if opts.aco_iter is not None:
        kwargs["max_iter"] = opts.aco_iter
    if opts.aco_rho is not None:
        kwargs["rho"] = opts.aco_rho
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed,
                     weights=cfg["weights"], proc_hi=cfg["proc_hi"],
                     progress=progress, log_path=getattr(opts, "record_path", None),
                     **kwargs)


def _run_ga(n, cfg, instances, seed, opts, progress=None):
    from GA import benchmark
    kwargs = {"N": max(8, GA_SETTINGS["N_factor"] * n),   # test mode, or the paper
              "G": GA_SETTINGS["G"]}                      # values of Section 6
    if opts.ga_pop is not None:
        kwargs["N"] = opts.ga_pop
    if opts.ga_gen is not None:
        kwargs["G"] = opts.ga_gen
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed,
                     weights=cfg["weights"], proc_hi=cfg["proc_hi"],
                     progress=progress, log_path=getattr(opts, "record_path", None),
                     **kwargs)


# ---------------------------------------------------------------------------
# settings of the two metaheuristics
# ---------------------------------------------------------------------------
#
# The parameters of paper Section 6, fixed by the calibration run that paragraph
# reports.  A quicker check of the pipeline is a matter of the command line --
# fewer instances (--instances), or a smaller budget for a method (--aco-ants,
# --aco-iter, --aco-rho, --ga-pop, --ga-gen), which override what is below:
#
#   GA   N = n, G = 10 reproduces the mean objective of N = 2n, G = 30 on both
#        weight regimes, at a quarter of the running time: the population reaches
#        its fixed point within a few generations, and even N = n/2, G = 10
#        still matches.
#   ACO  at an equal budget of m * max_iter = 1000, ten ants for a hundred
#        iterations beat twenty for fifty -- 1718.7 against 1734.7 on the
#        independent setting and 5018.7 against 5175.0 on the correlated one --
#        so the budget is spent on iterations rather than on ants.  Against
#        fifty ants and a hundred iterations, at five times the cost, ten ants
#        cost 1.7 and 2.8 per cent of mean objective.
#: ACO arguments: colony size, and iterations of the stopping criterion
ACO_SETTINGS = {"m": 10, "max_iter": 100}
#: GA arguments: population size as a multiple of n, and generations
GA_SETTINGS = {"N_factor": 1, "G": 10}


#: output order of the methods, and the runner of each.  The baseline is not
#: among them: it is a property of the instance rather than a method, and each
#: algorithm computes it on the instance it is handed (see instance_baseline()
#: in the algorithm modules).
METHODS = {
    "neh":  ("NEH",  _run_neh),
    "aco":  ("ACO",  _run_aco),
    "ga":   ("GA",   _run_ga),
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

    A configuration can take an hour or more -- the ant colony optimization at
    the largest size of the scale runs for a couple of minutes on each of twenty
    instances -- so the driver would otherwise be silent for the whole of it.
    The reporter carries the running time of the instance and the time elapsed
    since the method started on this configuration.
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


def gaps_against(objs, base_objs):
    """
    Per-instance percentage gaps of `objs` against `base_objs`.

    Both lists are indexed by instance, so the pairing is exact.  An instance on
    which either side produced no value contributes no gap and is reported as
    missing instead.
    """
    out, missing = [], 0
    for o, b in zip(objs, base_objs):
        if o is None or b is None or b == 0:
            missing += 1
            continue
        out.append(100.0 * (o - b) / b)
    return out, missing


def run_scale(methods, args, stream=sys.stdout):
    """
    Run every selected method on every configuration of the scale.

    Returns a list of records, one per (method, n, weights, range):
        algo, n, weights, range, instances, gaps, mean_gap, min_gap, max_gap,
        mean_time, missing, hit
    """
    instances = args.instances or INSTANCES
    total = len(CONFIGS) * len(methods)
    records, done = [], 0

    for (n, weights, rng) in CONFIGS:
        cfg = config_of(n, weights, rng)
        seed = seed_of(n, weights, rng, args.seed)
        wlabel, rlabel = cfg["weight_label"], cfg["range_label"]

        # The baseline of the configuration is the denominator of every gap of
        # that configuration, and on this scale it is the reference value
        # W_max * sum_j (a_j + b_j), a property of the instance alone: every
        # method computes it from the instance it has just been handed -- see
        # instance_baseline() of the algorithm modules -- so nothing extra has
        # to be run at all.
        print("  n=%-4d %-6s %-7s | baseline %s (seed %d)"
              % (n, wlabel, rlabel, BASELINE_LABEL[BASELINE], seed),
              file=stream, flush=True)

        # Every method of the configuration is run first, because the number of
        # instances on which a method finds the best solution cannot be counted
        # until all of them have reported on the same instances.
        results = {}
        for name in methods:
            label, runner = METHODS[name]
            results[name] = runner(n, cfg, instances, seed, args,
                                   _instance_reporter(label, n, wlabel, stream))

        # The best value found on each instance by any of the methods run here.
        best_of_instance = []
        for i in range(instances):
            vals = [results[m]["objs"][i] for m in methods
                    if results[m]["objs"][i] is not None]
            best_of_instance.append(min(vals) if vals else None)

        for name in methods:
            label = METHODS[name][0]
            res = results[name]
            gaps, missing = gaps_against(res["objs"], res["bases"])
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
                "baseline": BASELINE,
                "instances": instances,
                "paired": len(gaps),
                "missing": missing,
                "mean_gap": statistics.fmean(gaps) if gaps else float("nan"),
                "min_gap": min(gaps) if gaps else float("nan"),
                "max_gap": max(gaps) if gaps else float("nan"),
                "mean_time": res["mean_time"],
                "hit": hit,
                # the raw per-instance values, so that the file still holds the
                # run after the driver has moved on: a change of baseline or of
                # gap formula can then be applied without repeating the run
                "objs": res["objs"],
                "times": res["times"],
                "bases": res["bases"],
                "best_of_instance": best_of_instance,
            }
            records.append(rec)
            done += 1
            print("  [%2d/%2d] %-4s n=%-4d %-6s %-7s | mean gap %+8.3f%% "
                  "| min %+8.3f%% | max %+9.3f%% | %9.4f s | best %2d/%-2d "
                  "| unpaired %d"
                  % (done, total, label, n, wlabel, rlabel, rec["mean_gap"],
                     rec["min_gap"], rec["max_gap"], rec["mean_time"],
                     hit, instances, missing),
                  file=stream, flush=True)

    return records


def summarise(records):
    """Mean gap and mean CPU time of each method, over the configurations of the scale."""
    by_algo = {}
    for rec in records:
        by_algo.setdefault(rec["algo"], []).append(rec)
    out = []
    for name in METHODS:
        recs = [r for r in by_algo.get(name, []) if r["paired"]]
        if not recs:
            continue
        out.append({
            "algo": name,
            "label": METHODS[name][0],
            "configurations": len(recs),
            "mean_gap": statistics.fmean(r["mean_gap"] for r in recs),
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
        "metaheuristics: ACO %s; GA %s"
        % (", ".join("%s=%s" % kv for kv in sorted(ACO_SETTINGS.items())),
           ", ".join("%s=%s" % kv for kv in sorted(GA_SETTINGS.items()))),
        "gap against   : the baseline of each configuration, which on this scale "
                        "is the reference value W_max*sum(a+b)",
        "seed          : %d (+1000*n, + 100*weight setting, + range)" % args.seed,
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
    lines.append("# %-5s %5s %-7s %-7s %9s %9s %10s %10s %9s %9s %6s"
                 % ("algo", "n", "weights", "range", "paired", "unpaired",
                    "mean_gap", "min_gap", "max_gap", "time(s)", "best"))
    for r in records:
        lines.append("  %-5s %5d %-7s %-7s %9d %9d %10.3f %10.3f %9.3f %9.4f %6d"
                     % (r["label"], r["n"], r["weights"], r["range"],
                        r["paired"], r["missing"], r["mean_gap"], r["min_gap"],
                        r["max_gap"], r["mean_time"], r["hit"]))
    lines.append("#")
    lines.append("# the column 'best' counts the instances on which the method attained the")
    lines.append("# best value found on that instance by any method run there")
    lines.append("#")
    lines.append("# per-instance baseline, objective values and CPU times, in instance order")
    seen = set()
    for r in records:
        key = (r["n"], r["weights"], r["range"])
        if key not in seen:            # the same for every method of the configuration
            seen.add(key)
            lines.append("  %-5s %5d %-7s %-7s best  : %s"
                         % ("-", r["n"], r["weights"], r["range"],
                            " ".join("none" if b is None else "%.1f" % b
                                     for b in r["best_of_instance"])))
        lines.append("  %-5s %5d %-7s %-7s base  : %s"
                     % (r["label"], r["n"], r["weights"], r["range"],
                        " ".join("%.1f" % b for b in r["bases"])))
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
    lines.append("# %-8s %14s %11s %13s"
                 % ("method", "configurations", "mean_gap", "mean_time(s)"))
    for s in summary:
        lines.append("  %-8s %14d %11.3f %13.4f"
                     % (s["label"], s["configurations"], s["mean_gap"],
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
        description="Run the large-scale part of the F2 || WCmax algorithm "
                    "comparison of paper Section 6 and record the results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[-1],
    )
    p.add_argument("--algos", nargs="+", default=None,
                   help="methods to run, space- or comma-separated, or 'all' "
                        "(default: the methods the paper reports at this scale, "
                        "which are listed by --list)")
    p.add_argument("--instances", type=int, default=None,
                   help="instances per configuration (default: %d)" % INSTANCES)
    p.add_argument("--seed", type=int, default=42, help="base random seed")
    p.add_argument("--aco-ants", type=int, default=None, help="ACO: number of ants")
    p.add_argument("--aco-iter", type=int, default=None, help="ACO: iterations")
    p.add_argument("--aco-rho", type=float, default=None, help="ACO: evaporation rate")
    p.add_argument("--ga-pop", type=int, default=None, help="GA: population size")
    p.add_argument("--ga-gen", type=int, default=None, help="GA: generations")
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
            print("  n=%-4d %-7s %-7s  baseline %s"
                  % (n, config_of(n, wt, rg)["weight_label"], rg,
                     BASELINE_LABEL[BASELINE]))
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
    print("  metaheur.: ACO %s; GA %s"
          % (", ".join("%s=%s" % kv for kv in sorted(ACO_SETTINGS.items())),
             ", ".join("%s=%s" % kv for kv in sorted(GA_SETTINGS.items()))))
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
    print("  %-8s %14s %11s %13s"
          % ("method", "configurations", "mean_gap", "mean_time(s)"))
    for s in summarise(records):
        print("  %-8s %14d %11.3f %13.4f"
              % (s["label"], s["configurations"], s["mean_gap"], s["mean_time"]))
    print()
    print("results written to %s" % out_path)
    print("archived as        %s" % archive_path)


if __name__ == "__main__":
    main()
