# -*- coding: utf-8 -*-
"""
Driver for the small scale of the computational study of paper Section 6.

This is run_experiments.py for the small scale alone, with the same code and the
same output; only the configurations and the methods below differ, so that the
scale can be run, interrupted and repeated on its own.

* **Instances.**  The small scale varies two factors.  The weights are the
  geometric ladder 1, 10, ..., 10^(K-1), whose length K the configuration fixes
  at 2 and 3, and the processing times are drawn from the ranges [1,20], [1,40]
  and [1,60], which both weight counts take.  The two are crossed with three
  instance sizes, and each of the 18 configurations is replicated 20 times.

* **Methods.**  MILP and DP, the two exact methods of the paper, which are still
  within reach at this scale.  Neither of them is given a gap: the reference of
  this scale is the value of the model itself, so what is reported for each of
  them is how many instances it settled to optimality and how long it took.
  Every method runs once on every instance, so no comparison rests on a
  different sample.

* **Parameters.**  Those of paper Section 6: processing times U(1,20), U(1,40)
  and U(1,60); a time limit of 600 s shared by MILP and DP; and Gurobi on a
  single thread.

* **Best count.**  Once both methods of a configuration have run, the best value
  found on each instance is determined.  The model is one of the two, so that
  value is the optimum of the instance, and the count of instances on which a
  method reached it is the number of instances it solved to optimality.  A
  configuration in which a method established no optimum at all is marked
  "none".

* **Seed.**  The seed of a configuration depends only on (n, K, range), so every
  method sees literally the same instances.

* **Output.**  The run is reported line by line as it happens: one line per
  instance, carrying the value found, whether the method established the optimum
  of that instance, the running time of that instance and the time elapsed since
  the method started on the configuration, and one line per configuration once
  its methods have all run.  stdout is put in line-buffered mode, so the same
  holds when the output is redirected to a file.

The run is written to the results file, and a copy is filed away under the date
and the scale, so that running the script again never overwrites the results of
an earlier run::

    论文/F2/代码/记录/结果_small_最新.txt               <- the run that just finished
    论文/F2/代码/记录/结果存档/<date>_small.txt          <- one file per run, kept
    论文/F2/代码/记录/实例记录_<scales>_<date>.txt       <- one line per instance, live
    论文/F2/代码/run_small.py                         <- this script
    论文/F2/代码/{milp,dp}.py                         <- the two exact methods

Usage
-----
    python run_small.py                             # the whole scale
    python run_small.py --list                      # what would be run
    python run_small.py --algos milp                # one method
    python run_small.py --instances 2               # a quick check
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
#: everything the runs produce is written here, and nowhere else: the results of
#: the run that has just finished, the kept copy of every earlier run, and the
#: per-instance records
RECORD_DIR = HERE / "记录"
#: default results file: holds the run that has just finished, and is overwritten
#: by the next one -- the kept copies are the archive below
DEFAULT_OUT = RECORD_DIR / "结果_small_最新.txt"
#: every run is also filed here under its date and scales, so that no run
#: overwrites the results of an earlier one
ARCHIVE_DIR = RECORD_DIR / "结果存档"

#: how the baseline of a configuration is written in the output: "milp" is the
#: value of the model itself, "base" the reference value of the instance
BASELINE_LABEL = {"milp": "MILP", "base": "LB"}

#: the scale this file runs; there is only one, so the name is a constant
SCALE = "small"

#: the processing-time ranges run at each weight count: both counts take the
#: three of them, the widest included
RANGES_OF_K = {2: (20, 40, 60), 3: (20, 40, 60)}

#: The configuration of this scale, and the only place where it is written down:
#: three instance sizes crossed with two weight counts and three processing-time
#: ranges, 20 instances each, for the 18 configurations and 360 instances of the
#: scale.  "configs" holds one (n, K, processing-time range) triple per
#: configuration; "instances" is the replication count; "methods" the two exact
#: methods; and "baseline" the value every instance of the scale is measured
#: against, which is the model itself.
SCALES = {
    "small": {
        "instances": 20,
        "methods": ("milp", "dp"),
        "baseline": "milp",
        "configs": tuple((n, K, proc_hi)
                         for K in (2, 3)          # K first, as in the runs
                         for n in (20, 40, 60)
                         for proc_hi in RANGES_OF_K[K]),
    },
}


# ---------------------------------------------------------------------------
# runners: one thin adapter per algorithm, each calling that module's benchmark
# ---------------------------------------------------------------------------

def _run_milp(n, cfg, instances, seed, opts, progress=None):
    from milp import benchmark
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed,
                     weights=cfg["weights"], geometric=cfg["geometric"],
                     proc_hi=cfg["proc_hi"],
                     time_limit=opts.time_limit, progress=progress,
                     log_path=getattr(opts, "record_path", None))


def _run_dp(n, cfg, instances, seed, opts, progress=None):
    from dp import benchmark
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed,
                     weights=cfg["weights"], geometric=cfg["geometric"],
                     proc_hi=cfg["proc_hi"],
                     time_limit=opts.time_limit, progress=progress,
                     log_path=getattr(opts, "record_path", None))


def _run_neh(n, cfg, instances, seed, opts, progress=None):
    from NEH import benchmark
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed,
                     weights=cfg["weights"], geometric=cfg["geometric"],
                     progress=progress,
                     log_path=getattr(opts, "record_path", None))


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
                     weights=cfg["weights"], geometric=cfg["geometric"],
                     progress=progress,
                     log_path=getattr(opts, "record_path", None),
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
                     weights=cfg["weights"], geometric=cfg["geometric"],
                     proc_hi=cfg["proc_hi"],
                     progress=progress, log_path=getattr(opts, "record_path", None),
                     **kwargs)


# ---------------------------------------------------------------------------
# settings of the two metaheuristics
# ---------------------------------------------------------------------------
#
# The parameters of paper Section 6, fixed by the calibration run that paragraph
# reports.  A quicker check of the pipeline is a matter of the command line --
# fewer instances (--instances), a single scale (--scale), or a smaller budget
# for a method (--aco-ants, --aco-iter, --ga-pop, --ga-gen), which override what
# is below:
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


#: output order of the methods, and the runner of each.  The baseline of the
#: heuristic scale is not among them: it is the reference value of the schedule a
#: method reports, which each algorithm computes on the instance it is handed
#: (see instance_baseline() in the algorithm modules).
METHODS = {
    "milp": ("MILP", _run_milp),
    "dp":   ("DP",   _run_dp),
    "neh":  ("NEH",  _run_neh),
    "aco":  ("ACO",  _run_aco),
    "ga":   ("GA",   _run_ga),
}
#: the scale run when none is named: this file has only the one
DEFAULT_SCALES = [SCALE]


# ---------------------------------------------------------------------------
# experiment driver
# ---------------------------------------------------------------------------

def config_of(n, K, proc_hi):
    """
    The arguments of one configuration, and the labels its row is written with.

    K is the number of distinct weights, drawn from the geometric ladder
    {1, 10, ..., 10^(K-1)} -- the rule of paper Section 6.1 for this scale, which
    keeps the classes decades apart rather than neighbouring integers -- and
    proc_hi the upper end of the range the processing times are drawn from.
    """
    return {
        "K": K,
        "weights": "narrow",
        "geometric": True,
        "proc_hi": proc_hi,
        "weight_label": "K=%d" % K,
        "range_label": "[1,%d]" % proc_hi,
    }


def seed_of(n, K, proc_hi, base):
    """Seed of a configuration; depends only on its settings, never on the method."""
    return base + 1000 * n + 100 * K + proc_hi


def _hms(seconds):
    """Seconds as h:mm:ss, or m:ss below an hour."""
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return ("%d:%02d:%02d" % (h, m, s)) if h else ("%d:%02d" % (m, s))


#: what the per-instance line says about optimality: True when the method closed
#: the instance, False when it ran into the time limit, None when the method is a
#: heuristic and proves nothing
STATUS = {True: "optimal", False: "time limit", None: "-"}

#: separates one configuration from the next in the live output, so that each
#: block of instances stands apart in a long run
RULE = "-" * 100


def _record(path, line):
    """
    Append one line of the live output to the record file of the run.

    The file then reads as the terminal did: the header of a configuration, the
    line of each of its instances -- written by the algorithm itself, with the
    seed and without the elapsed column -- and the summary line of the
    configuration.  The line is written word for word as it is printed, so that
    the file and the terminal can be read side by side.  `path` may be None, in
    which case the run keeps no record.
    """
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _instance_reporter(label, n, weights, rlabel, stream):
    """
    One line per instance, printed as soon as that instance is done.

    A configuration can take an hour or more -- the ant colony optimization at
    n = 700 runs for a couple of minutes on each of twenty instances -- so the
    driver would otherwise be silent for the whole of it.  The reporter carries
    the running time of the instance and the time elapsed since the method
    started on this configuration.  A method may also hand it the reference of
    the instance; on this scale every method is measured against the value of the
    model instead, so no gap is printed here and the reference is ignored.
    """
    started = time.perf_counter()

    def report(done, total, obj, seconds, proved=None, base=None):
        print("      %-4s n=%-4d %-6s %-7s | instance %2d/%-2d | value %-12s"
              " | %-10s %9s | elapsed %s"
              % (label, n, weights, rlabel, done, total,
                 "none" if obj is None else "%.1f" % obj,
                 STATUS.get(proved, "-"),
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


def run_scale(scale, methods, args, stream=sys.stdout):
    """
    Run every selected method on every configuration of `scale`.

    Returns a list of records, one per (method, n, weights):
        algo, n, weights, instances, gaps, mean_gap, min_gap, max_gap,
        mean_time, missing, hit
    """
    spec = SCALES[scale]
    instances = args.instances or spec["instances"]
    total = len(spec["configs"]) * len(methods)
    records, done = [], 0
    # The record file of the run, written by the algorithms instance by instance
    # and by this driver for the header and the summary of each configuration,
    # so that it reads as the live output did.
    record_path = getattr(args, "record_path", None)

    # The order of the runs follows the order of the tables of Section 6.2: the
    # methods one after the other, MILP first, and inside a method the weight
    # counts one after the other, K = 2 before K = 3.  Every configuration is
    # therefore run once by MILP and, later on, once by DP, the two of them from
    # the same seed and hence on the same instances.
    state = {}
    for name in methods:                    # all of MILP, then all of DP
        label, runner = METHODS[name]
        for K in sorted({k for _, k, _ in spec["configs"]}):   # K = 2, then K = 3
            for (n, k, proc_hi) in spec["configs"]:
                if k != K:
                    continue
                cfg = config_of(n, k, proc_hi)
                seed = seed_of(n, k, proc_hi, args.seed)
                wlabel, rlabel = cfg["weight_label"], cfg["range_label"]
                if done:                  # separate the configurations
                    print(RULE, file=stream, flush=True)
                head = ("  n=%-4d %-6s %-7s | %s (seed %d)"
                        % (n, wlabel, rlabel, label, seed))
                print(head, file=stream, flush=True)
                _record(record_path, head)
                res = runner(n, cfg, instances, seed, args,
                             _instance_reporter(label, n, wlabel, rlabel, stream))
                state[(name, n, k, proc_hi)] = res

                missing = sum(1 for o in res["objs"] if o is None)
                # Both methods of this scale are exact, so the instances they
                # settle to optimality are counted directly: the model reports
                # those whose search it closed, the dynamic program every
                # instance it finished.
                opt = res.get("optimal", res.get("solved", 0))
                # The objective values and the running times of the
                # configuration, by their smallest, their mean and their largest.
                vals = [o for o in res["objs"] if o is not None]
                secs = [t for t in res["times"] if t is not None]
                stat_obj = ((statistics.fmean(vals), min(vals), max(vals))
                            if vals else None)
                stat_time = ((statistics.fmean(secs), min(secs), max(secs))
                             if secs else None)
                rec = {
                    "algo": name,
                    "label": label,
                    "n": n,
                    "K": k,
                    "proc_hi": proc_hi,
                    "range": rlabel,
                    "instances": instances,
                    "missing": missing,
                    "opt": opt,
                    "obj": stat_obj,
                    "time": stat_time,
                    "mean_time": res["mean_time"],
                    # the raw per-instance values, so that the file still holds
                    # the run after the driver has moved on: another reading of
                    # it can then be applied without repeating the run
                    "objs": res["objs"],
                    "times": res["times"],
                }
                records.append(rec)
                done += 1
                o_mean, o_min, o_max = stat_obj or (float("nan"),) * 3
                t_mean, t_min, t_max = stat_time or (float("nan"),) * 3
                # The summary line of the configuration, written to the record
                # file as it is printed, so that the file carries the totals of
                # every group of instances as well as the instances themselves.
                summary = ("  [%2d/%2d] %-4s n=%-4d %-6s %-7s | opt %-7s "
                           "| obj %9.1f (%8.1f, %9.1f) "
                           "| time %9.4f s (%8.4f, %10.4f) | no value %d"
                           % (done, total, label, n, wlabel, rlabel,
                              "%d/%d" % (opt, instances) if opt else "none",
                              o_mean, o_min, o_max, t_mean, t_min, t_max, missing))
                print(summary, file=stream, flush=True)
                _record(record_path, summary)

    # The best value found on each instance, now that every method has run on it.
    # The model is one of them, so that value is the optimum of the instance.
    for (n, k, proc_hi) in spec["configs"]:
        best = []
        for i in range(instances):
            vals = [state[(m, n, k, proc_hi)]["objs"][i] for m in methods
                    if (m, n, k, proc_hi) in state
                    and state[(m, n, k, proc_hi)]["objs"][i] is not None]
            best.append(min(vals) if vals else None)
        for rec in records:
            if (rec["n"], rec["K"], rec["proc_hi"]) == (n, k, proc_hi):
                rec["best_of_instance"] = best

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
            "opt": sum(r["opt"] for r in recs),
            "mean_time": statistics.fmean(r["mean_time"] for r in recs),
        })
    return out


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

def _header(scales, methods, args):
    return [
        "F2 || WCmax -- computational study (paper Section 6)",
        "generated     : %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scales        : %s" % ", ".join(scales),
        "methods       : %s" % "; ".join(
            "%s: %s" % (s, ", ".join(METHODS[m][0] for m in SCALES[s]["methods"]))
            for s in scales),
        "instances/cfg : %d" % (args.instances or 20),
        "baseline      : %s, the value of the model itself"
        % BASELINE_LABEL["milp"],
        "seed          : %d (+1000*n, + 100*K, + range)" % args.seed,
        "time limit    : %s s" % (args.time_limit if args.time_limit else "none"),
        "records       : %s" % getattr(args, "record_path", "-"),
    ]


def _scale_block(lines, scale, records, summary, args):
    spec = SCALES[scale]
    lines.append("")
    lines.append("# " + "=" * 84)
    lines.append("# scale %s  --  configurations: %s"
                 % (scale, ", ".join("n=%d/K=%d/[1,%d]" % (n, K, proc_hi)
                                     for n, K, proc_hi in spec["configs"])))
    lines.append("# " + "=" * 84)
    lines.append("# per-configuration results")
    lines.append("# %-5s %5s %-7s %-7s %9s %8s %23s %31s"
                 % ("algo", "n", "weights", "range", "instances", "opt",
                    "objective mean (min, max)", "time(s) mean (min, max)"))
    for r in records:
        o = r["obj"] or (float("nan"),) * 3
        t = r["time"] or (float("nan"),) * 3
        lines.append("  %-5s %5d %-7s %-7s %9d %8s %10.1f (%8.1f, %9.1f)"
                     " %12.4f (%8.4f, %10.4f)"
                     % (r["label"], r["n"], "K=%d" % r["K"], r["range"],
                        r["instances"],
                        "%d/%d" % (r["opt"], r["instances"]) if r["opt"] else "none",
                        o[0], o[1], o[2], t[0], t[1], t[2]))
    lines.append("#")
    lines.append("# the column 'opt' counts the instances, out of those of the configuration,")
    lines.append("# whose optimum the method itself established: the model counts the ones")
    lines.append("# whose search it closed, the dynamic program every one it finished, being")
    lines.append("# exact.  'none' marks a configuration in which the method established no")
    lines.append("# optimum at all.")
    lines.append("#")
    lines.append("# per-instance objective values and CPU times, in instance order")
    seen = set()
    for r in records:
        key = (r["n"], r["K"], r["range"])
        if key not in seen:            # the same for every method of the configuration
            seen.add(key)
            lines.append("  %-5s %5d %-7s %-7s best  : %s"
                         % ("-", r["n"], "K=%d" % r["K"], r["range"],
                            " ".join("none" if b is None else "%.1f" % b
                                     for b in r["best_of_instance"])))
        lines.append("  %-5s %5d %-7s %-7s objs  : %s"
                     % (r["label"], r["n"], "K=%d" % r["K"], r["range"],
                        " ".join("none" if o is None else "%.1f" % o
                                 for o in r["objs"])))
        lines.append("  %-5s %5d %-7s %-7s times : %s"
                     % (r["label"], r["n"], "K=%d" % r["K"], r["range"],
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
    """One file holding every scale that was run, in the order given."""
    lines = ["# " + h for h in _header(list(by_scale), methods, args)]
    for scale, (records, summary) in by_scale.items():
        _scale_block(lines, scale, records, summary, args)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path, by_scale, methods, args):
    rows = []
    for scale, (records, _) in by_scale.items():
        for r in records:
            rows.append(dict(r, scale=scale))
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write("# " + "; ".join(_header(list(by_scale), methods, args)) + "\n")
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the F2 || WCmax algorithm comparison of paper "
                    "Section 6 and record the results in a single file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[-1],
    )
    p.add_argument("--scale", nargs="+", choices=sorted(SCALES), default=None,
                   help="instance scale(s); several may be given and are written "
                        "to the same file (default: %s)"
                        % ",".join(DEFAULT_SCALES))
    p.add_argument("--algos", nargs="+", default=None,
                   help="methods to run, space- or comma-separated, or 'all' "
                        "(default: the methods the paper reports at each scale, "
                        "which are listed by --list)")
    p.add_argument("--instances", type=int, default=None,
                   help="instances per configuration (default: 20, or 5 for the "
                        "pilot tests)")
    p.add_argument("--seed", type=int, default=42, help="base random seed")
    p.add_argument("--time-limit", type=float, default=600.0,
                   help="solver time limit in seconds for MILP / DP "
                        "(default: 600, the limit of paper Section 6)")
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
    """
    The methods named on the command line, or None when the scale decides.

    None (no --algos) and "all" both mean "the methods the paper reports at each
    scale", which differ from one scale to the next; the baseline is never among
    them, as it is a reference value and not a method, and it is computed anyway
    wherever it serves as a denominator.
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
        print("methods:")
        for name, (label, _) in METHODS.items():
            print("  %-8s %s" % (name, label))
        print("scale %s:" % SCALE)
        print("methods:")
        for name in SCALES[SCALE]["methods"]:
            print("  %-8s %s" % (name, METHODS[name][0]))
        print("configurations:")
        for n, K, proc_hi in SCALES[SCALE]["configs"]:
            print("  n=%-4d %-7s %-7s"
                  % (n, "K=%d" % K, "[1,%d]" % proc_hi))
        return

    methods = resolve_methods(args.algos)
    scales = args.scale or list(DEFAULT_SCALES)
    # Without an explicit --algos, every scale runs the methods the paper reports
    # there; with one, the same list is run at every scale.
    methods_by_scale = {s: (list(methods) if methods
                            else list(SCALES[s]["methods"])) for s in scales}
    out_path = Path(args.out)

    # The per-instance records of the whole run go into one file, whose name
    # carries the scales and the time the run started.  Every method appends to
    # it as each of its instances finishes, so a run that is interrupted here
    # leaves behind everything it has already solved.
    started = datetime.now()
    stamp = started.strftime("%Y-%m-%d_%H-%M-%S")
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    args.record_path = RECORD_DIR / ("实例记录_%s_%s.txt"
                                     % ("-".join(scales), stamp))

    print("F2 || WCmax -- computational study (paper Section 6)")
    print("  scales    : %s" % ", ".join(scales))
    for s in scales:
        print("  methods   : %-7s %s"
              % (s, ", ".join(METHODS[m][0] for m in methods_by_scale[s])))
    print("  instances : %d per configuration" % (args.instances or 20))
    print("  time limit: %s s" % args.time_limit)
    print("  output    : %s" % out_path)
    print("  archive   : %s" % ARCHIVE_DIR)
    print("  records   : %s" % args.record_path)
    print(flush=True)

    by_scale = {}
    for scale in scales:
        print("=== scale %s ===" % scale)
        records = run_scale(scale, methods_by_scale[scale], args)
        if not records:
            print("  no results produced for this scale", file=sys.stderr)
            continue
        by_scale[scale] = (records, summarise(records))

    if not by_scale:
        raise SystemExit("no results produced")

    # Every run is kept: the file named by --out holds this run, and a copy that
    # carries the date and the scales is filed away so that no run overwrites an
    # earlier one.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    # named date first, so that a listing of the folder reads chronologically
    archive_path = ARCHIVE_DIR / ("%s_%s%s"
                                  % (stamp, "-".join(by_scale),
                                     out_path.suffix or ".txt"))
    for path in (out_path, archive_path):
        if path.suffix.lower() == ".csv":
            write_csv(path, by_scale, methods, args)
        else:
            write_text(path, by_scale, methods, args)

    for scale, (_, summary) in by_scale.items():
        print()
        print("summary for %s (total opt over its configurations):" % scale)
        print("  %-8s %14s %6s %13s"
              % ("method", "configurations", "opt", "mean_time(s)"))
        for s in summary:
            print("  %-8s %14d %6d %13.4f"
                  % (s["label"], s["configurations"], s["opt"], s["mean_time"]))
    print()
    print("results written to %s" % out_path)
    print("archived as        %s" % archive_path)


if __name__ == "__main__":
    main()
