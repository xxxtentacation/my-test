# -*- coding: utf-8 -*-
"""
Master driver for the computational study of paper Section 6 (Experiments).

The driver follows the design of that section rather than inventing its own:

* **Instances.**  Every configuration is a pair (n, weight setting) drawn from
  paper Table 1.  The small scale uses the narrow regime, in which the weights
  are uniform on {1, ..., K} with K = 2; the medium and large scales use the two
  wide regimes, in which the weights fall into a fixed number of classes, 3 in
  the independent setting ("indep") and 9 in the correlated one ("corr"), the
  classes of the latter following the order of a_j + b_j.  The 320 instances of
  the paper are therefore 20 replications of each of the 16 configurations of
  SCALES below.

* **Methods.**  Each scale runs the methods the paper reports there: MILP and DP
  on the small scale, where the exact methods are still within reach, and NEH,
  ACO and GA on the medium and the large scale, where they are not.  The baseline
  is not a method but the reference value of those two scales: it returns no
  schedule, it is never run, and it appears in no table.  Every method runs once
  on every instance, so no comparison rests on a different sample.

* **Parameters.**  Those of paper Section 6, all of them the defaults of the
  algorithm modules or the constants at the top of this file: processing times
  U(1,10); a time limit of 600 s shared by MILP and DP; ACO with 10 ants,
  alpha=1, beta=30, rho=0.1 and 100 iterations; GA with N=n individuals, G=10
  generations, p_c=0.9, p_m=0.1 and an elitism size of N/10; and Gurobi on a
  single thread.

* **Gap.**  The quantity reported is the relative percentage gap of a method
  against the *baseline of the configuration*, which is fixed instance by
  instance: the value of the MILP model on the small configurations, and the
  reference value LB = max{LB_1, LB_2, LB_3} of paper Section 4.3, the largest of
  three prefix bounds on the optimum, on the rest.  The latter is not a schedule but a
  number, and each algorithm computes it on the instance it has just been handed
  (`instance_baseline()` of the algorithm modules), so nothing extra is run and
  the denominator is always the value
  instance that was solved.  Being below the value of the schedule it is read
  from, it makes every
  gap of those two scales non-negative, and the smaller the better.  The gap is
  computed **per instance** and then aggregated, so the mean, the minimum and the
  maximum below are the statistics of the same 20 gaps.  A method that hits the
  time limit without producing a value contributes no gap and is counted
  separately.

* **Best count.**  Once every method of a configuration has run, the best value
  found on each instance by any of them is determined, and each method reports
  the number of instances on which it attains that value.  On the small scale
  the MILP model is one of the methods, so the value is the optimum and the count
  is the number of instances solved to optimality; on the other two scales it is
  the best value known for the instance.

* **Seed.**  The seed of a configuration depends only on (n, weight setting), so
  every method sees literally the same instances and the gaps are paired.

* **Output.**  The run is reported line by line as it happens: one line per
  instance, carrying the value found, the running time of that instance and the
  time elapsed since the method started on the configuration, and one line per
  configuration once its methods have all run.  stdout is put in line-buffered
  mode, so the same holds when the output is redirected to a file.

Every scale that is run is written to the results file, and a copy is filed away
under the date and the scales, so that running the script again never overwrites
the results of an earlier run::

    论文/F2/代码/记录/结果_最新.txt                    <- the run that just finished
    论文/F2/代码/记录/结果存档/<date>_<scales>.txt      <- one file per run, kept
    论文/F2/代码/记录/实例记录_<scales>_<date>.txt      <- one line per instance, live
    论文/F2/代码/run_experiments.py                   <- this script
    论文/F2/代码/{MILP,DP,NEH,ACO,GA}.py              <- the algorithms

Usage
-----
    python run_experiments.py                       # the whole study
    python run_experiments.py --list                # what would be run
    python run_experiments.py --scale small         # one scale
    python run_experiments.py --scale medium --algos neh aco ga
    python run_experiments.py --scale small --instances 2   # a quick check
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
DEFAULT_OUT = RECORD_DIR / "结果_最新.txt"
#: every run is also filed here under its date and scales, so that no run
#: overwrites the results of an earlier one
ARCHIVE_DIR = RECORD_DIR / "结果存档"

#: weight settings of paper Section 6.1.1, in the order used to derive the seed
WEIGHT_SETTINGS = ("narrow", "indep", "corr")

#: how a weight setting is written in the output and in the files: by the number
#: of distinct weights it produces, which is how the tables of the paper label
#: their rows.  The names above remain the internal keys, and the ones the
#: instance generator takes.
WEIGHTS_LABEL = {"narrow": "K=2", "indep": "K=3", "corr": "K=9"}

#: how the baseline of a configuration is written in the output: "milp" is the
#: value of the model itself, "base" the reference value of the instance
BASELINE_LABEL = {"milp": "MILP", "base": "LB"}

#: the processing times of this driver are drawn from one range only, the one the
#: algorithm modules default to; the per-scale drivers vary it and label it per
#: configuration instead
RANGE_LABEL = "[1,10]"

#: The configuration of paper Section 6 and Table 1, and the only place where it
#: is written down: 16 configurations over three scales, 20 instances each, for
#: the 320 instances of the study.  "configs" holds one (n, weight setting,
#: baseline) triple per configuration; "instances" is the replication count of
#: the scale; and "methods" the methods the paper reports there -- the exact ones
#: on the small scale, where they are still within reach, and the three
#: heuristics on the other two, where they are not.  Running this file with no
#: arguments runs all three scales exactly as they stand here.
SCALES = {
    "small": {                  # 4 configurations x 20 =  80 instances, K = 2
        "instances": 20,
        "methods": ("milp", "dp"),
        "configs": tuple((n, "narrow", "milp") for n in (30, 40, 50, 60)),
    },
    "medium": {                 # 6 configurations x 20 = 120 instances, K = 3, 9
        "instances": 20,
        "methods": ("neh", "aco", "ga"),
        "configs": tuple((n, wt, "base")
                         for n in (200, 250, 300) for wt in ("indep", "corr")),
    },
    "large": {                  # 6 configurations x 20 = 120 instances, K = 3, 9
        "instances": 20,
        "methods": ("neh", "aco", "ga"),
        "configs": tuple((n, wt, "base")
                         for n in (500, 600, 700) for wt in ("indep", "corr")),
    },
}


# ---------------------------------------------------------------------------
# runners: one thin adapter per algorithm, each calling that module's benchmark
# ---------------------------------------------------------------------------

def _run_milp(n, weights, instances, seed, opts, progress=None):
    from milp import benchmark
    return benchmark(n=n, K=2, instances=instances, seed=seed, weights=weights,
                     time_limit=opts.time_limit, progress=progress,
                     log_path=getattr(opts, "record_path", None))


def _run_dp(n, weights, instances, seed, opts, progress=None):
    from dp import benchmark
    return benchmark(n=n, K=2, instances=instances, seed=seed, weights=weights,
                     time_limit=opts.time_limit, progress=progress,
                     log_path=getattr(opts, "record_path", None))


def _run_neh(n, weights, instances, seed, opts, progress=None):
    from NEH import benchmark
    return benchmark(n=n, K=2, instances=instances, seed=seed, weights=weights,
                     progress=progress, log_path=getattr(opts, "record_path", None))


def _run_aco(n, weights, instances, seed, opts, progress=None):
    from ACO import benchmark
    kwargs = dict(ACO_SETTINGS)                 # test mode, or the paper values
    if opts.aco_ants is not None:
        kwargs["m"] = opts.aco_ants
    if opts.aco_iter is not None:
        kwargs["max_iter"] = opts.aco_iter
    if opts.aco_rho is not None:
        kwargs["rho"] = opts.aco_rho
    return benchmark(n=n, K=2, instances=instances, seed=seed, weights=weights,
                     progress=progress, log_path=getattr(opts, "record_path", None),
                     **kwargs)


def _run_ga(n, weights, instances, seed, opts, progress=None):
    from GA import benchmark
    kwargs = {"N": max(8, GA_SETTINGS["N_factor"] * n),   # test mode, or the paper
              "G": GA_SETTINGS["G"]}                      # values of Section 6
    if opts.ga_pop is not None:
        kwargs["N"] = opts.ga_pop
    if opts.ga_gen is not None:
        kwargs["G"] = opts.ga_gen
    return benchmark(n=n, K=2, instances=instances, seed=seed, weights=weights,
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
#: medium and large scales is not among them: it is the reference value of the
#: schedule a method reports, which each algorithm computes on the instance it is
#: handed (see instance_baseline() in the algorithm modules).
METHODS = {
    "milp": ("MILP", _run_milp),
    "dp":   ("DP",   _run_dp),
    "neh":  ("NEH",  _run_neh),
    "aco":  ("ACO",  _run_aco),
    "ga":   ("GA",   _run_ga),
}
#: scales run when none is named: the three of paper Table 1
DEFAULT_SCALES = ["small", "medium", "large"]


# ---------------------------------------------------------------------------
# experiment driver
# ---------------------------------------------------------------------------

def seed_of(n, weights, base):
    """Seed of a configuration; depends only on (n, weights), never on the method."""
    return base + 1000 * n + WEIGHT_SETTINGS.index(weights)


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
    started on this configuration.
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

    for (n, weights, baseline) in spec["configs"]:
        seed = seed_of(n, weights, args.seed)

        # The baseline of the configuration is the denominator of every gap of
        # that configuration, and it is obtained in one of two ways.  On the
        # small scale it is the value of the MILP model, so the model is run
        # first and its values are the denominator.  On the other two scales it
        # is the reference value LB = max{LB_1, LB_2, LB_3} of paper Section 4.3,
        # a property of the instance: every method computes it from the instance
        # it has just been handed -- see instance_baseline() of the algorithm
        # modules -- so nothing extra has to be run at all.
        base_objs, base_res = None, None
        wlabel = WEIGHTS_LABEL.get(weights, weights)
        rlabel = RANGE_LABEL                  # this driver varies no range
        if done:                          # separate the configurations
            print(RULE, file=stream, flush=True)


        # Every method of the configuration runs under its own header and is
        # followed by the summary of its own instances, so that the live output
        # and the record file with it read group by group, as the small scale
        # does.  Where the baseline of the configuration is the model, the model
        # is the first group: it is run here, under its own header, and its
        # instances are not run a second time when its turn comes.
        if baseline == "milp":
            label, runner = METHODS["milp"]
            head = ("  n=%-4d %-6s | %s (seed %d) | baseline %s"
                    % (n, wlabel, label, seed, label))
            print(head, file=stream, flush=True)
            _record(record_path, head)
            base_res = runner(n, weights, instances, seed, args,
                              _instance_reporter(label, n, wlabel, rlabel, stream))
            base_objs = base_res["objs"]

        results, group = {}, []
        for name in methods:
            label, runner = METHODS[name]
            if base_res is not None and name == "milp":
                res = base_res               # already run, as the baseline above
            else:
                head = ("  n=%-4d %-6s | %s (seed %d) | baseline %s"
                        % (n, wlabel, label, seed, BASELINE_LABEL[baseline]))
                print(head, file=stream, flush=True)
                _record(record_path, head)
                res = runner(n, weights, instances, seed, args,
                             _instance_reporter(label, n, wlabel, rlabel, stream))
            results[name] = res

            denom = base_objs if base_objs is not None else res.get("bases")
            gaps, missing = gaps_against(res["objs"], denom)
            # The objective values and the running times of the configuration,
            # summarised by their smallest, their mean and their largest.
            vals = [o for o in res["objs"] if o is not None]
            secs = [t for t in res["times"] if t is not None]
            stat_obj = (statistics.fmean(vals), min(vals), max(vals)) if vals else None
            stat_time = ((statistics.fmean(secs), min(secs), max(secs))
                         if secs else None)
            rec = {
                "algo": name,
                "label": label,
                "n": n,
                "weights": weights,
                "baseline": baseline,
                "instances": instances,
                "paired": len(gaps),
                "missing": missing,
                "mean_gap": statistics.fmean(gaps) if gaps else float("nan"),
                "min_gap": min(gaps) if gaps else float("nan"),
                "max_gap": max(gaps) if gaps else float("nan"),
                "mean_time": res["mean_time"],
                "obj": stat_obj,
                "time": stat_time,
                "hit": 0,       # counted below, once every method has reported
                # the raw per-instance values, so that the file still holds the
                # run after the driver has moved on: a change of baseline or of
                # gap formula can then be applied without repeating the run
                "objs": res["objs"],
                "times": res["times"],
                "bases": res.get("bases"),
            }
            records.append(rec)
            group.append(rec)
            done += 1
            o_mean, o_min, o_max = stat_obj or (float("nan"),) * 3
            t_mean, t_min, t_max = stat_time or (float("nan"),) * 3
            # The summary of the group, written to the record file as it is
            # printed, so that the file carries the totals of every group of
            # instances as well as the instances themselves.
            summary = ("  [%2d/%2d] %-4s n=%-4d %-6s "
                       "| gap %+8.3f%% (%+8.3f%%, %+9.3f%%) "
                       "| obj %9.1f (%8.1f, %9.1f) "
                       "| time %9.4f s (%8.4f, %10.4f) | unpaired %d"
                       % (done, total, label, n, wlabel, rec["mean_gap"],
                          rec["min_gap"], rec["max_gap"], o_mean, o_min, o_max,
                          t_mean, t_min, t_max, missing))
            print(summary, file=stream, flush=True)
            _record(record_path, summary)

        # The best value found on each instance by any of the methods run here:
        # on the small scale the MILP model is among them and the value is the
        # optimum, on the other two scales it is the best known one.  The count
        # needs every method of the configuration, so it is taken here, after
        # the groups, and reaches the results file rather than the summary line
        # of the group it belongs to.
        best_of_instance = []
        for i in range(instances):
            vals = [results[m]["objs"][i] for m in methods
                    if results[m]["objs"][i] is not None]
            best_of_instance.append(min(vals) if vals else None)
        for rec in group:
            objs = results[rec["algo"]]["objs"]
            rec["hit"] = sum(1 for i, o in enumerate(objs)
                             if o is not None and best_of_instance[i] is not None
                             and abs(o - best_of_instance[i]) < 1e-9)
            rec["best_of_instance"] = best_of_instance

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

def _header(scales, methods, args):
    return [
        "F2 || WCmax -- computational study (paper Section 6)",
        "generated     : %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scales        : %s" % ", ".join(scales),
        "methods       : %s" % "; ".join(
            "%s: %s" % (s, ", ".join(METHODS[m][0] for m in SCALES[s]["methods"]))
            for s in scales),
        "instances/cfg : %d" % (args.instances or 20),
        "metaheuristics: ACO %s; GA %s"
        % (", ".join("%s=%s" % kv for kv in sorted(ACO_SETTINGS.items())),
           ", ".join("%s=%s" % kv for kv in sorted(GA_SETTINGS.items()))),
        "gap against   : the baseline of each configuration "
                        "(MILP on the small scale, the reference value "
                        "LB = max{LB_1,LB_2,LB_3} (paper Section 4.3) elsewhere)",
        "seed          : %d (+1000*n, + weight setting)" % args.seed,
        "time limit    : %s s" % (args.time_limit if args.time_limit else "none"),
        "records       : %s" % getattr(args, "record_path", "-"),
    ]


def _scale_block(lines, scale, records, summary, args):
    spec = SCALES[scale]
    lines.append("")
    lines.append("# " + "=" * 84)
    lines.append("# scale %s  --  configurations: %s"
                 % (scale, ", ".join("n=%d/%s" % (n, WEIGHTS_LABEL.get(wt, wt))
                                     for n, wt, _ in spec["configs"])))
    lines.append("# " + "=" * 84)
    lines.append("# per-configuration results")
    lines.append("# %-5s %5s %-7s %9s %9s %27s %31s %23s %6s"
                 % ("algo", "n", "weights", "paired", "unpaired",
                    "gap % mean (min, max)", "time(s) mean (min, max)",
                    "objective mean (min, max)", "best"))
    for r in records:
        o = r.get("obj") or (float("nan"),) * 3
        t = r.get("time") or (float("nan"),) * 3
        lines.append("  %-5s %5d %-7s %9d %9d %+9.3f (%+8.3f, %+9.3f)"
                     " %12.4f (%8.4f, %10.4f) %11.1f (%8.1f, %9.1f) %6d"
                     % (r["label"], r["n"], WEIGHTS_LABEL.get(r["weights"],
                                                              r["weights"]),
                        r["paired"], r["missing"], r["mean_gap"], r["min_gap"],
                        r["max_gap"], t[0], t[1], t[2],
                        o[0], o[1], o[2], r["hit"]))
    lines.append("#")
    lines.append("# the column 'best' counts the instances on which the method attained the")
    lines.append("# best value found on that instance by any method run there")
    lines.append("#")
    lines.append("# per-instance baseline, objective values and CPU times, in instance order")
    seen = set()
    for r in records:
        wl = WEIGHTS_LABEL.get(r["weights"], r["weights"])
        key = (r["n"], r["weights"])
        if key not in seen:            # the same for every method of the configuration
            seen.add(key)
            lines.append("  %-5s %5d %-7s best  : %s"
                         % ("-", r["n"], wl,
                            " ".join("none" if b is None else "%.1f" % b
                                     for b in r["best_of_instance"])))
        if r.get("bases") is not None:      # the exact methods carry no baseline
            lines.append("  %-5s %5d %-7s base  : %s"
                         % (r["label"], r["n"], wl,
                            " ".join("%.1f" % b for b in r["bases"])))
        lines.append("  %-5s %5d %-7s objs  : %s"
                     % (r["label"], r["n"], wl,
                        " ".join("none" if o is None else "%.1f" % o
                                 for o in r["objs"])))
        lines.append("  %-5s %5d %-7s times : %s"
                     % (r["label"], r["n"], wl,
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
        print("scales:")
        for name, spec in SCALES.items():
            print("  %-7s %d instances/configuration, methods: %s"
                  % (name, spec["instances"],
                     ", ".join(METHODS[m][0] for m in spec["methods"])))
            for n, wt, base in spec["configs"]:
                print("            n=%-4d %-6s  baseline %s"
                      % (n, WEIGHTS_LABEL[wt], BASELINE_LABEL[base]))
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
    print("  metaheur.: ACO %s; GA %s"
          % (", ".join("%s=%s" % kv for kv in sorted(ACO_SETTINGS.items())),
             ", ".join("%s=%s" % kv for kv in sorted(GA_SETTINGS.items()))))
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
        print("summary for %s (averaged over its configurations):" % scale)
        print("  %-8s %14s %11s %13s"
              % ("method", "configurations", "mean_gap", "mean_time(s)"))
        for s in summary:
            print("  %-8s %14d %11.3f %13.4f"
                  % (s["label"], s["configurations"], s["mean_gap"], s["mean_time"]))
    print()
    print("results written to %s" % out_path)
    print("archived as        %s" % archive_path)


if __name__ == "__main__":
    main()
