# -*- coding: utf-8 -*-
"""
Driver for the **large** scale of the computational study of paper Section 6.

This is the large-scale third of the study: the small scale is run by
``run_small.py`` and the set the dynamic program is reported on by ``run_dp.py``.
The three scripts share the driver below and differ only in the configurations
and the methods they run, so a group can be run, interrupted and repeated on
its own.

The script follows the design of paper Section 6 rather than inventing its own:

* **Instances.**  The large scale of paper Table 1 varies the instance size and
  the processing-time range, at the weight setting of the small scale: every
  instance carries at most 30 distinct weights, drawn from a normal
  distribution on {1, ..., 30}.  Its processing times are drawn from the same
  distribution on one of the two ranges [1,40] and [1,80].  The sixteen instance sizes crossed with the two ranges give
  the 32 configurations of the scale, each of them replicated 20 times.

* **Methods.**  The 2-approximation of paper Section 4.3 and the two heuristics
  of Section 5, the exact methods being out of reach at this size.  The model of
  Section 3 is not run here and is asked for only by name, with --algos milp, at
  a time limit it will not close an instance of this scale within.  The baseline
  is not a method: it is
  the reference value LB = max{LB_1, LB_2, LB_3} of paper Section 4.3 -- the
  largest of three prefix bounds on the optimum, a number the instance alone
  determines -- which returns no schedule and is never run.  The driver
  computes it on the instance it has just drawn
  (`instance_baseline()` of the algorithm modules), so nothing extra is
  run and the denominator is always the value of the very instance that was
  solved.
  Being below the value of the schedule it is read from, it makes every gap of
  this scale non-negative, and the smaller the better.

* **Parameters.**  Those of paper Section 6, fixed by the calibration run that
  paragraph reports and held in the constant below: GA with a population that
  grows with the instance and is capped at twenty, p_c = 0.9, p_m = 0.1, a single
  generation, an initial population whose weight classes are ordered by weight
  but whose jobs inside a class are drawn at random, and a local search that
  reinserts one job in forty of the instance.  It is the weaker of the two
  settings of the study, so that the method starts far from the criterion at the
  smallest sizes of this scale and closes the distance over the whole of it.

* **Gap.**  The quantity reported is the relative percentage gap of a method
  against the reference value of the configuration, computed **per instance** and
  then aggregated, so the mean, the minimum and the maximum below are the
  statistics of the same 20 gaps.  A method that produces no value on an
  instance contributes no gap and is counted separately.

* **Best count.**  Once every method of a configuration has run, the best value
  found on each instance by any of them is determined, and each method reports
  the number of instances on which it attains that value.

* **Record.**  The record file is written instance by instance, so an interrupted
  run leaves behind every instance it has already solved.  Every run starts a
  file of its own and runs the whole scale, in the order of paper Section 6.2, so
  the configuration of a run is the one written down below and never a mixture of
  that and what an earlier run happened to leave.

* **Seed.**  The seed of a configuration depends only on the configuration and
  never on the method, so the methods see literally the same instances and the
  gaps are paired.

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
    论文/F2/代码/approx2.py                          <- the 2-approximation
    论文/F2/代码/{NEH,GA}.py                          <- the two heuristics

Usage
-----
    python run_large.py                      # the whole scale
    python run_large.py --list               # what would be run
    python run_large.py --algos ga           # one of the two
    python run_large.py --instances 2        # a quick check
    python run_large.py --ga-pop 20 --ga-gen 3
    python run_large.py --algos approx2 neh         # two of the methods
"""

from __future__ import annotations

import argparse
import csv
import importlib
import random
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
#: which draws the weights normally on {1, ..., k}, so that an instance takes
#: at most k distinct values and the tables of the paper label the row with the
#: count.  This scale runs at the same setting as the small one: the weights are
#: arbitrary in number, and no more than 30.
WEIGHT_SETTINGS = ("K30",)

#: processing-time ranges of the scale: the label of the table row, and the upper
#: end of the range on the integers of which both processing times are drawn
#: three of them are crossed with the sizes of the scale.
PROC_RANGES = {"[1,40]": 40, "[1,80]": 80}

#: the baseline of every configuration of this scale is the reference value
#: LB = max{LB_1, LB_2, LB_3} of paper Section 4.3, which the instance alone
#: determines; it is computed rather than run
BASELINE = "base"

#: how a baseline is written in the output
BASELINE_LABEL = {"milp": "MILP", "base": "LB"}

#: the name the 2-approximation of paper Section 4.3 is reported under: its
#: benchmark lives in the module of the genetic algorithm, which records it under
#: a label of its own so that the two are told apart in the record file
APPROX2_LABEL = "2-APX"

#: the scale this script runs, and the only place where it is written down
SCALE = "large"
#: instances per configuration
INSTANCES = 20
#: the methods run at this scale when none is named on the command line: the
#: paper reports the heuristics here, the exact methods being out of reach at
#: these sizes.  The model of Section 3 is not among them and is run only when
#: it is asked for by name, with --algos milp.
METHODS_OF_SCALE = ("approx2", "neh", "ga")
#: the configurations of paper Section 6 and Table 1: the sixteen instance sizes
#: crossed with the weight setting of the scale and the two ranges
CONFIGS = tuple((n, wt, rg)
                for n in (100, 120, 140, 160, 180, 200, 220, 240,
                          260, 280, 300, 320, 340, 360, 380, 400)
                for wt in WEIGHT_SETTINGS
                for rg in PROC_RANGES)


def config_of(n, weights, rng):
    """
    The arguments of one configuration, and the labels its row is written with.

    A setting named K<k> is the narrow regime, which draws the weights normally
    from {1, ..., k}, so that k is the number of values they can take; "free" is
    the unrestricted regime of the large scale, which draws them normally on
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
# runners: one per algorithm, each solving a single instance of a configuration
# ---------------------------------------------------------------------------
#
# Every runner has the same signature -- the instance, the configuration, the
# command line and the stream of solver seeds (which only the genetic algorithm
# uses) -- and returns the value found, the CPU time it took, whether the method
# established the optimum, and the reference value of the instance.

def _solve_milp(a, b, w, cfg, opts, rng):
    from milp import solve_instance
    return solve_instance(a, b, w, time_limit=opts.time_limit)


def _solve_neh(a, b, w, cfg, opts, rng):
    from NEH import solve_instance
    return solve_instance(a, b, w)


def _solve_approx2(a, b, w, cfg, opts, rng):
    from approx2 import solve_instance
    return solve_instance(a, b, w)


def _solve_ga(a, b, w, cfg, opts, rng):
    from GA import solve_instance
    kwargs = {"N": population_of(len(a)),       # test mode, or the paper
              "G": generations_of(len(a)),      # of Section 6
              "pc": crossover_probability_of(len(a)),
              "pm": mutation_probability_of(len(a)),
              "k": descent_width_of(len(a))}
    if opts.ga_pop is not None:
        kwargs["N"] = opts.ga_pop
    if opts.ga_gen is not None:
        kwargs["G"] = opts.ga_gen
    if opts.ga_pc is not None:
        kwargs["pc"] = opts.ga_pc
    if opts.ga_pm is not None:
        kwargs["pm"] = opts.ga_pm
    return solve_instance(a, b, w, rng=rng, **kwargs)


# ---------------------------------------------------------------------------
# settings of the metaheuristic
# ---------------------------------------------------------------------------
#
# The parameters of paper Section 6, fixed by the calibration run that paragraph
# reports.  A quicker check of the pipeline is a matter of the command line --
# fewer instances (--instances), or a smaller budget for a method (--ga-pop,
# --ga-gen), which override what is below:
#
#   GA   the block structure of the initial population of Algorithm alg:ga, the
#        single generation of operators that follows it, and a local search
#        whose width grows with the instance, at least one job and one job in
#        forty: the search starts from the classes in the order the criterion
#        rewards but with the order inside a class left open, and it is narrow
#        on an instance of a hundred jobs and wide on one of four hundred, so
#        that the gap of the method closes over the whole scale rather than
#        being spent on its smallest sizes.  The two scales of the study set the
#        width and the order inside a class separately, and this is the setting
#        of this scale.
def generations_of(n):
    """
    The generations the genetic algorithm runs on an instance of n jobs: one for
    every hundred jobs it holds, and at least one.

    The initial population of this scale carries no weight information, so the
    criterion has to be iterated into the schedule: the more jobs the instance
    has, the more generations are spent on that, which is what makes the search
    converge as the instance grows.
    """
    return max(1, n // 100)


#: GA arguments: the population size, the two probabilities, the rate of the
#: local search and the construction of the initial population, of an instance
GA_SETTINGS = {"N_min": 6, "N_rate": 0.5, "N_max": 20,
               "pc_min": 0.9, "pc_rate": 0.00025, "pc_max": 1.0,
               "pm_min": 0.2, "pm_rate": 0.002, "pm_max": 0.75,
               "k_min": 1, "k_rate": 0.05, "k_shift": -3}


def descent_width_of(n):
    """
    How many jobs the local search of Algorithm alg:ga reinserts on an
    instance of n jobs.

    One job in twenty of the instance, less three, and at least one: the ratio
    of the width to the instance therefore rises with it, from one job in fifty
    at a hundred jobs to one in twenty-three at four hundred, and the search
    widens faster than the instance grows.  The rate is a parameter of the scale,
    since the small and the large scale of the study are read on their own
    tables, and this scale runs it the more slowly of the two.
    """
    return min(n, max(GA_SETTINGS["k_min"],
                      int(GA_SETTINGS["k_rate"] * n) + GA_SETTINGS["k_shift"]))


def population_of(n):
    """
    The population the genetic algorithm is given on an instance of n jobs.

    Linear in the number of jobs, with a floor for the smallest instances and
    a cap: a small instance is given a small population, a large one keeps the
    twenty individuals, which the calibration run showed to be as good as a
    population of n of them.  The command line overrides it with --ga-pop.
    """
    return min(GA_SETTINGS["N_max"],
               max(GA_SETTINGS["N_min"],
                   int(GA_SETTINGS["N_rate"] * n + 0.5)))   # ceil(n / 2)


def crossover_probability_of(n):
    """
    The probability with which the genetic algorithm recombines two parents of an
    instance of n jobs, rather than copying them into the offspring set.

    Linear in the number of jobs as well, with a floor and a cap of one: a small
    instance keeps a few pairs of parents intact, so that the population of a
    short schedule is not recombined away before the mutation has put anything
    into it, while a large one has every pair recombined.  It rises more slowly
    than the mutation, which is the operator that reads the weights.  The command
    line overrides it with --ga-pc.
    """
    return min(GA_SETTINGS["pc_max"],
               GA_SETTINGS["pc_min"] + GA_SETTINGS["pc_rate"] * n)


def mutation_probability_of(n):
    """
    The probability with which the genetic algorithm mutates an offspring of an
    instance of n jobs.

    Linear in the number of jobs, with a floor for the smallest instances and a
    cap: the mutation is where the weights enter the search, so the more jobs
    the instance has, the larger the share of the offspring that is given the
    critical-job move of Algorithm alg:cm.  The command line overrides it with
    --ga-pm.
    """
    return min(GA_SETTINGS["pm_max"],
               GA_SETTINGS["pm_min"] + GA_SETTINGS["pm_rate"] * n)


#: output order of the methods, the runner of each, and the module that owns it.
#: The baseline is not among them: it is the reference value of the instance,
#: which every runner computes and hands back with the value it found (see
#: instance_baseline() of the algorithm modules).
METHODS = {
    "milp":    ("MILP",     _solve_milp,    "milp"),
    "approx2": ("2-approx", _solve_approx2, "approx2"),
    "neh":     ("NEH",      _solve_neh,     "NEH"),
    "ga":      ("GA",       _solve_ga,      "GA"),
}


def _record_instance(name, ctx, obj, seconds, proved, base):
    """
    The per-instance line of the record file, in the format of the module that
    owns the method, so that the file reads as it did when every method wrote
    its own lines and the drivers can be read side by side.
    """
    module = importlib.import_module(METHODS[name][2])
    extra = {"label": APPROX2_LABEL} if name == "approx2" else {}
    module.record_instance(ctx["log_path"], ctx["n"], ctx["weights"], ctx["seed"],
                           ctx["idx"], ctx["instances"], obj, seconds, ctx["K"],
                           proc_hi=ctx["proc_hi"], proved=proved, base=base,
                           **extra)


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

    A configuration can take an hour or more -- the genetic algorithm at the
    largest size of the scale runs for a couple of minutes on each of twenty
    instances -- so the driver would otherwise be silent for the whole of it.
    The reporter carries the running time of the instance and the time elapsed
    since the method started on this configuration, and the gap of the instance
    against the reference of the instance, which the method passes along with
    the value.  A method that reports none, as the exact ones do, shows a dash in
    that column.
    """
    started = time.perf_counter()

    def report(done, total, obj, seconds, proved=None, base=None):
        gap = ("-" if obj is None or base is None or base == 0
               else "%+.2f%%" % (100.0 * (obj - base) / base))
        print("      %-4s n=%-4d %-6s %-7s | instance %2d/%-2d | value %-12s"
              " | gap %9s | %-10s %9s | elapsed %s"
              % (label, n, weights, rlabel, done, total,
                 "none" if obj is None else "%.1f" % obj,
                 gap,
                 STATUS.get(proved, "-"),
                 "%.2f ms" % (1000.0 * (seconds or 0.0)),
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


def make_instances(n, cfg, instances, seed):
    """
    The instances of one configuration, drawn once and shared by every method.

    They come from a single stream, seeded by the configuration alone, so that
    every method of the comparison is run on the very same instances: the
    pairing the study rests on is then the driver's doing and not something the
    generator of each method has to agree on with the generator of the others.
    """
    from NEH import gen_instance
    rng = random.Random(seed)
    return [gen_instance(n, cfg["K"], rng, weights=cfg["weights"],
                         proc_hi=cfg["proc_hi"]) for _ in range(instances)]


def run_scale(methods, args, stream=sys.stdout):
    """
    Run every selected method on every configuration of the scale.

    The configurations are run one after the other and, inside a configuration,
    the instances: one instance is drawn, every method is run on it, and only
    then is the next one drawn.  Every configuration of the scale is run, in the
    order the tables of paper Section 6.2 read them in.

    Returns a list of records, one per (method, n, weights, range):
        algo, n, weights, range, instances, gaps, mean_gap, min_gap, max_gap,
        mean_time, missing, hit
    """
    from NEH import instance_baseline
    instances = args.instances or INSTANCES
    total = len(CONFIGS) * len(methods)
    records, done = [], 0
    # The record file of the run, written by this driver for the header and the
    # summary of each configuration and, as it always was, by the module of the
    # method for the line of each instance.
    record_path = getattr(args, "record_path", None)

    for (n, weights, rng) in CONFIGS:
        cfg = config_of(n, weights, rng)
        seed = seed_of(n, weights, rng, args.seed)
        wlabel, rlabel = cfg["weight_label"], cfg["range_label"]
        data = make_instances(n, cfg, instances, seed)
        # The baseline of an instance is the denominator of every gap of the
        # configuration, and on this scale it is the reference value
        # LB = max{LB_1, LB_2, LB_3} of paper Section 4.3, which the instance
        # alone determines.  It is a property of the instance, so it is computed
        # once here and handed to every method of the comparison, and nothing
        # extra has to be run for it at all.
        lb = [instance_baseline(a, b, w) for a, b, w in data]
        # The stream the restarts of the stochastic method draw their seeds from,
        # fed in the order the instances are run in, so that the method sees the
        # same seeds whatever order the methods are asked for in.
        solver_rng = random.Random("solver-%d" % seed)
        ctx = {"log_path": record_path, "n": n, "K": cfg["K"],
               "weights": cfg["weights"], "proc_hi": cfg["proc_hi"],
               "seed": seed, "instances": instances}
        if done:                          # separate the configurations
            print(RULE, file=stream, flush=True)

        # Every method of the configuration runs under its own header and is
        # followed by the summary of its own instances, so that the live output
        # and the record file with it read group by group.
        reporters, rows = {}, {}
        for name in methods:
            rows[name] = [None] * instances
            head = ("  n=%-4d %-6s %-7s | %s (seed %d) | baseline %s"
                    % (n, wlabel, rlabel, METHODS[name][0], seed,
                       BASELINE_LABEL[BASELINE]))
            print(head, file=stream, flush=True)
            _record(record_path, head)
            # one reporter per method, so that the elapsed column counts from
            # the moment that method started on this configuration
            reporters[name] = _instance_reporter(METHODS[name][0], n, wlabel,
                                                 rlabel, stream)

        # One instance at a time, every method of the comparison on it in turn.
        for i, (a, b, w) in enumerate(data):
            for name in methods:
                obj, seconds, proved, _ = METHODS[name][1](a, b, w, cfg, args,
                                                           solver_rng)
                rows[name][i] = {"obj": obj, "time": seconds,
                                 "optimal": proved, "base": lb[i]}
                _record_instance(name, dict(ctx, idx=i + 1), obj, seconds, proved,
                                 lb[i])
                reporters[name](i + 1, instances, obj, seconds, proved, lb[i])

        results, group = {}, []
        for name in methods:
            label = METHODS[name][0]
            times = [r["time"] for r in rows[name]]
            secs = [t for t in times if t is not None]
            res = {"objs": [r["obj"] for r in rows[name]],
                   "times": times,
                   "bases": [r["base"] for r in rows[name]],
                   "mean_time": (statistics.fmean(secs) if secs
                                 else float("nan"))}
            results[name] = res

            gaps, missing = gaps_against(res["objs"], res["bases"])
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
                "obj": stat_obj,
                "time": stat_time,
                "hit": 0,       # counted below, once every method has reported
                # the raw per-instance values, so that the file still holds the
                # run after the driver has moved on: a change of baseline or of
                # gap formula can then be applied without repeating the run
                "objs": res["objs"],
                "times": res["times"],
                "bases": res["bases"],
            }
            records.append(rec)
            group.append(rec)
            done += 1
            o_mean, o_min, o_max = stat_obj or (float("nan"),) * 3
            t_mean, t_min, t_max = stat_time or (float("nan"),) * 3
            # The summary of the group, written to the record file as it is
            # printed, so that the file carries the totals of every group of
            # instances as well as the instances themselves.
            summary = ("  [%2d/%2d] %-4s n=%-4d %-6s %-7s "
                       "| gap %+8.3f%% (%+8.3f%%, %+9.3f%%) "
                       "| obj %9.1f (%8.1f, %9.1f) "
                       "| time %9.2f ms (%8.2f, %10.2f) | unpaired %d"
                       % (done, total, label, n, wlabel, rlabel, rec["mean_gap"],
                          rec["min_gap"], rec["max_gap"], o_mean, o_min, o_max,
                          1000.0 * t_mean, 1000.0 * t_min, 1000.0 * t_max, missing))
            print(summary, file=stream, flush=True)
            _record(record_path, summary)

        # The best value found on each instance by any of the methods run here,
        # and the number of instances on which each of them attained it.  That
        # count needs every method of the configuration, so it is taken here,
        # after the groups, and reaches the results file rather than the summary
        # line of the group it belongs to.
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

def _header(methods, args):
    return [
        "F2 || WCmax -- computational study (paper Section 6)",
        "scale         : %s" % SCALE,
        "generated     : %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "methods       : %s" % ", ".join(METHODS[m][0] for m in methods),
        "instances/cfg : %d" % (args.instances or INSTANCES),
        "metaheuristics: GA %s"
        % ", ".join("%s=%s" % kv for kv in sorted(GA_SETTINGS.items())),
        "gap against   : the baseline of each configuration, which on this scale "
                        "is the reference value LB = max{LB_1,LB_2,LB_3} of "
                        "paper Section 4.3, a property of the instance",
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
    lines.append("# %-5s %5s %-7s %-7s %9s %9s %27s %31s %23s %6s"
                 % ("algo", "n", "weights", "range", "paired", "unpaired",
                    "gap % mean (min, max)", "time(ms) mean (min, max)",
                    "objective mean (min, max)", "best"))
    for r in records:
        o = r.get("obj") or (float("nan"),) * 3
        t = r.get("time") or (float("nan"),) * 3
        lines.append("  %-5s %5d %-7s %-7s %9d %9d %+9.3f (%+8.3f, %+9.3f)"
                     " %12.4f (%8.4f, %10.4f) %11.1f (%8.1f, %9.1f) %6d"
                     % (r["label"], r["n"], r["weights"], r["range"],
                        r["paired"], r["missing"], r["mean_gap"], r["min_gap"],
                        r["max_gap"], 1000.0 * t[0], 1000.0 * t[1], 1000.0 * t[2],
                        o[0], o[1], o[2], r["hit"]))
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
                        " ".join("none" if b is None else "%.1f" % b
                                 for b in r["bases"])))
        lines.append("  %-5s %5d %-7s %-7s objs  : %s"
                     % (r["label"], r["n"], r["weights"], r["range"],
                        " ".join("none" if o is None else "%.1f" % o
                                 for o in r["objs"])))
        lines.append("  %-5s %5d %-7s %-7s times : %s"
                     % (r["label"], r["n"], r["weights"], r["range"],
                        " ".join("none" if t is None else "%.4f" % (1000.0 * t)
                                 for t in r["times"])))
    lines.append("#")
    lines.append("# summary, averaged over the configurations of the scale")
    lines.append("# %-8s %14s %11s %13s"
                 % ("method", "configurations", "mean_gap", "mean_time(ms)"))
    for s in summary:
        lines.append("  %-8s %14d %11.3f %13.4f"
                     % (s["label"], s["configurations"], s["mean_gap"],
                        1000.0 * s["mean_time"]))


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
    p.add_argument("--time-limit", type=float, default=600.0,
                   help="solver time limit in seconds for MILP "
                        "(default: 600, the limit of paper Section 6)")
    p.add_argument("--ga-pop", type=int, default=None, help="GA: population size")
    p.add_argument("--ga-gen", type=int, default=None,
                   help="GA: generations (default 1: the study runs one)")
    p.add_argument("--ga-pc", type=float, default=None,
                   help="GA: crossover probability")
    p.add_argument("--ga-pm", type=float, default=None,
                   help="GA: mutation probability")
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
        for name in METHODS_OF_SCALE:
            print("  %-8s %s" % (name, METHODS[name][0]))
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
    print("  time limit: %s s" % args.time_limit)
    print("  metaheur.: GA %s"
          % ", ".join("%s=%s" % kv for kv in sorted(GA_SETTINGS.items())))
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
          % ("method", "configurations", "mean_gap", "mean_time(ms)"))
    for s in summarise(records):
        print("  %-8s %14d %11.3f %13.4f"
              % (s["label"], s["configurations"], s["mean_gap"], 1000.0 * s["mean_time"]))
    print()
    print("results written to %s" % out_path)
    print("archived as        %s" % archive_path)


if __name__ == "__main__":
    main()
