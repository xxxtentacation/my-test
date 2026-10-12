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
  instance carries at most 30 distinct weights, drawn uniformly from
  {1, ..., 30}.  Its processing times are drawn from one of the two ranges
  [1,40] and [1,80].  The sixteen instance sizes crossed with the two ranges give
  the 32 configurations of the scale, each of them replicated 20 times.

* **Methods.**  The 2-approximation of paper Section 4.3 and the two heuristics of
  Section 5, the exact methods being out of reach at this size.  The baseline is not a method: it is
  the reference value LB = max{LB_1, LB_2, LB_3} of paper Section 4.3 -- the
  largest of three prefix bounds on the optimum, a number the instance alone
  determines -- which returns no schedule and is never run.  Each algorithm
  computes it on the instance it has just been handed
  (`instance_baseline()` of the algorithm modules), so nothing extra is
  run and the denominator is always the value of the very instance that was
  solved.
  Being below the value of the schedule it is read from, it makes every gap of
  this scale non-negative, and the smaller the better.

* **Parameters.**  Those of paper Section 6, fixed by the calibration run that
  paragraph reports and held in the constant below: GA with N=n individuals, G=10
  generations, p_c=0.9, p_m=0.1 and an elitism size of N/10.

* **Gap.**  The quantity reported is the relative percentage gap of a method
  against the reference value of the configuration, computed **per instance** and
  then aggregated, so the mean, the minimum and the maximum below are the
  statistics of the same 20 gaps.  A method that produces no value on an
  instance contributes no gap and is counted separately.

* **Best count.**  Once every method of a configuration has run, the best value
  found on each instance by any of them is determined, and each method reports
  the number of instances on which it attains that value.

* **Continuation.**  The record file is written instance by instance, so an
  interrupted run is continued rather than repeated: started again, the driver
  takes up the newest unfinished record file of the scale, reads back the groups
  of instances it already holds and runs only what is missing.  --fresh starts a
  new file whatever the earlier runs left behind, and --resume takes one up by
  name.

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
    论文/F2/代码/{NEH,GA}.py                          <- the two heuristics

Usage
-----
    python run_large.py                      # the whole scale
    python run_large.py --list               # what would be run
    python run_large.py --algos ga           # one of the two
    python run_large.py --instances 2        # a quick check
    python run_large.py --ga-pop 20 --ga-gen 3
    python run_large.py --fresh                     # start over, do not continue
"""

from __future__ import annotations

import argparse
import csv
import random
import re
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
#: which draws the weights uniformly from {1, ..., k}, so that an instance takes
#: at most k distinct values and the tables of the paper label the row with the
#: count.  This scale runs at the same setting as the small one: the weights are
#: arbitrary in number, and no more than 30.
WEIGHT_SETTINGS = ("K30",)

#: processing-time ranges of the scale: the label of the table row, and the upper
#: end of the discrete uniform range both processing times are drawn from.  The
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
#: the methods the paper reports at this scale
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

def _run_neh(n, cfg, instances, seed, opts, progress=None, first=1):
    from NEH import benchmark
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed, first=first,
                     weights=cfg["weights"], proc_hi=cfg["proc_hi"],
                     progress=progress, log_path=getattr(opts, "record_path", None))


def _run_approx2(n, cfg, instances, seed, opts, progress=None, first=1):
    from GA import benchmark
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed, first=first,
                     weights=cfg["weights"], proc_hi=cfg["proc_hi"],
                     method="2-approx", label=APPROX2_LABEL,
                     progress=progress, log_path=getattr(opts, "record_path", None))


def _run_ga(n, cfg, instances, seed, opts, progress=None, first=1):
    from GA import benchmark
    kwargs = {"N": max(8, GA_SETTINGS["N_factor"] * n),   # test mode, or the paper
              "G": GA_SETTINGS["G"]}                      # values of Section 6
    if opts.ga_pop is not None:
        kwargs["N"] = opts.ga_pop
    if opts.ga_gen is not None:
        kwargs["G"] = opts.ga_gen
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed, first=first,
                     weights=cfg["weights"], proc_hi=cfg["proc_hi"],
                     progress=progress, log_path=getattr(opts, "record_path", None),
                     **kwargs)


# ---------------------------------------------------------------------------
# settings of the metaheuristic
# ---------------------------------------------------------------------------
#
# The parameters of paper Section 6, fixed by the calibration run that paragraph
# reports.  A quicker check of the pipeline is a matter of the command line --
# fewer instances (--instances), or a smaller budget for a method (--ga-pop,
# --ga-gen), which override what is below:
#
#   GA   N = n, G = 10 reproduces the mean objective of N = 2n, G = 30 on both
#        weight regimes, at a quarter of the running time: the population reaches
#        its fixed point within a few generations, and even N = n/2, G = 10
#        still matches.
#: GA arguments: population size as a multiple of n, and generations
GA_SETTINGS = {"N_factor": 1, "G": 10}


#: output order of the methods, and the runner of each.  The baseline is not
#: among them: it is the reference value of the schedule a method reports, which
#: each algorithm computes on the instance it is handed (see instance_baseline()
#: in the algorithm modules).
METHODS = {
    "approx2": ("2-approx", _run_approx2),
    "neh":     ("NEH",      _run_neh),
    "ga":      ("GA",       _run_ga),
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


#: what the per-instance line says about optimality: True when the method closed
#: the instance, False when it ran into the time limit, None when the method is a
#: heuristic and proves nothing
STATUS = {True: "optimal", False: "time limit", None: "-"}

#: separates one configuration from the next in the live output, so that each
#: block of instances stands apart in a long run
RULE = "-" * 100

#: the per-instance line of the record file, as the algorithm modules write it:
#: the moment the instance finished, the method, the configuration, the seed, the
#: number of the instance, the value, the gap against the baseline of that
#: instance, the status and the running time
INSTANCE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+(\S+)\s+n=(\d+)\s+(.+?)\s+"
    r"\[1,(\d+)\]\s+\|\s+seed=(\d+)\s+\|\s+instance\s+(\d+)/(\d+)\s+\|\s+"
    r"value\s+(\S+)\s+\|\s+gap\s+(\S+)\s+\|\s+(.+?)\s+([\d.]+) s$")

#: the summary line this driver writes once a group of instances is over; it is
#: what marks that group as run to the end, and hence as not to be run again
SUMMARY_RE = re.compile(
    r"^\s*\[\s*(\d+)/(\d+)\]\s+(\S+)\s+n=(\d+)\s+(.+?)\s+\[1,(\d+)\]\s+\|\s+gap")

#: method name of each label, and weight setting of each label, so that the lines
#: of a record file can be traced back to the group that wrote them: the algorithm
#: modules label the narrow regime "K=k" and the unrestricted one "all weights"
NAME_OF_LABEL = {label: name for name, (label, _) in METHODS.items()}
#: the label the 2-approximation is recorded under, which its benchmark writes
#: instead of the name of the module it lives in
NAME_OF_LABEL[APPROX2_LABEL] = "approx2"
WEIGHTS_OF_LABEL = dict([("K=%s" % w[1:], w) for w in WEIGHT_SETTINGS
                         if w != "free"] + [("all weights", "free")])


def key_of(label, n, wlabel, proc_hi):
    """
    The configuration a line of the record file belongs to, or None.

    None means this run has no such configuration -- a line of another scale, or
    of a configuration that has been taken out of this one -- and the line is then
    ignored rather than taken up again.
    """
    name = NAME_OF_LABEL.get(label)
    weights = WEIGHTS_OF_LABEL.get(wlabel)
    rng = "[1,%d]" % proc_hi
    if name is None or weights is None or (n, weights, rng) not in CONFIGS:
        return None
    return (name, n, weights, rng)


def replay_record(path):
    """
    The groups of instances a record file already holds, read back from it.

    A record file is written instance by instance, so an interrupted run is
    continued rather than repeated at the grain the file keeps: every instance a
    group has already solved is taken from it, and the run starts at the first
    instance of that group the file does not hold -- the one the earlier run was
    in the middle of, if it was.  An instance the file holds twice, because a run
    was taken up again after solving part of the group, is read from its last
    line, the one the run that was taken up again wrote.

    Returns a dict keyed by (method, n, weights, range) whose values are, for
    every instance of the group, its value and its running time; together with
    the replication count of the group, the number of the first instance the file
    does not hold, and whether the group is finished.  A group whose lines are
    unknown here is left out.
    """
    done, order = {}, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = INSTANCE_RE.match(line)
        if m:
            key = key_of(m.group(1), int(m.group(2)), m.group(3), int(m.group(4)))
            if key is None:
                continue
            entry = done.get(key)
            if entry is None:
                entry = done[key] = {"instances": int(m.group(7)),
                                     "rows": [None] * int(m.group(7)),
                                     "summarised": False}
                order.append(key)
            # A method reports a value only when it found a schedule, and a
            # running time only for the instances it found one on.
            value = m.group(8)
            entry["rows"][int(m.group(6)) - 1] = {
                "obj": None if value == "none" else float(value),
                "time": float(m.group(11)) if value != "none" else None,
                "optimal": m.group(10) == "optimal",
            }
            continue
        m = SUMMARY_RE.match(line)
        if m:
            key = key_of(m.group(3), int(m.group(4)), m.group(5), int(m.group(6)))
            if key is not None and key in done:
                done[key]["summarised"] = True
    for entry in done.values():
        entry["first"] = next((i + 1 for i, row in enumerate(entry["rows"])
                               if row is None), entry["instances"] + 1)
        entry["complete"] = (entry["summarised"]
                             and entry["first"] > entry["instances"])
    return {k: done[k] for k in order}


def unfinished_record(record_dir, scale, instances):
    """
    The record file of the newest unfinished run of a scale, or None.

    Each run writes its record into a file of its own, named after the scale and
    the moment it began, so the newest of them is the one an interrupted run was
    writing.  A file says whether it is finished itself, in the numbering of its
    summary lines: the last of them reads "[k/n]", and k < n means the run it
    belongs to was cut short.  A file whose last summary closes its own total
    belongs to a run that went through, and is left alone -- the next run starts
    a file of its own rather than adding to it.  A file written by a shorter run,
    a pilot of a few instances per configuration, is left alone as well: it is
    not the run this one would be continuing.
    """
    if not record_dir.is_dir():
        return None
    files = sorted(record_dir.glob("实例记录_%s_*.txt" % scale))
    for path in reversed(files):          # newest first: the names are dated
        last, count = None, None
        for line in path.read_text(encoding="utf-8").splitlines():
            m = SUMMARY_RE.match(line)
            if m:
                last = (int(m.group(1)), int(m.group(2)))
            m = INSTANCE_RE.match(line)
            if m:
                count = int(m.group(7))
        if count is not None and count != instances:
            continue
        if last is None or last[0] < last[1]:
            return path
    return None


def bases_of(n, cfg, instances, seed):
    """
    The reference value of every instance of a configuration, recomputed.

    The record file reports the gap of each instance but not the baseline it was
    measured against, and the results file reports that baseline.  The instances
    themselves are what the seed generates, so the values are recomputed from
    them -- the very call the algorithm modules make on the instances they solve,
    their copies of the generator agreeing -- instead of being recovered from the
    gaps, which the file rounds to two decimals.
    """
    from NEH import gen_instance, instance_baseline
    rng = random.Random(seed)
    bases = []
    for _ in range(instances):
        a, b, w = gen_instance(n, cfg["K"], rng, False, cfg["weights"],
                               cfg["proc_hi"])
        bases.append(instance_baseline(a, b, w))
    return bases


def _from_rows(run, instances, bases=None):
    """
    The result of a group of instances a record file holds, as the benchmark of
    the method would have returned it.
    """
    objs = [row["obj"] for row in run["rows"]]
    times = [row["time"] for row in run["rows"]]
    secs = [t for t in times if t is not None]
    out = {
        "objs": objs,
        "times": times,
        "mean_time": statistics.fmean(secs) if secs else float("nan"),
        "solved": len([o for o in objs if o is not None]),
        # the heuristics of this scale prove nothing about optimality, and the
        # exact methods, which do, are reported at the small scale
        "optimal": 0,
    }
    if bases is not None:
        out["bases"] = bases
    return out


def _continued(run, res, first, bases=None):
    """
    The result of a group whose first instances an earlier run solved.

    The instances before `first` are taken from that run's record file and put in
    front of the ones just solved, so that the group is reported as if it had
    been run in one go.
    """
    prefix = run["rows"][:first - 1]
    objs = [row["obj"] for row in prefix] + list(res["objs"])
    times = [row["time"] for row in prefix] + list(res["times"])
    secs = [t for t in times if t is not None]
    out = dict(res, objs=objs, times=times,
               mean_time=statistics.fmean(secs) if secs else float("nan"),
               solved=len([o for o in objs if o is not None]))
    if bases is not None:
        out["bases"] = bases
    return out


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
    # The record file of the run, written by the algorithms instance by instance
    # and by this driver for the header and the summary of each configuration,
    # so that it reads as the live output did.
    record_path = getattr(args, "record_path", None)
    # The groups an earlier, interrupted run finished, read back from the record
    # file it left behind: they are not run again, and this run starts at the
    # first group that file has no summary for.
    earlier = getattr(args, "resumed", None) or {}

    for (n, weights, rng) in CONFIGS:
        cfg = config_of(n, weights, rng)
        seed = seed_of(n, weights, rng, args.seed)
        wlabel, rlabel = cfg["weight_label"], cfg["range_label"]

        # The baseline of the configuration is the denominator of every gap of
        # that configuration, and on this scale it is the reference value
        # LB = max{LB_1, LB_2, LB_3} of paper Section 4.3, which the instance
        # alone determines: every method computes it from the instance it has
        # just been handed -- see instance_baseline() of the algorithm modules
        # -- so nothing extra has to be run at all.
        todo = [name for name in methods
                if not (earlier.get((name, n, weights, rng)) or {})
                .get("complete")]
        if done and todo:                 # separate the configurations
            print(RULE, file=stream, flush=True)

        # Every method of the configuration runs under its own header and is
        # followed by the summary of its own instances, so that the live output
        # and the record file with it read group by group, as the small scale
        # does.
        results, group = {}, []
        for name in methods:
            label, runner = METHODS[name]
            run = earlier.get((name, n, weights, rng))
            if run is not None and run["instances"] != instances:
                raise SystemExit(
                    "record file %s holds %d instances of n=%d %s, not %d: take "
                    "the run up again with the replication count it was started "
                    "with"
                    % (record_path, run["instances"], n, wlabel, instances))
            if run is not None and run["complete"]:
                # Run to the end by the run this one continues: the instances come
                # from its record file, which is not written to again, and the
                # baselines, which the file does not carry, are recomputed from
                # the instances the seed regenerates.
                res = _from_rows(run, instances,
                                 bases_of(n, cfg, instances, seed))
            else:
                # The group is run on from the first instance the file does not
                # hold, so that an interruption inside a group costs only the
                # instance the run was solving when it stopped
                head = ("  n=%-4d %-6s %-7s | %s (seed %d) | baseline %s"
                        % (n, wlabel, rlabel, label, seed,
                           BASELINE_LABEL[BASELINE]))
                print(head, file=stream, flush=True)
                _record(record_path, head)
                res = runner(n, cfg, instances, seed, args,
                             _instance_reporter(label, n, wlabel, rlabel, stream),
                             first=1 if run is None else run["first"])
                if run is not None:
                    res = _continued(run, res, run["first"],
                                     bases_of(n, cfg, instances, seed))
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
                       "| time %9.4f s (%8.4f, %10.4f) | unpaired %d"
                       % (done, total, label, n, wlabel, rlabel, rec["mean_gap"],
                          rec["min_gap"], rec["max_gap"], o_mean, o_min, o_max,
                          t_mean, t_min, t_max, missing))
            if run is None or not run["complete"]:
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
        "resumed       : %s" % (getattr(args, "resume", None) or "-"),
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
                    "gap % mean (min, max)", "time(s) mean (min, max)",
                    "objective mean (min, max)", "best"))
    for r in records:
        o = r.get("obj") or (float("nan"),) * 3
        t = r.get("time") or (float("nan"),) * 3
        lines.append("  %-5s %5d %-7s %-7s %9d %9d %+9.3f (%+8.3f, %+9.3f)"
                     " %12.4f (%8.4f, %10.4f) %11.1f (%8.1f, %9.1f) %6d"
                     % (r["label"], r["n"], r["weights"], r["range"],
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
    p.add_argument("--ga-pop", type=int, default=None, help="GA: population size")
    p.add_argument("--ga-gen", type=int, default=None, help="GA: generations")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help="results file (default: %s; a .csv suffix writes CSV)"
                        % DEFAULT_OUT)
    p.add_argument("--resume", default=None, metavar="RECORD",
                   help="continue the run whose record file is RECORD: the "
                        "groups of instances that file already holds are read "
                        "back from it instead of being run again, and everything "
                        "this run produces is appended to the same file.  By "
                        "default the newest unfinished record file of the scale "
                        "is taken up, if there is one")
    p.add_argument("--fresh", action="store_true",
                   help="start a new run and a new record file, ignoring what "
                        "earlier runs left behind")
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
    # An interrupted run is continued rather than repeated: the file this run
    # appends to is the one named on the command line, or else the newest record
    # file of the scale, provided that run was cut short.  A run that went
    # through is left as it is, and a new file is started -- as it always is
    # under --fresh.
    if args.resume:
        resume_path = Path(args.resume)
    elif args.fresh:
        resume_path = None
    else:
        resume_path = unfinished_record(RECORD_DIR, SCALE,
                                        args.instances or INSTANCES)
    if resume_path is not None:
        if not resume_path.is_file():
            raise SystemExit("no record file at %s" % resume_path)
        args.resumed = replay_record(resume_path)
        if not args.resumed and resume_path.stat().st_size:
            # The file holds no group of the scale as it now stands: it was
            # written by a run of another design, whose configurations this run
            # cannot continue.  It is left where it is, and a record file of this
            # run is started beside it.
            print("  note      : %s holds no configuration of this scale;"
                  " starting a new record file" % resume_path.name)
            resume_path = None
    if resume_path is None:
        args.resumed = {}
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        args.record_path = RECORD_DIR / ("实例记录_%s_%s.txt" % (SCALE, stamp))
    else:
        args.record_path = resume_path

    print("F2 || WCmax -- computational study (paper Section 6)")
    print("  scale     : %s" % SCALE)
    print("  methods   : %s" % ", ".join(METHODS[m][0] for m in methods))
    print("  instances : %d per configuration" % (args.instances or INSTANCES))
    print("  metaheur.: GA %s"
          % ", ".join("%s=%s" % kv for kv in sorted(GA_SETTINGS.items())))
    print("  output    : %s" % out_path)
    print("  archive   : %s" % ARCHIVE_DIR)
    print("  records   : %s" % args.record_path)
    if args.resumed:
        left = [(key, run) for key, run in sorted(args.resumed.items())
                if not run["complete"]]
        print("  resuming  : %d groups already in that file%s"
              % (len(args.resumed) - len(left),
                 "" if not left else "; to continue: %s"
                 % ", ".join("%s n=%d/%s/%s from instance %d"
                             % (METHODS[m][0], n, w, rg, run["first"])
                             for (m, n, w, rg), run in left)))
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
