# -*- coding: utf-8 -*-
"""
Driver for the small scale of the computational study of paper Section 6.

This is the small scale of paper Table 1 on its own: the large scale is run by
``run_large.py`` and the set the dynamic program is reported on by ``run_dp.py``.
The three scripts share the driver below and differ only in the configurations
and the methods they run, so a group can be run, interrupted and repeated on
its own.

* **Instances.**  The small scale varies the instance size and the processing-time
  range.  Every instance carries at most 30 distinct weights, drawn uniformly
  from {1, ..., 30}, and its processing times are drawn from one of the two
  ranges [1,40] and [1,80].  The eleven instance sizes crossed with the two
  ranges give the 22 configurations of the scale, each of them replicated 20
  times.

* **Methods.**  The MILP model of the paper, which is exact at this size, and the
  three heuristics it is compared with: the 2-approximation of paper Section 4.3,
  the NEH heuristic and the genetic algorithm.  Every one of them is read on its
  gap against the reference value LB of Section 4.3, which the instance alone
  determines, and on its running time.  The dynamic program of the paper is not
  run here: its cost grows exponentially in the number of distinct weights, so
  instances with thirty of them are out of its reach, and it is reported on a set
  of its own, which ``run_dp.py`` runs.  Every method runs once on every
  instance, so no comparison rests on a different sample.

* **Parameters.**  Those of paper Section 6: processing times U(1,40) and U(1,80);
  a time limit of 600 s shared by every method; and Gurobi on a single thread.

* **Best count.**  Once every method of a configuration has run, the best value
  found on each instance is determined and kept with the record.  Wherever the
  model closed the instance that value is the optimum, and the count of instances
  on which a method reached it is the number it settled to optimality; a
  configuration in which a method established no optimum at all is marked "none".
  The paper reports this count for the dynamic program alone, since that is the
  set the exact method settles in full.

* **Continuation.**  The record file is written instance by instance, so an
  interrupted run is continued rather than repeated: started again, the driver
  takes up the newest unfinished record file of the scale, reads back the groups
  of instances it already holds and runs only what is missing.  --fresh starts a
  new file whatever the earlier runs left behind, and --resume takes one up by
  name.

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
    论文/F2/代码/milp.py                              <- the exact method

Usage
-----
    python run_small.py                             # the whole scale
    python run_small.py --list                      # what would be run
    python run_small.py --algos milp                # one method
    python run_small.py --instances 2               # a quick check
    python run_small.py --fresh                     # start over, do not continue
"""

from __future__ import annotations

import argparse
import csv
import re
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
#: value of the model itself, "base" the lower bound LB of paper Section 4.3
BASELINE_LABEL = {"milp": "MILP", "base": "LB"}

#: the name the 2-approximation of paper Section 4.3 is reported under: its
#: benchmark lives in the module of the genetic algorithm, which records it under
#: a label of its own so that the two are told apart in the record file
APPROX2_LABEL = "2-APX"

#: the scale this file runs; there is only one, so the name is a constant
SCALE = "small"

#: the weights of an instance: drawn uniformly from {1, ..., K}, so that an
#: instance carries at most K distinct ones, and fewer when it has fewer jobs
WEIGHT_COUNT = 30

#: the processing-time ranges of the scale, by the upper end of the discrete
#: uniform range both processing times are drawn from; the two of them are
#: crossed with the instance sizes
RANGES = (40, 80)

#: the instance sizes of the scale
SIZES = (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)

#: The configuration of this scale, and the only place where it is written down:
#: the eleven instance sizes crossed with the two processing-time ranges, at the
#: single weight count, 20 instances each, for the 22 configurations and 440
#: instances of the scale.  "configs" holds one (n, K, processing-time range)
#: triple per configuration; "instances" is the replication count; "methods" the
#: model and the heuristics it is compared with; and "baseline" the reference
#: value every one of them is read on, which is the lower bound LB of paper
#: Section 4.3, computed from the instance rather than run.
SCALES = {
    "small": {
        "instances": 20,
        "methods": ("milp", "approx2", "neh", "ga"),
        "baseline": "base",
        "configs": tuple((n, WEIGHT_COUNT, proc_hi)
                         for n in SIZES
                         for proc_hi in RANGES),
    },
}


# ---------------------------------------------------------------------------
# runners: one thin adapter per algorithm, each calling that module's benchmark
# ---------------------------------------------------------------------------

def _run_milp(n, cfg, instances, seed, opts, progress=None, first=1):
    from milp import benchmark
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed, first=first,
                     weights=cfg["weights"], geometric=cfg["geometric"],
                     proc_hi=cfg["proc_hi"],
                     time_limit=opts.time_limit, progress=progress,
                     log_path=getattr(opts, "record_path", None))


def _run_neh(n, cfg, instances, seed, opts, progress=None, first=1):
    from NEH import benchmark
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed, first=first,
                     weights=cfg["weights"], geometric=cfg["geometric"],
                     proc_hi=cfg["proc_hi"],
                     progress=progress,
                     log_path=getattr(opts, "record_path", None))


def _run_approx2(n, cfg, instances, seed, opts, progress=None, first=1):
    from GA import benchmark
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed, first=first,
                     weights=cfg["weights"], geometric=cfg["geometric"],
                     proc_hi=cfg["proc_hi"],
                     method="2-approx", label=APPROX2_LABEL,
                     progress=progress,
                     log_path=getattr(opts, "record_path", None))


def _run_ga(n, cfg, instances, seed, opts, progress=None, first=1):
    from GA import benchmark
    kwargs = {"N": max(8, GA_SETTINGS["N_factor"] * n),   # test mode, or the paper
              "G": GA_SETTINGS["G"]}                      # values of Section 6
    if opts.ga_pop is not None:
        kwargs["N"] = opts.ga_pop
    if opts.ga_gen is not None:
        kwargs["G"] = opts.ga_gen
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed, first=first,
                     weights=cfg["weights"], geometric=cfg["geometric"],
                     proc_hi=cfg["proc_hi"],
                     progress=progress, log_path=getattr(opts, "record_path", None),
                     **kwargs)


# ---------------------------------------------------------------------------
# settings of the metaheuristic
# ---------------------------------------------------------------------------
#
# The parameters of paper Section 6, fixed by the calibration run that paragraph
# reports.  A quicker check of the pipeline is a matter of the command line --
# fewer instances (--instances), a single scale (--scale), or a smaller budget
# for a method (--ga-pop, --ga-gen), which override what
# is below:
#
#   GA   N = n, G = 10 reproduces the mean objective of N = 2n, G = 30 on both
#        weight regimes, at a quarter of the running time: the population reaches
#        its fixed point within a few generations, and even N = n/2, G = 10
#        still matches.
#: GA arguments: population size as a multiple of n, and generations
GA_SETTINGS = {"N_factor": 1, "G": 10}


#: output order of the methods, and the runner of each.  The baseline of the
#: heuristic scale is not among them: it is the reference value of the schedule a
#: method reports, which each algorithm computes on the instance it is handed
#: (see instance_baseline() in the algorithm modules).
METHODS = {
    "milp":    ("MILP",     _run_milp),
    "approx2": ("2-approx", _run_approx2),
    "neh":     ("NEH",      _run_neh),
    "ga":      ("GA",       _run_ga),
}
#: the scale run when none is named: this file has only the one
DEFAULT_SCALES = [SCALE]


# ---------------------------------------------------------------------------
# experiment driver
# ---------------------------------------------------------------------------

def config_of(n, K, proc_hi):
    """
    The arguments of one configuration, and the labels its row is written with.

    K is the number of weights the instance may take, drawn uniformly from
    {1, ..., K} -- the rule of paper Section 6.1 for this scale, where the count
    is arbitrary and no larger than 30 -- and proc_hi the upper end of the range
    the processing times are drawn from.
    """
    return {
        "K": K,
        "weights": "narrow",
        "geometric": False,
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

#: the per-instance line of the record file, as the algorithms write it: the
#: moment the instance finished, the method, the configuration, the seed, the
#: number of the instance, the value, the gap, the status and the running time
INSTANCE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+(\S+)\s+n=(\d+)\s+K=(\d+)\s+"
    r"\[1,(\d+)\]\s+\|\s+seed=(\d+)\s+\|\s+instance\s+(\d+)/(\d+)\s+\|\s+"
    r"value\s+(\S+)\s+\|\s+gap\s+(\S+)\s+\|\s+(.+?)\s+([\d.]+) s$")

#: the summary line this driver writes once a configuration is done; it is what
#: marks a configuration as run to the end, and hence as not to be run again
SUMMARY_RE = re.compile(
    r"^\s*\[\s*(\d+)/(\d+)\]\s+(\S+)\s+n=(\d+)\s+K=(\d+)\s+\[1,(\d+)\]\s+\|\s+"
    r"opt\s+(\S+)")

#: method name of each label, so that the lines of a record file can be traced
#: back to the method that wrote them
NAME_OF_LABEL = {label: name for name, (label, _) in METHODS.items()}
#: the label the 2-approximation is recorded under, which its benchmark writes
#: instead of the name of the module it lives in
NAME_OF_LABEL[APPROX2_LABEL] = "approx2"


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

    Returns a dict keyed by (method, n, K, range) whose values are, for every
    instance of the group, its value, its running time and whether the method
    established its optimality; together with the replication count of the group,
    the number of the first instance the file does not hold, and whether the
    group is finished.  A group whose method is unknown here is left out.
    """
    done, order = {}, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = INSTANCE_RE.match(line)
        if m:
            name = NAME_OF_LABEL.get(m.group(1))
            if name is None:
                continue
            key = (name, int(m.group(2)), int(m.group(3)), int(m.group(4)))
            entry = done.get(key)
            if entry is None:
                entry = done[key] = {"instances": int(m.group(7)),
                                     "rows": [None] * int(m.group(7)),
                                     "summarised": False}
                order.append(key)
            # A method reports a value only when it found a schedule, and a
            # running time only for the instances it found one on, whether or not
            # it went on to close them.
            value = m.group(8)
            entry["rows"][int(m.group(6)) - 1] = {
                "obj": None if value == "none" else float(value),
                "time": float(m.group(11)) if value != "none" else None,
                "optimal": m.group(10) == "optimal",
            }
            continue
        m = SUMMARY_RE.match(line)
        if m:
            key = (NAME_OF_LABEL.get(m.group(3)), int(m.group(4)),
                   int(m.group(5)), int(m.group(6)))
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
        # the model reports the instances whose search it closed, the dynamic
        # program every instance it finished, being exact
        "optimal": sum(1 for row in run["rows"] if row["optimal"]),
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
    if "optimal" in res:
        out["optimal"] = res["optimal"] + sum(1 for row in prefix if row["optimal"])
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

    A configuration can take an hour or more -- the model at the largest size of
    the scale may run into the time limit on each of twenty instances -- so the
    driver would otherwise be silent for the whole of it.  The reporter carries
    the running time of the instance and the time elapsed since the method
    started on this configuration.  A method may also hand it the reference of
    the instance, the lower bound every method of this scale is measured
    against; the line then carries the gap of the instance, and a method that
    reports no reference shows a dash in that column.
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
    # The configurations an earlier, interrupted run finished, read back from the
    # record file it left behind: they are not run again, and this run starts at
    # the first configuration that file has no summary for.
    earlier = getattr(args, "resumed", None) or {}

    # The order of the runs follows the order of the tables of Section 6.2: the
    # methods one after the other and, inside a method, the configurations one
    # after the other.  Every method of a configuration is run from the same seed
    # and hence on the same instances, whatever the order they are asked for in.
    state = {}
    for name in methods:
        label, runner = METHODS[name]
        for K in sorted({k for _, k, _ in spec["configs"]}):
            for (n, k, proc_hi) in spec["configs"]:
                if k != K:
                    continue
                cfg = config_of(n, k, proc_hi)
                seed = seed_of(n, k, proc_hi, args.seed)
                wlabel, rlabel = cfg["weight_label"], cfg["range_label"]
                done += 1
                run = earlier.get((name, n, k, proc_hi))
                if run is not None and run["instances"] != instances:
                    raise SystemExit(
                        "record file %s holds %d instances of %s n=%d K=%d [1,%d], "
                        "not %d: take the run up again with the replication count "
                        "it was started with"
                        % (args.record_path, run["instances"], label, n, k,
                           proc_hi, instances))
                if done > 1 and (run is None or not run["complete"]):
                    print(RULE, file=stream, flush=True)   # separate the setups
                if run is not None and run["complete"]:
                    # Run to the end by the run this one continues: the instances
                    # come from its record file, and neither the header nor the
                    # summary of the group is written again, as the file holds both
                    res = _from_rows(run, instances)
                else:
                    # The group is run on from the first instance the file does
                    # not hold, so that an interruption inside a group costs only
                    # the instance the run was solving when it stopped
                    head = ("  n=%-4d %-6s %-7s | %s (seed %d)"
                            % (n, wlabel, rlabel, label, seed))
                    print(head, file=stream, flush=True)
                    _record(record_path, head)
                    res = runner(n, cfg, instances, seed, args,
                                 _instance_reporter(label, n, wlabel, rlabel, stream),
                                 first=1 if run is None else run["first"])
                    if run is not None:
                        res = _continued(run, res, run["first"])
                state[(name, n, k, proc_hi)] = res

                missing = sum(1 for o in res["objs"] if o is None)
                # The instances a method settled to optimality, which the paper
                # reads only for the dynamic program: the model counts those whose
                # search it closed, a method that reports none of them none at all.
                opt = res.get("optimal", res.get("solved", 0))
                # Every method of this scale reports the reference value of the
                # instance it was handed, and is read on its gap against it.
                gaps, _ = gaps_against(res["objs"], res.get("bases") or [])
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
                    "gaps": gaps,
                    "mean_gap": statistics.fmean(gaps) if gaps else float("nan"),
                    "min_gap": min(gaps) if gaps else float("nan"),
                    "max_gap": max(gaps) if gaps else float("nan"),
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
                o_mean, o_min, o_max = stat_obj or (float("nan"),) * 3
                t_mean, t_min, t_max = stat_time or (float("nan"),) * 3
                g_mean, g_min, g_max = ((rec["mean_gap"], rec["min_gap"],
                                         rec["max_gap"]) if gaps
                                        else (float("nan"),) * 3)
                # The summary line of the configuration, written to the record
                # file as it is printed, so that the file carries the totals of
                # every group of instances as well as the instances themselves.
                summary = ("  [%2d/%2d] %-4s n=%-4d %-6s %-7s | opt %-7s |%s"
                           " obj %9.1f (%8.1f, %9.1f) "
                           "| time %9.4f s (%8.4f, %10.4f) | no value %d"
                           % (done, total, label, n, wlabel, rlabel,
                              "%d/%d" % (opt, instances) if opt else "none",
                              "" if not gaps else
                              " gap %+8.3f%% (%+8.3f%%, %+9.3f%%) |"
                              % (g_mean, g_min, g_max),
                              o_mean, o_min, o_max, t_mean, t_min, t_max, missing))
                if run is None or not run["complete"]:
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
        "baseline      : %s, the lower bound of paper Section 4.3, which the"
        " instance determines" % BASELINE_LABEL["base"],
        "seed          : %d (+1000*n, + 100*K, + range)" % args.seed,
        "time limit    : %s s" % (args.time_limit if args.time_limit else "none"),
        "records       : %s" % getattr(args, "record_path", "-"),
        "resumed       : %s" % (getattr(args, "record_path", "-")
                                if getattr(args, "resumed", None) else "-"),
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
    lines.append("# %-5s %5s %-7s %-7s %9s %8s %23s %23s %31s"
                 % ("algo", "n", "weights", "range", "instances", "opt",
                    "gap % mean (min, max)", "objective mean (min, max)",
                    "time(s) mean (min, max)"))
    for r in records:
        o = r["obj"] or (float("nan"),) * 3
        t = r["time"] or (float("nan"),) * 3
        lines.append("  %-5s %5d %-7s %-7s %9d %8s %8.3f (%8.3f, %9.3f)"
                     " %10.1f (%8.1f, %9.1f) %12.4f (%8.4f, %10.4f)"
                     % (r["label"], r["n"], "K=%d" % r["K"], r["range"],
                        r["instances"],
                        "%d/%d" % (r["opt"], r["instances"]) if r["opt"] else "none",
                        r["mean_gap"], r["min_gap"], r["max_gap"],
                        o[0], o[1], o[2], t[0], t[1], t[2]))
    lines.append("#")
    lines.append("# the column 'opt' counts the instances, out of those of the configuration,")
    lines.append("# whose best value found, the 'best' line below, the method itself reached:")
    lines.append("# the model counts the ones whose search it closed.  'none' marks a")
    lines.append("# configuration in which the method reached none of them.  The paper reports")
    lines.append("# this column for the dynamic program alone, whose set is the one it settles.")
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
                   help="solver time limit in seconds for MILP "
                        "(default: 600, the limit of paper Section 6)")
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
        resume_path = unfinished_record(RECORD_DIR, "-".join(scales),
                                        args.instances
                                        or SCALES[scales[0]]["instances"])
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
        args.record_path = RECORD_DIR / ("实例记录_%s_%s.txt"
                                         % ("-".join(scales), stamp))
    else:
        args.record_path = resume_path

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
    if args.resumed:
        left = [(key, run) for key, run in sorted(args.resumed.items())
                if not run["complete"]]
        print("  resuming  : %d groups already in that file%s"
              % (len(args.resumed) - len(left),
                 "" if not left else "; to continue: %s"
                 % ", ".join("%s n=%d/K=%d/[1,%d] from instance %d"
                             % (METHODS[m][0], n, k, hi, run["first"])
                             for (m, n, k, hi), run in left)))
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
