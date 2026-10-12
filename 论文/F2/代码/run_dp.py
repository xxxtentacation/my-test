# -*- coding: utf-8 -*-
"""
Driver for the set of instances the dynamic program is reported on, paper
Section 6.

This is the first of the three groups of the computational study: the small scale
is run by ``run_small.py`` and the large one by ``run_large.py``.  The three
scripts share the driver below and differ only in the configurations and the
methods they run, so a group can be run, interrupted and repeated on its own.

The script follows the design of paper Section 6.1 rather than inventing its
own:

* **Instances.**  The set the dynamic program is reported on is the one of the
  study that preceded this one: the three instance sizes 20, 40 and 60, crossed
  with the two weight counts K = 2 and K = 3, whose weights are the geometric
  ladder {1, 10} and {1, 10, 100}, so that the weight classes are decades apart,
  and with the three processing-time ranges [1,20], [1,40] and [1,60].  The
  eighteen configurations are each replicated 20 times, for the 360 instances of
  the group.

* **Methods.**  The dynamic program of the paper alone.  Its running time
  O(n K V^(3K-3)) grows so fast in the number K of distinct weights that the
  sizes the other algorithms are read at are out of its reach, so it is not
  compared with them and is reported on a set of its own.  Being exact, it is read
  on the number of instances whose optimum it establishes and on its running
  time, rather than on a gap.

* **Parameters.**  Those of paper Section 6: processing times U(1,20), U(1,40)
  and U(1,60), and a time limit of 600 s, which bounds the state space an
  instance may build rather than the search itself.

* **Optimum count.**  Once the method has run on a configuration, the count of
  instances whose optimum it established is what the two tables of the dynamic
  program report, together with the mean, the smallest and the largest objective
  value of the configuration and the same three of its running time.  A method
  that establishes no optimum at all in a configuration is marked "none".

* **Continuation.**  The record file is written instance by instance, so an
  interrupted run is continued rather than repeated: started again, the driver
  takes up the newest unfinished record file of the group, reads back the
  configurations it already holds and runs only what is missing.  --fresh starts
  a new file whatever the earlier runs left behind, and --resume takes one up by
  name.

* **Seed.**  The seed of a configuration depends only on (n, K, range), which is
  the rule the earlier study used as well, so a configuration is run on the very
  instances it was run on then.

* **Output.**  The run is reported line by line as it happens: one line per
  instance, carrying the value found, whether the method closed the instance, the
  running time of that instance and the time elapsed since the method started on
  the configuration, and one line per configuration once it has run.  stdout is
  put in line-buffered mode, so the same holds when the output is redirected to a
  file.

The run is written to the results file, and a copy is filed away under the date
and the group, so that running the script again never overwrites the results of
an earlier run::

    论文/F2/代码/记录/结果_dp_最新.txt                  <- the run that just finished
    论文/F2/代码/记录/结果存档/<date>_dp.txt             <- one file per run, kept
    论文/F2/代码/记录/实例记录_dp_<date>.txt             <- one line per instance, live
    论文/F2/代码/run_dp.py                            <- this script
    论文/F2/代码/dp.py                                <- the method it runs

Usage
-----
    python run_dp.py                                # the whole group
    python run_dp.py --list                         # what would be run
    python run_dp.py --instances 2                  # a quick check
    python run_dp.py --fresh                        # start over, do not continue

This group was first run by the script that has since become ``run_small.py``,
and the record file of that run, an ``实例记录_small_*.txt`` of the record
folder, holds the configurations it already closed.  It is taken up by name, so
that they are read back rather than run again::

    python run_dp.py --resume 记录/实例记录_small_2026-10-09_15-08-55.txt
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
DEFAULT_OUT = RECORD_DIR / "结果_dp_最新.txt"
#: every run is also filed here under its date and group, so that no run
#: overwrites the results of an earlier one
ARCHIVE_DIR = RECORD_DIR / "结果存档"

#: the group this file runs; there is only one, so the name is a constant
SCALE = "dp"

#: the numbers of distinct weights of the group, one configuration each: the
#: weights are drawn from the geometric ladder {1, 10, ..., 10^(K-1)}
WEIGHT_COUNTS = (2, 3)

#: the processing-time ranges of the group, by the upper end of the discrete
#: uniform range both processing times are drawn from; the three of them are
#: crossed with the instance sizes
RANGES = (20, 40, 60)

#: the instance sizes of the group
SIZES = (20, 40, 60)

#: The configuration of this group, and the only place where it is written down:
#: the three instance sizes crossed with the two weight counts and the three
#: processing-time ranges, 20 instances each, for the 18 configurations and 360
#: instances of the group.  "configs" holds one (n, K, processing-time range)
#: triple per configuration; "instances" is the replication count; "methods" the
#: method the group is reported on; and "baseline" is None, the dynamic program
#: being exact and read on the number of instances it settles rather than on a
#: gap against a reference value.
SCALES = {
    "dp": {
        "instances": 20,
        "methods": ("dp",),
        "baseline": None,
        "configs": tuple((n, K, proc_hi)
                         for n in SIZES
                         for K in WEIGHT_COUNTS
                         for proc_hi in RANGES),
    },
}


# ---------------------------------------------------------------------------
# runners: one thin adapter per algorithm, each calling that module's benchmark
# ---------------------------------------------------------------------------

def _run_dp(n, cfg, instances, seed, opts, progress=None, first=1):
    from dp import benchmark
    return benchmark(n=n, K=cfg["K"], instances=instances, seed=seed, first=first,
                     weights=cfg["weights"], geometric=cfg["geometric"],
                     proc_hi=cfg["proc_hi"],
                     time_limit=opts.time_limit, progress=progress,
                     log_path=getattr(opts, "record_path", None))


#: output order of the methods, and the runner of each: the group is reported on
#: the dynamic program alone
METHODS = {
    "dp":      ("DP",       _run_dp),
}
#: the group run when none is named: this file has only the one
DEFAULT_SCALES = [SCALE]


# ---------------------------------------------------------------------------
# experiment driver
# ---------------------------------------------------------------------------

def config_of(n, K, proc_hi):
    """
    The arguments of one configuration, and the labels its row is written with.

    K is the number of weights the instance may take, drawn from the geometric
    ladder {1, 10, ..., 10^(K-1)} -- the rule of paper Section 6.1 for this group,
    where the weight classes are decades apart and the exact method stays within
    reach at K = 2 and K = 3 -- and proc_hi the upper end of the range the
    processing times are drawn from.
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

    A configuration can take an hour or more -- the dynamic program at K = 3 may
    build a state space of hours on a single instance -- so the driver would
    otherwise be silent for the whole of it.  The reporter carries whether the
    method closed the instance, the running time of the instance and the time
    elapsed since the method started on this configuration.  Being exact, the
    method reports no gap against a reference, so the line carries none.
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


def run_scale(scale, methods, args, stream=sys.stdout):
    """
    Run the method on every configuration of `scale`.

    Returns a list of records, one per configuration, each of them carrying the
    objective values and the running times of its instances, the number of them
    whose optimum the method established, and the same statistics of the two.
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
                # The instances the method established the optimum of, which is
                # what the two tables of the dynamic program report: every
                # instance it finished within the time limit is one it settled.
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
                o_mean, o_min, o_max = stat_obj or (float("nan"),) * 3
                t_mean, t_min, t_max = stat_time or (float("nan"),) * 3
                # The summary line of the configuration, written to the record
                # file as it is printed, so that the file carries the totals of
                # every group of instances as well as the instances themselves.
                summary = ("  [%2d/%2d] %-4s n=%-4d %-6s %-7s | opt %-7s |"
                           " obj %9.1f (%8.1f, %9.1f) "
                           "| time %9.4f s (%8.4f, %10.4f) | no value %d"
                           % (done, total, label, n, wlabel, rlabel,
                              "%d/%d" % (opt, instances) if opt else "none",
                              o_mean, o_min, o_max, t_mean, t_min, t_max, missing))
                if run is None or not run["complete"]:
                    print(summary, file=stream, flush=True)
                    _record(record_path, summary)

    # The best value found on each instance, now that the method has run on it:
    # the dynamic program is exact, so that value is the optimum of the instance.
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
        "reference     : the optimum of the instance, which the dynamic program"
        " establishes itself",
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
    lines.append("# %-5s %5s %-7s %-7s %9s %8s %23s %31s"
                 % ("algo", "n", "weights", "range", "instances", "opt",
                    "objective mean (min, max)", "time(s) mean (min, max)"))
    for r in records:
        o = r["obj"] or (float("nan"),) * 3
        t = r["time"] or (float("nan"),) * 3
        lines.append("  %-5s %5d %-7s %-7s %9d %8s"
                     " %10.1f (%8.1f, %9.1f) %12.4f (%8.4f, %10.4f)"
                     % (r["label"], r["n"], "K=%d" % r["K"], r["range"],
                        r["instances"],
                        "%d/%d" % (r["opt"], r["instances"]) if r["opt"] else "none",
                        o[0], o[1], o[2], t[0], t[1], t[2]))
    lines.append("#")
    lines.append("# the column 'opt' counts the instances, out of those of the configuration,")
    lines.append("# whose optimum the method established: a value is the optimum of its")
    lines.append("# instance wherever the dynamic program returned one, the method being")
    lines.append("# exact.  'none' marks a configuration in which it established none.")
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
                   help="time limit in seconds for the dynamic program "
                        "(default: 600, the limit of paper Section 6)")
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

    None (no --algos) and "all" both mean the method the group is reported on,
    which here is the dynamic program alone.
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
