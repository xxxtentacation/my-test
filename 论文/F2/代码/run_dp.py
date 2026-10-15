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
  with the weight count K = 2, whose weights are the geometric ladder {1, 10},
  so that the weight classes are decades apart, and with the three
  processing-time ranges [1,20], [1,40] and [1,60].  The nine configurations are
  each replicated 20 times, for the 180 instances of the group.

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

* **Record.**  The record file is written instance by instance, so an interrupted
  run leaves behind every instance it has already solved.  Every run starts a
  file of its own and runs the whole group, in the order of paper Section 6.2.

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
DEFAULT_OUT = RECORD_DIR / "结果_dp_最新.txt"
#: every run is also filed here under its date and group, so that no run
#: overwrites the results of an earlier one
ARCHIVE_DIR = RECORD_DIR / "结果存档"

#: the group this file runs; there is only one, so the name is a constant
SCALE = "dp"

#: the number of distinct weights of the group, one configuration each: the
#: weights are drawn from the geometric ladder {1, 10, ..., 10^(K-1)}, and the
#: group runs at the smallest count the ladder allows, where the state space the
#: exact method builds stays within reach
WEIGHT_COUNTS = (2,)

#: the processing-time ranges of the group, by the upper end of the discrete
#: range on the integers of which both processing times are drawn; the three are
#: crossed with the instance sizes
RANGES = (20, 40, 60)

#: the instance sizes of the group
SIZES = (20, 40, 60)

#: The configuration of this group, and the only place where it is written down:
#: the three instance sizes crossed with the weight count and the three
#: processing-time ranges, 20 instances each, for the 9 configurations and 180
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
    reach -- and proc_hi the upper end of the range the processing times are
    drawn from.
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

    A configuration can take an hour or more -- the dynamic program at the
    largest size of the group may build a state space of hours on a single
    instance -- so the driver would
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
                 "%.2f ms" % (1000.0 * (seconds or 0.0)),
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

    # The order of the runs follows the order of the tables of Section 6.2: the
    # configurations one after the other, every one of them run to the end.
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
                if done > 1:
                    print(RULE, file=stream, flush=True)   # separate the setups
                head = ("  n=%-4d %-6s %-7s | %s (seed %d)"
                        % (n, wlabel, rlabel, label, seed))
                print(head, file=stream, flush=True)
                _record(record_path, head)
                res = runner(n, cfg, instances, seed, args,
                             _instance_reporter(label, n, wlabel, rlabel, stream))
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
                           "| time %9.2f ms (%8.2f, %10.2f) | no value %d"
                           % (done, total, label, n, wlabel, rlabel,
                              "%d/%d" % (opt, instances) if opt else "none",
                              o_mean, o_min, o_max, 1000.0 * t_mean, 1000.0 * t_min, 1000.0 * t_max, missing))
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
                    "objective mean (min, max)", "time(ms) mean (min, max)"))
    for r in records:
        o = r["obj"] or (float("nan"),) * 3
        t = r["time"] or (float("nan"),) * 3
        lines.append("  %-5s %5d %-7s %-7s %9d %8s"
                     " %10.1f (%8.1f, %9.1f) %12.4f (%8.4f, %10.4f)"
                     % (r["label"], r["n"], "K=%d" % r["K"], r["range"],
                        r["instances"],
                        "%d/%d" % (r["opt"], r["instances"]) if r["opt"] else "none",
                        o[0], o[1], o[2],
                        1000.0 * t[0], 1000.0 * t[1], 1000.0 * t[2]))
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
                        " ".join("none" if t is None else "%.4f" % (1000.0 * t)
                                 for t in r["times"])))
    lines.append("#")
    lines.append("# summary, averaged over the configurations of the scale")
    lines.append("# %-8s %14s %6s %13s"
                 % ("method", "configurations", "opt", "mean_time(ms)"))
    for s in summary:
        lines.append("  %-8s %14d %6d %13.4f"
                     % (s["label"], s["configurations"], s["opt"],
                        1000.0 * s["mean_time"]))


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
              % ("method", "configurations", "opt", "mean_time(ms)"))
        for s in summary:
            print("  %-8s %14d %6d %13.4f"
                  % (s["label"], s["configurations"], s["opt"], 1000.0 * s["mean_time"]))
    print()
    print("results written to %s" % out_path)
    print("archived as        %s" % archive_path)


if __name__ == "__main__":
    main()
