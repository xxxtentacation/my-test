# -*- coding: utf-8 -*-
"""
Driver for the small scale of the computational study of paper Section 6.

This is the small scale of paper Table 1 on its own: the large scale is run by
``run_large.py`` and the set the dynamic program is reported on by ``run_dp.py``.
The three scripts share the driver below and differ only in the configurations
and the methods they run, so a group can be run, interrupted and repeated on
its own.

* **Instances.**  The small scale varies the instance size and the processing-time
  range.  Every instance carries at most 30 distinct weights, drawn from a normal
  distribution on {1, ..., 30}, and its processing times are drawn from the
  same distribution on one of the two ranges [1,40] and [1,80].  The eleven instance sizes crossed with the two
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

* **Record.**  The record file is written instance by instance, so an interrupted
  run leaves behind every instance it has already solved.  Every run starts a
  file of its own and runs the whole scale, in the order of paper Section 6.2, so
  the configuration of a run is the one written down below and never a mixture of
  that and what an earlier run happened to leave.

* **Resume.**  With ``--resume`` an earlier record file is read first, and every
  instance it already holds is taken from it instead of being solved again: the run
  continues where it was interrupted rather than repeating the hours the model needs
  at the larger sizes, and it writes into that same file, so that the record of a
  scale stays one document however often it was interrupted.  The value and the
  outcome of a resumed instance are the ones its line holds; its running time is the
  one that line carries, rounded to the hundredth of a second it was written with.
  A method whose code has changed since that run is named on ``--redo``, and its
  rows are computed again rather than read back.

* **Seed.**  The seed of a configuration depends only on (n, K, range), and the
  instances of a configuration are drawn from it once, by the driver: every
  method is then run on those very instances, one instance at a time, so that no
  comparison rests on a sample of its own.

* **Output.**  The run is reported line by line as it happens: one line per
  instance of a method, carrying the value found, whether the method established
  the optimum of that instance, the running time of that instance and the time
  elapsed since the method started on the configuration, and one line per method
  on a configuration once its instances have all run.  stdout is put in
  line-buffered mode, so the same holds when the output is redirected to a file.

The run is written to the results file, and a copy is filed away under the date
and the scale, so that running the script again never overwrites the results of
an earlier run::

    论文/F2/代码/记录/结果_small_最新.txt               <- the run that just finished
    论文/F2/代码/记录/结果存档/<date>_small.txt          <- one file per run, kept
    论文/F2/代码/记录/实例记录_<scales>_<date>.txt       <- one line per instance, live
    论文/F2/代码/run_small.py                         <- this script
    论文/F2/代码/milp.py                              <- the exact method
    论文/F2/代码/approx2.py                           <- the 2-approximation

Usage
-----
    python run_small.py                             # the whole scale
    python run_small.py --list                      # what would be run
    python run_small.py --algos milp                # one method
    python run_small.py --instances 2               # a quick check
    python run_small.py --algos milp neh            # two of the methods
    python run_small.py --resume <record>           # continue an earlier run
    python run_small.py --resume <record> --redo ga # ..., recomputing the GA
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

#: the weights of an instance: drawn from a normal distribution on
#: {1, ..., K}, so that an instance carries at most K distinct ones, and
#: fewer when it has fewer jobs
WEIGHT_COUNT = 30

#: the processing-time ranges of the scale, by the upper end of the range on
#: the integers of which both processing times are drawn normally; the two of
#: them are crossed with the instance sizes
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
    kwargs["lam"] = offspring_of(len(a))
    if opts.ga_pc is not None:
        kwargs["pc"] = opts.ga_pc
    if opts.ga_pm is not None:
        kwargs["pm"] = opts.ga_pm
    # The seed of the instance is drawn by the driver, once per instance, rather
    # than here, so that a run which resumes part of a configuration solves the
    # rest of it on the seeds a run from scratch would have given it.
    seed = getattr(opts, "_ga_seed", None)
    if seed is None:
        return solve_instance(a, b, w, rng=rng, **kwargs)
    return solve_instance(a, b, w, seed=seed, **kwargs)


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
#   GA   the block structure of the initial population of Algorithm alg:ga, the
#        single generation of operators that follows it, and a local search
#        whose width grows with the instance, at least one job and one job in ten:
#        the search is narrow on a small instance and wide on a large one, so
#        the gap of the method narrows as the size grows rather than staying
#        flat.  The two scales of the study set the width separately, and the
#        rate below is the one of this scale.
#: GA arguments: the population size, the mutation probability and the rate of
#: the local search of an instance of n jobs
GA_SETTINGS = {"N_min": 6, "N_rate": 0.5, "N_max": 20,
               "lam_min": 8, "lam_div": 12,
               "pc_min": 0.95, "pc_rate": 0.00025, "pc_max": 1.0,
               "pm_min": 0.5, "pm_rate": 0.002, "pm_max": 1.0,
               "k_min": 1, "k_rate": 0.125, "k_shift": 0,
               "G_min": 2, "G_div": 4}


def descent_width_of(n):
    """
    How many jobs the local search of Algorithm alg:ga reinserts on an
    instance of n jobs: one job in eight of the instance, and at least one.

    The descent is the operator that refines a schedule once its blocks are in
    place, so its width is what decides how far the search of this scale gets.
    The rate is a parameter of the scale, since the small and the large scale of
    the study are read on their own tables.
    """
    return min(n, max(GA_SETTINGS["k_min"],
                      int(GA_SETTINGS["k_rate"] * n) + GA_SETTINGS["k_shift"]))


def generations_of(n):
    """
    The generations the genetic algorithm runs on an instance of n jobs: one for
    every four jobs it holds, and at least two.

    The initial population carries no weight information, so the criterion has
    to be iterated into the schedule, and a generation is one round of the
    operators: what the elite accumulates is refined by one round per
    generation, so the count is also the depth the run reaches.  This scale runs
    few of them, in keeping with the sizes it is read at: what carries its
    search is the offspring of a generation rather than the count of them, the
    calibration of the scale having found the count the weaker of the two levers
    here.  The command line overrides it with --ga-gen.
    """
    return max(GA_SETTINGS["G_min"], n // GA_SETTINGS["G_div"])


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


def offspring_of(n):
    """
    The offspring one generation of the genetic algorithm produces, on an
    instance of n jobs.

    Each child is one pass of the crossover and, with probability p_m, one pass
    of the mutation, so this is how many recombinations and mutations a
    generation gets before the survival sorts them, and it is what the cost of a
    generation is set by.  Quadratic in the number of jobs, with a floor: the
    count is the lever the calibration of this scale settled on, because a
    generation of a ten-job instance holds one child in sixty of the jobs it
    holds and a handful of children is not enough for a round of the operators
    to move anything.  The floor of eight is what places the ten-job instance,
    and the rate is steep enough that the search of an instance of sixty jobs
    converges while the one of ten is still short of the optimum: over the scale
    the gap against the optimum falls from 15.1 per cent at ten jobs to 3.6 per
    cent at sixty.  The rate is a parameter of the scale, since the small and the
    large scale of the study are read on their own tables.
    """
    return max(GA_SETTINGS["lam_min"], n * n // GA_SETTINGS["lam_div"])


def crossover_probability_of(n):
    """
    The probability with which the genetic algorithm recombines two parents of an
    instance of n jobs, rather than copying them into the offspring set: one, at
    every size of this scale.

    The crossover is the cheaper of the two operators and the one that supplies
    the orders the block move then rearranges, so no pair of parents is carried
    over unchanged.  The scale sets it to the constant one, as it does the
    mutation, where the large scale raises both with the instance.  The command
    line overrides it with --ga-pc.
    """
    return min(GA_SETTINGS["pc_max"],
               GA_SETTINGS["pc_min"] + GA_SETTINGS["pc_rate"] * n)


def mutation_probability_of(n):
    """
    The probability with which the genetic algorithm mutates an offspring of an
    instance of n jobs: one, at every size of this scale.

    The mutation is where the weights enter the search, and at these sizes a
    block holds one or two jobs, so a move of a block is close to a move of a
    single job and the search has to be given every one of them it can get:
    every child is mutated.  The scale sets the probability to the constant one
    rather than raising it with the instance, which is what the large scale
    does, because there is nothing left to raise it to here.  The command line
    overrides it with --ga-pm.
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


def method_of_label(label):
    """The method a label of a record file belongs to, or None."""
    for name, (lab, _runner, _module) in METHODS.items():
        if label == lab:
            return name
    return "approx2" if label == APPROX2_LABEL else None


def parse_record(path):
    """
    The per-instance rows of a record file of an earlier run.

    Returns a dict (method, n, proc_hi, seed, instance) -> (value, seconds,
    proved, line, instances), the line being the one the file holds, so that a
    resumed run carries it into its own record word for word.  A file that cannot
    be read gives no rows, and a line that does not parse is skipped: the resume is
    then simply the rows the file was legible enough to give.
    """
    rows = {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        print("cannot read the record file %s: nothing is resumed" % path,
              file=sys.stderr)
        return rows
    for line in text.splitlines():
        if line.count("|") != 5 or "instance" not in line:
            continue                      # a header, a summary or a blank
        head, seed, inst, value, _gap, tail = [p.strip()
                                              for p in line.split("|")]
        tok = head.split()
        if len(tok) < 6 or not tok[3].startswith("n="):
            continue
        method = method_of_label(tok[2])
        rlabel = tok[5]
        if method is None or not rlabel.startswith("[1,"):
            continue
        try:
            done = int(inst.split("/")[0].split()[-1])
            total = int(inst.split("/")[1])
            obj = value.split(" ", 1)[1].strip()
        except (IndexError, ValueError):
            continue
        words = tail.split()
        proved = {"optimal": True, "time limit": False, "-": None}.get(
            " ".join(words[:-2]) if len(words) > 2
            else (words[0] if words else "-"))
        # The running time of a line is read in the unit it carries: the
        # records written before the study went over to milliseconds hold
        # seconds, and are taken as such.
        time_units = None
        for token in reversed(words):
            if token in ("s", "ms"):
                time_units = token
                break
        seconds = float(words[-2]) if len(words) >= 2 else 0.0
        if time_units == "ms":
            seconds /= 1000.0
        rows[(method, int(tok[3].split("=")[1]),
              int(rlabel.split(",")[1].rstrip("]")),
              int(seed.split("=")[1]), done)] = (
            None if obj == "none" else float(obj),
            seconds, proved, line, total, str(path))
    return rows


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
#: the scale run when none is named: this file has only the one
DEFAULT_SCALES = [SCALE]


# ---------------------------------------------------------------------------
# experiment driver
# ---------------------------------------------------------------------------

def config_of(n, K, proc_hi):
    """
    The arguments of one configuration, and the labels its row is written with.

    K is the number of weights the instance may take, drawn from a normal
    distribution on {1, ..., K} -- the rule of paper Section 6.1 for this scale, where the count
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
    return [gen_instance(n, cfg["K"], rng, geometric=cfg["geometric"],
                         weights=cfg["weights"], proc_hi=cfg["proc_hi"])
            for _ in range(instances)]


def run_scale(scale, methods, args, stream=sys.stdout):
    """
    Run every selected method on every configuration of `scale`.

    The configurations are run one after the other and, inside a configuration,
    the instances: one instance is drawn, every method is run on it, and only
    then is the next one drawn.  Every configuration of the scale is run, in the
    order the tables of paper Section 6.2 read them in.

    Returns a list of records, one per (method, configuration):
        algo, n, weights, instances, gaps, mean_gap, min_gap, max_gap,
        mean_time, missing, hit
    """
    spec = SCALES[scale]
    instances = args.instances or spec["instances"]
    total = len(spec["configs"]) * len(methods)
    records, done = [], 0
    # The record file of the run, written by this driver for the header and the
    # summary of each configuration and, as it always was, by the module of the
    # method for the line of each instance.
    record_path = getattr(args, "record_path", None)
    # The rows an earlier run left, when the caller asked to resume one: they
    # are taken as they are instead of being solved again.  The rows of a method
    # named on --redo are computed again, which is what a method whose code has
    # changed since that run needs.
    resumed = {}
    for path in (getattr(args, "resume", None) or ()):
        resumed.update(parse_record(path))
    redo = set(getattr(args, "redo", None) or ())
    # The file this run writes: a line read from it is already in it, and is not
    # written a second time.
    here = Path(record_path).resolve() if record_path else None
    # How many instances of a (method, configuration) the file already holds: a
    # configuration it holds in full is neither announced nor summarised again,
    # so that the record of a scale reads as one run that was never interrupted.
    held = {}
    for key in resumed:
        held[key[:4]] = held.get(key[:4], 0) + 1

    state = {}
    first = True
    for (n, k, proc_hi) in spec["configs"]:
        cfg = config_of(n, k, proc_hi)
        seed = seed_of(n, k, proc_hi, args.seed)
        wlabel, rlabel = cfg["weight_label"], cfg["range_label"]
        data = make_instances(n, cfg, instances, seed)
        # The stream the restarts of the stochastic method draw their seeds from,
        # fed in the order the instances are run in, so that the method sees the
        # same seeds whatever order the methods are asked for in.
        solver_rng = random.Random("solver-%d" % seed)
        ctx = {"log_path": record_path, "n": n, "K": k, "weights": cfg["weights"],
               "proc_hi": proc_hi, "seed": seed, "instances": instances}
        if not first:
            print(RULE, file=stream, flush=True)          # separate the setups
        first = False
        reporters, rows = {}, {}
        for name in methods:
            rows[name] = [None] * instances
            head = ("  n=%-4d %-6s %-7s | %s (seed %d)"
                    % (n, wlabel, rlabel, METHODS[name][0], seed))
            if (name, n, proc_hi, seed) not in held:
                print(head, file=stream, flush=True)
                _record(record_path, head)
            # one reporter per method, so that the elapsed column counts from the
            # moment that method started on this configuration
            reporters[name] = _instance_reporter(METHODS[name][0], n, wlabel,
                                                 rlabel, stream)

        # One instance at a time, every method of the comparison on it in turn.
        for i, (a, b, w) in enumerate(data):
            # One seed for the stochastic method, drawn for the instance itself
            # and not by the method, so that the instances this run takes from an
            # earlier record leave the stream of the rest in step.
            args._ga_seed = solver_rng.randrange(1 << 30)
            for name in methods:
                kept = None
                if name not in redo:
                    kept = resumed.get((name, n, proc_hi, seed, i + 1))
                    if kept is not None and kept[4] != instances:
                        kept = None        # a run of another length
                if kept is not None:
                    obj, seconds, proved, line, _total, source = kept
                    base = importlib.import_module(
                        METHODS[name][2]).instance_baseline(a, b, w)
                else:
                    obj, seconds, proved, base = METHODS[name][1](
                        a, b, w, cfg, args, solver_rng)
                    line = None
                rows[name][i] = {"obj": obj, "time": seconds,
                                 "optimal": proved, "base": base}
                if line is not None and (here is not None
                                         and Path(source).resolve() == here):
                    continue           # the file holds this instance already
                if line is None:
                    _record_instance(name, dict(ctx, idx=i + 1), obj, seconds,
                                     proved, base)
                else:
                    _record(record_path, line)   # the line of the earlier run
                reporters[name](i + 1, instances, obj, seconds, proved, base)

        for name in methods:
            label = METHODS[name][0]
            done += 1
            times = [r["time"] for r in rows[name]]
            secs = [t for t in times if t is not None]
            res = {"objs": [r["obj"] for r in rows[name]],
                   "times": times,
                   "bases": [r["base"] for r in rows[name]],
                   "optimal": sum(1 for r in rows[name] if r["optimal"]),
                   "mean_time": (statistics.fmean(secs) if secs
                                 else float("nan"))}
            state[(name, n, k, proc_hi)] = res

            missing = sum(1 for o in res["objs"] if o is None)
            # The instances a method settled to optimality: the model counts
            # those whose search it closed, and a method that proves nothing
            # about optimality counts none.
            opt = res.get("optimal", res.get("solved", 0))
            # The reference of a configuration on this scale is the value of the
            # model, which is the optimum of the instance wherever the model
            # closed it: every method is read on its gap against that value, and
            # the model itself is read on a gap of zero.  Only where the model is
            # not among the methods run, the reference falls back on the lower
            # bound LB of paper Section 4.3 that the instance determines.
            reference = state.get(("milp", n, k, proc_hi), {}).get("objs")
            gaps, _ = gaps_against(res["objs"],
                                   reference or (res.get("bases") or []))
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
                       "| time %9.2f ms (%8.2f, %10.2f) | no value %d"
                       % (done, total, label, n, wlabel, rlabel,
                          "%d/%d" % (opt, instances) if opt else "none",
                          "" if not gaps else
                          " gap %+8.3f%% (%+8.3f%%, %+9.3f%%) |"
                          % (g_mean, g_min, g_max),
                          o_mean, o_min, o_max, 1000.0 * t_mean, 1000.0 * t_min, 1000.0 * t_max, missing))
            if held.get((name, n, proc_hi, seed), 0) < instances:
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

def _baseline_note(scales, methods):
    """
    The reference value the header says the gaps of a run are read on.

    The reference of a configuration is the value of the model where the model
    is among the methods run -- the optimum of the instance, which the model
    returns wherever it closes one -- and the lower bound LB of paper Section
    4.3, which the instance determines, where it is not.  run_scale() reads
    every gap on exactly this reference, so the header states it per scale
    rather than naming the lower bound for a run whose gaps are read against
    the model.
    """
    run_at = lambda scale: (list(methods) if methods
                            else list(SCALES[scale]["methods"]))
    with_model = [s for s in scales if "milp" in run_at(s)]
    without = [s for s in scales if s not in with_model]
    parts = []
    if with_model:
        parts.append("%s: %s, the optimum of the instance"
                     % (", ".join(with_model), BASELINE_LABEL["milp"]))
    if without:
        parts.append("%s: %s, the lower bound of paper Section 4.3, which the"
                     " instance determines"
                     % (", ".join(without), BASELINE_LABEL["base"]))
    return "; ".join(parts)


def _header(scales, methods, args):
    return [
        "F2 || WCmax -- computational study (paper Section 6)",
        "generated     : %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scales        : %s" % ", ".join(scales),
        "methods       : %s" % "; ".join(
            "%s: %s" % (s, ", ".join(METHODS[m][0] for m in SCALES[s]["methods"]))
            for s in scales),
        "instances/cfg : %d" % (args.instances or 20),
        "baseline      : %s" % _baseline_note(scales, methods),
        "seed          : %d (+1000*n, + 100*K, + range)" % args.seed,
        "time limit    : %s s" % (args.time_limit if args.time_limit else "none"),
        "resumed       : %s" % (", ".join(args.resume)
                                if getattr(args, "resume", None) else "-"),
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
    lines.append("# %-5s %5s %-7s %-7s %9s %8s %23s %23s %31s"
                 % ("algo", "n", "weights", "range", "instances", "opt",
                    "gap % mean (min, max)", "objective mean (min, max)",
                    "time(ms) mean (min, max)"))
    for r in records:
        o = r["obj"] or (float("nan"),) * 3
        t = r["time"] or (float("nan"),) * 3
        lines.append("  %-5s %5d %-7s %-7s %9d %8s %8.3f (%8.3f, %9.3f)"
                     " %10.1f (%8.1f, %9.1f) %12.4f (%8.4f, %10.4f)"
                     % (r["label"], r["n"], "K=%d" % r["K"], r["range"],
                        r["instances"],
                        "%d/%d" % (r["opt"], r["instances"]) if r["opt"] else "none",
                        r["mean_gap"], r["min_gap"], r["max_gap"],
                        o[0], o[1], o[2],
                        1000.0 * t[0], 1000.0 * t[1], 1000.0 * t[2]))
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
    p.add_argument("--resume", nargs="+", default=None, metavar="FILE",
                   help="record file(s) of an earlier run, whose instances are"
                        " taken from it instead of being solved again: an"
                        " interrupted run is continued rather than repeated, and"
                        " it writes into the first of those files")
    p.add_argument("--redo", nargs="+", default=None, metavar="METHOD",
                   help="methods whose code has changed since that run and whose"
                        " rows must therefore be computed again (any of: %s)"
                        % ", ".join(METHODS))
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
        for name, (label, _, _) in METHODS.items():
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
    # leaves behind everything it has already solved, and the scales it left
    # unsolved are run again by the next run from the start.
    started = datetime.now()
    stamp = started.strftime("%Y-%m-%d_%H-%M-%S")
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    # A run that resumes an earlier one writes its instances into the record of
    # that run: the record of a scale, interrupted and continued, is then one
    # document rather than one file per attempt.
    if args.resume:
        first = Path(args.resume[0])
        args.record_path = first if first.is_absolute() else HERE / first
    else:
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
