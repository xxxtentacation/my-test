# -*- coding: utf-8 -*-
"""
Why the 2-approximation of paper Section 4.3 does as well as it does.

The rule of alg:approx2 is two things and nothing else: the jobs are ordered by
non-increasing weight w_j, and the jobs of equal weight are ordered by
non-increasing a_j + b_j.  This script asks which of the two carries the result,
and how much of it is the rule rather than the instances.

    part A  the rule against variations of itself, on the instances of the
            small scale, every one of them read on its gap against the optimum
            the model returned for it (the lower bound LB where a configuration
            has no recorded optimum):

                alg3      w desc, ties by a_j + b_j desc   (Algorithm 3)
                tie_asc   w desc, ties by a_j + b_j asc
                tie_draw  w desc, ties left in the order drawn
                tie_rand  w desc, ties shuffled
                tie_john  w desc, ties by Johnson within the class

            with three controls that are not the rule:

                w_asc     w asc, the rule read backwards
                johnson   Johnson's rule, which reads no weight at all
                random    a random permutation

    part B  how often the tie-break is exercised at all: the weight classes of
            an instance, their sizes, and how often a reordering inside them
            moves the value.

    part C  the tie-break under the four weight settings of the study, so that
            the same variation is read where ties are the rule rather than the
            exception: the narrow regime of the small scale (K = 30), the
            independent one (values 1, 2, 3), the unrestricted one (1, ..., n)
            and the correlated one (nine classes by a_j + b_j).

The instances are the ones of the study: drawn by the driver's rule of
run_small.py, from the seed of their configuration, so that the optimum the
records hold for an instance is the optimum of that instance.  Nothing here
runs the model or any heuristic beyond the sorts above.

Usage:
    python test_approx2.py                  every part, the sizes of the study
    python test_approx2.py --part a         one part only
    python test_approx2.py --instances 5 --sizes 10 30 60
"""
from __future__ import annotations

import argparse
import io
import random
import re
import statistics
import sys
from pathlib import Path

import NEH
import approx2
import run_small

HERE = Path(__file__).resolve().parent
RECORD_DIR = HERE / "记录"

#: the sizes and the ranges of the small scale of the study (run_small.py)
SIZES = run_small.SIZES
RANGES = run_small.RANGES
K = run_small.WEIGHT_COUNT
INSTANCES = 20
SEED = 42

#: how often a random order is drawn where a rule draws one, in part A
RANDOM_DRAWS = 20


# ---------------------------------------------------------------------------
# orders: the rule of alg:approx2 and the variations of it this script reads
# ---------------------------------------------------------------------------

def _by_weight(a, b, w, key, rng):
    """The jobs by non-increasing weight, ties by `key`, one job at a time."""
    return sorted(range(len(a)), key=lambda j: (-w[j], key(a, b, j)))


def alg3(a, b, w, rng):
    """Algorithm alg:approx2: w desc, ties by a_j + b_j desc."""
    return _by_weight(a, b, w, lambda a, b, j: -(a[j] + b[j]), rng)


def tie_asc(a, b, w, rng):
    """The same rule with the tie-break reversed."""
    return _by_weight(a, b, w, lambda a, b, j: (a[j] + b[j]), rng)


def tie_draw(a, b, w, rng):
    """The same rule with no tie-break: the order the jobs were drawn in."""
    return _by_weight(a, b, w, lambda a, b, j: 0, rng)


def _classes(a, b, w):
    """The weight classes, heaviest first, each a list of jobs."""
    out = {}
    for j in range(len(a)):
        out.setdefault(w[j], []).append(j)
    return [out[weight] for weight in sorted(out, reverse=True)]


def tie_rand(a, b, w, rng):
    """The same rule, with the classes shuffled inside."""
    order = []
    for group in _classes(a, b, w):
        group = list(group)
        rng.shuffle(group)
        order.extend(group)
    return order


def tie_john(a, b, w, rng):
    """The same rule, with Johnson's rule inside every class."""
    order = []
    for group in _classes(a, b, w):
        order.extend(johnson_of(a, b, group))
    return order


def johnson_of(a, b, jobs):
    """Johnson's rule on `jobs`, the two machines read on their own."""
    first = [j for j in jobs if a[j] <= b[j]]
    second = [j for j in jobs if a[j] > b[j]]
    first.sort(key=lambda j: a[j])
    second.sort(key=lambda j: -b[j])
    return first + second


def johnson(a, b, w, rng):
    """Johnson's rule on the whole instance: it reads no weight."""
    return johnson_of(a, b, range(len(a)))


def w_asc(a, b, w, rng):
    """The jobs by non-decreasing weight: the rule read backwards."""
    return sorted(range(len(a)), key=lambda j: (w[j], -(a[j] + b[j])))


def random_order(a, b, w, rng):
    """One random permutation."""
    order = list(range(len(a)))
    rng.shuffle(order)
    return order


#: the rules of part A, in the order the table reads them
RULES = (("alg3", alg3, "w desc, ties a+b desc  (Algorithm 3)"),
         ("tie_asc", tie_asc, "w desc, ties a+b asc"),
         ("tie_draw", tie_draw, "w desc, ties as drawn"),
         ("tie_rand", tie_rand, "w desc, ties shuffled"),
         ("tie_john", tie_john, "w desc, ties by Johnson"),
         ("w_asc", w_asc, "w asc  (backwards)"),
         ("johnson", johnson, "Johnson, no weight read"),
         ("random", random_order, "best of %d random permutations"
          % RANDOM_DRAWS))

#: the four weight settings of the study (NEH.gen_instance of the modules)
SETTINGS = (("narrow", "K = 30, the small scale"),
            ("indep", "values 1, 2, 3"),
            ("free", "values 1, ..., n"),
            ("corr", "nine classes by a_j + b_j"))


# ---------------------------------------------------------------------------
# instances and the optimum the records hold for them
# ---------------------------------------------------------------------------

#: the per-instance line of a record file, whose value the model closed the
#: instance at
_MILP_LINE = re.compile(r"\bMILP\s+n=(\d+)\s+K=(\d+)\s+\[(\d+),(\d+)\]\s+\| seed=\d+\s+"
                        r"\| instance\s+(\d+)/(\d+)\s+\| value\s+([0-9.]+)")


def recorded_optima():
    """
    The optima the model returned, by (n, processing-time range, number of the
    instance), the oldest record read first so that a later run replaces an
    earlier one on the same instance.
    """
    optima = {}
    for path in sorted(RECORD_DIR.glob("实例记录_small_*.txt")):
        for line in io.open(path, encoding="utf-8", errors="replace"):
            m = _MILP_LINE.search(line)
            if m:
                optima[(int(m.group(1)), int(m.group(4)), int(m.group(5)))] = \
                    float(m.group(7))
    return optima


def instances_of(n, proc_hi, count, weights="narrow"):
    """
    The instances of one configuration, exactly as run_small.py draws them: the
    stream its seed gives, both processing times normal on {1, ..., proc_hi},
    the weights by the setting named.
    """
    seed = run_small.seed_of(n, K, proc_hi, SEED)
    rng = random.Random(seed)
    return [NEH.gen_instance(n, K, rng, geometric=False, weights=weights,
                             proc_hi=proc_hi) for _ in range(count)]


def value_of(order, a, b, w):
    """The W C_max of a permutation schedule."""
    return approx2._wcmax_fast(order, a, b, w)


# ---------------------------------------------------------------------------
# part A: the rule against its own variations
# ---------------------------------------------------------------------------

def part_a(args, optima):
    """
    Every rule of RULES on the instances of the small scale, read on its gap
    against the optimum of the instance and on the instances it settles.

    Returns, by rule, the gap of every instance: (gaps, exact, instances,
    against_lb), where `exact` counts the instances whose recorded optimum the
    rule attained and `against_lb` counts those read on LB instead.
    """
    rows = {name: ([], 0, 0, 0) for name, _fn, _doc in RULES}
    per_size = {name: {} for name, _fn, _doc in RULES}
    values = {name: {} for name, _fn, _doc in RULES}    # by instance, for the
    rng = random.Random("test-%d" % SEED)               # paired comparison
    for proc_hi in args.ranges:
        for n in args.sizes:
            data = instances_of(n, proc_hi, args.instances)
            for idx, (a, b, w) in enumerate(data, start=1):
                opt = optima.get((n, proc_hi, idx))
                base = opt or approx2.instance_baseline(a, b, w)
                for name, fn, _doc in RULES:
                    draw = RANDOM_DRAWS if name == "random" else 1
                    best = min(value_of(fn(a, b, w, rng), a, b, w)
                               for _ in range(draw))
                    gap = 100.0 * (best - base) / base
                    gaps, exact, seen, lb = rows[name]
                    gaps.append(gap)
                    # The instances the rule settled: counted only where the
                    # reference is the optimum itself, since a value equal to LB
                    # says nothing about the rule having reached the optimum.
                    rows[name] = (gaps,
                                  exact + (1 if opt and abs(best - opt) < 1e-9
                                           else 0),
                                  seen + (1 if opt else 0),
                                  lb + (0 if opt else 1))
                    per_size[name].setdefault(n, []).append(gap)
                    values[name][(proc_hi, n, idx)] = gap
    return rows, per_size, values


def report_a(rows, per_size, sizes, instances, ranges, values):
    print("part A -- the rule of alg:approx2 against variations of itself")
    print("  the small scale, %d instances per configuration, the ranges %s;"
          % (instances, ", ".join("[1,%d]" % r for r in sorted(ranges))))
    print("  the gap is against the optimum of the instance where the records")
    print("  hold it and against LB where they do not, every rule read on the")
    print("  same reference for the same instance")
    print()
    print("%-9s %s %8s %8s" % ("rule", "".join("%7s" % ("n=%d" % n) for n in sizes),
                               "mean", "= opt"))
    print("-" * (9 + 1 + 7 * len(sizes) + 17))
    for name, _fn, doc in RULES:
        gaps, exact, seen, lb = rows[name]
        cells = "".join("%6.2f%%" % statistics.fmean(per_size[name][n])
                        for n in sizes)
        print("%-9s %s %7.2f%% %4d/%-3d" %
              (name, cells, statistics.fmean(gaps), exact, seen))
    print()
    print("  each rule against alg3, instance by instance, the same %d of them:"
          % len(values["alg3"]))
    print("  %-9s %10s %8s   %s" % ("rule", "mean diff", "stderr",
                                    "better / worse / equal"))
    print("  " + "-" * 60)
    base = values["alg3"]
    for name, _fn, _doc in RULES:
        if name == "alg3":
            continue
        d = [values[name][k] - base[k] for k in sorted(base)]
        better = sum(1 for x in d if x < -1e-9)
        worse = sum(1 for x in d if x > 1e-9)
        se = (statistics.stdev(d) / len(d) ** 0.5) if len(d) > 1 else 0.0
        print("  %-9s %+9.2f%% %7.2f%%   %4d /%5d /%5d"
              % (name, statistics.fmean(d), se, better, worse,
                 len(d) - better - worse))
    print("  a positive difference is worse than the rule of the paper")
    print()
    for name, _fn, doc in RULES:
        print("  %-9s %s" % (name, doc))
    print("  '= opt' counts the instances read against the optimum whose value")
    print("  the rule attained")
    lg = rows["alg3"][3]
    if lg:
        print("  (%d instances have no recorded optimum and are read on LB)"
              % lg)
    print()


# ---------------------------------------------------------------------------
# part B: how often the tie-break is exercised
# ---------------------------------------------------------------------------

def part_b(args, optima):
    """
    The weight classes of every instance of the small scale: how many there
    are, how large the largest is, and how often a reordering inside the
    classes moves the value of the schedule.
    """
    print("part B -- the tie-break inside the weight classes")
    print()
    print("%5s %7s %8s %8s %9s %12s %12s" %
          ("n", "range", "classes", "largest", "in a tie", "moved by a",
           "moved by a"))
    print("%5s %7s %8s %8s %9s %12s %12s" %
          ("", "", "mean", "class", "% of jobs", "shuffle", "Johnson"))
    print("-" * 69)
    rng = random.Random("ties-%d" % SEED)
    checked = 0
    for proc_hi in args.ranges:
        for n in args.sizes:
            data = instances_of(n, proc_hi, args.instances)
            nclass, big, tied, moved_r, moved_j = [], [], [], [], []
            for a, b, w in data:
                classes = _classes(a, b, w)
                nclass.append(len(classes))
                big.append(max(len(g) for g in classes))
                tied.append(sum(len(g) for g in classes if len(g) > 1))
                ref = value_of(tie_draw(a, b, w, rng), a, b, w)
                # the rule of the paper, checked on the instance it is run on
                want = sorted(range(len(a)), key=lambda j: (-w[j], -(a[j] + b[j])))
                assert alg3(a, b, w, rng) == want
                checked += 1
                moved_r.append(value_of(tie_rand(a, b, w, rng), a, b, w) != ref)
                moved_j.append(value_of(tie_john(a, b, w, rng), a, b, w) != ref)
            n_inst = len(data)
            print("%5d %7s %8.1f %8d %8.0f%% %11.0f%% %11.0f%%" %
                  (n, "[1,%d]" % proc_hi, statistics.fmean(nclass),
                   statistics.fmean(big),
                   100.0 * statistics.fmean(tied) / n,
                   100.0 * sum(moved_r) / n_inst,
                   100.0 * sum(moved_j) / n_inst))
    print()
    print("  the rule of alg:approx2 checked against the key (-w_j, -(a_j+b_j))")
    print("  on all %d instances above: identical on every one" % checked)
    print("  'moved by a shuffle' counts the instances whose value changes when")
    print("  the jobs of equal weight are reordered at random, and 'by Johnson'")
    print("  those whose value changes when Johnson's rule orders them inside")
    print("  the class; the reference is the order the jobs were drawn in")
    print()


# ---------------------------------------------------------------------------
# part C: the same variations where the ties are, and are not, the rule
# ---------------------------------------------------------------------------

def part_c(args, sizes):
    """The tie-break read under the four weight settings of the study."""
    print("part C -- the tie-break under the four weight settings")
    print("  n = %s, %d instances per cell, the gap against LB, which the"
          % (", ".join(str(n) for n in sizes), args.instances))
    print("  instance determines, so that every cell of a row is read on the")
    print("  same denominator; the differences between the cells of a row are")
    print("  what this part reads")
    print()
    rules = [("alg3", alg3), ("tie_asc", tie_asc), ("tie_rand", tie_rand),
             ("tie_john", tie_john)]
    rng = random.Random("settings-%d" % SEED)
    for setting, what in SETTINGS:
        print("  %s -- %s" % (setting, what))
        print("    %5s %9s %9s %s" % ("n", "largest", "in a tie",
                                      "".join("%11s" % name
                                              for name, _fn in rules)))
        print("    " + "-" * (5 + 20 + 11 * len(rules)))
        for n in sizes:
            data = instances_of(n, RANGES[0], args.instances, weights=setting)
            gaps = {name: [] for name, _fn in rules}
            big, tied = [], []
            for a, b, w in data:
                classes = _classes(a, b, w)
                big.append(max(len(g) for g in classes))
                tied.append(sum(len(g) for g in classes if len(g) > 1))
                lb = approx2.instance_baseline(a, b, w)
                for name, fn in rules:
                    v = value_of(fn(a, b, w, rng), a, b, w)
                    gaps[name].append(100.0 * (v - lb) / lb)
            print("    %5d %9.1f %8.0f%% %s"
                  % (n, statistics.fmean(big),
                     100.0 * statistics.fmean(tied) / n,
                     "".join("%10.2f%%" % statistics.fmean(gaps[name])
                             for name, _fn in rules)))
        print()


# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Why the 2-approximation of paper Section 4.3 does as well "
                    "as it does: the weight order, the tie-break, and the "
                    "instances they are read on.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--part", nargs="+", default=["a", "b", "c"],
                   choices=["a", "b", "c"], help="parts to run")
    p.add_argument("--sizes", nargs="+", type=int, default=list(SIZES),
                   help="instance sizes")
    p.add_argument("--ranges", nargs="+", type=int, default=list(RANGES),
                   help="upper end of the processing-time range")
    p.add_argument("--instances", type=int, default=INSTANCES,
                   help="instances per configuration")
    p.add_argument("--settings-sizes", nargs="+", type=int, default=None,
                   help="sizes of part C (default: the first, middle and last "
                        "of --sizes)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sizes = sorted(args.sizes)
    print("F2 || WCmax -- why the 2-approximation does as well as it does")
    print("  sizes     : %s" % ", ".join(str(n) for n in sizes))
    print("  ranges    : %s" % ", ".join("[1,%d]" % r for r in args.ranges))
    print("  instances : %d per configuration, seed %d" % (args.instances, SEED))
    print()
    if "a" in args.part or "b" in args.part:
        optima = recorded_optima()
        print("  recorded optima: %d instances of the small scale" % len(optima))
        print()
    if "a" in args.part:
        rows, per_size, values = part_a(args, optima)
        report_a(rows, per_size, sizes, args.instances, args.ranges, values)
    if "b" in args.part:
        part_b(args, optima)
    if "c" in args.part:
        csizes = args.settings_sizes or [sizes[0], sizes[len(sizes) // 2],
                                         sizes[-1]]
        part_c(args, sorted(set(csizes)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
