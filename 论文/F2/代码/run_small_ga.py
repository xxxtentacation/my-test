# -*- coding: utf-8 -*-
"""
The genetic algorithm alone, on the small scale, read against the optimum.

The configurations are the small scale of the study at the sizes it is read on
every tenth job -- 10, 20, ..., 60 -- with the weights drawn from {1, ..., 30}
and both processing times on [1,40], and every instance is replicated twenty
times.  The method run is the genetic algorithm, and the reference value of
every instance is the optimum the model of the study returned on it: the model
itself is not run here, its optima are read from the records of the runs of the
study (记录/实例记录_small_*.txt).  The instances of a configuration are drawn
from a stream its own settings seed, so the optimum recorded for an instance is
the optimum of the instance this script draws; where a configuration has no
recorded optimum yet, its gap falls back on the lower bound LB of paper Section
4.3, and the run says which sizes those were.

The instances are drawn with the generator of this script rather than with the
one of NEH.py: every integer of an instance -- both processing times of a job
and its weight -- comes from a normal distribution on the integers of its range,
with the mean at the middle of the range and the standard deviation a sixth of
its width, the draw rounded to an integer and clamped to the range.

Usage:
    python run_small_ga.py                    the six sizes, 20 instances each
    python run_small_ga.py --instances 5      a quicker pass over the six sizes
    python run_small_ga.py --ga-gen 20 --ga-pm 1
    python run_small_ga.py --list             the configurations, then exit
"""
from __future__ import annotations

import argparse
import io
import random
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import GA
import NEH
import run_small

HERE = Path(__file__).resolve().parent
RECORD_DIR = HERE / "记录"

#: the sizes of this test, and the settings every one of them is read at
SIZES = (10, 20, 30, 40, 50, 60)
WEIGHTS, PROC_HI, K = "narrow", 40, 30
INSTANCES = 20


# ---------------------------------------------------------------------------
# the instances: the generator of the study, every integer of it drawn normally
# ---------------------------------------------------------------------------

def _normal_int(rng, lo, hi):
    """One integer drawn from a normal distribution on lo, ..., hi."""
    if hi <= lo:
        return lo
    return max(lo, min(hi, int(round(rng.gauss(0.5 * (lo + hi), (hi - lo) / 6.0)))))


class _NormalStream:
    """The random stream of an instance, with every integer drawn normally."""

    def __init__(self, rng):
        self._rng = rng

    def randint(self, lo, hi):
        return _normal_int(self._rng, lo, hi)

    def randrange(self, k):
        return _normal_int(self._rng, 0, k - 1)

    def __getattr__(self, name):
        return getattr(self._rng, name)


_NEH_GEN_INSTANCE = NEH.gen_instance


def gen_instance(n, K, rng, **kwargs):
    """One instance of this test, every integer of it drawn normally."""
    return _NEH_GEN_INSTANCE(n, K, _NormalStream(rng), **kwargs)


NEH.gen_instance = gen_instance
GA.gen_instance = gen_instance


# ---------------------------------------------------------------------------
# the reference: the optimum the model returned, read from the records
# ---------------------------------------------------------------------------

#: the per-instance line of a record file, whose value is the optimum the model
#: closed the instance at
_MILP_LINE = re.compile(r"\bMILP\s+n=(\d+)\s+K=(\d+)\s+\[(\d+),(\d+)\]\s+\| seed=\d+\s+"
                        r"\| instance\s+(\d+)/(\d+)\s+\| value\s+([0-9.]+)")


def recorded_optima():
    """
    The optima the model has already returned, by (n, processing-time range,
    number of the instance).

    Every record file of the small scale is read, the oldest first, so that a
    later run of the model replaces an earlier one on the same instance.
    """
    optima = {}
    for path in sorted(RECORD_DIR.glob("实例记录_small_*.txt")):
        for line in io.open(path, encoding="utf-8", errors="replace"):
            m = _MILP_LINE.search(line)
            if m:
                optima[(int(m.group(1)), int(m.group(4)), int(m.group(5)))] = \
                    float(m.group(7))
    return optima


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="The genetic algorithm alone on the small scale of the "
                    "study, its gap read against the optimum of the model, "
                    "which is taken from the records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--instances", type=int, default=INSTANCES,
                   help="instances per configuration")
    p.add_argument("--seed", type=int, default=42, help="base random seed")
    p.add_argument("--ga-pop", type=int, default=None, help="GA: population size")
    p.add_argument("--ga-gen", type=int, default=None, help="GA: generations")
    p.add_argument("--ga-pc", type=float, default=None,
                   help="GA: crossover probability")
    p.add_argument("--ga-pm", type=float, default=None,
                   help="GA: mutation probability")
    p.add_argument("--list", action="store_true",
                   help="list the configurations, then exit")
    return p.parse_args(argv)


def solve_config(n, instances, args, optima, record_path):
    """The GA on every instance of one configuration, with its gaps."""
    seed = run_small.seed_of(n, K, PROC_HI, args.seed)
    rng = random.Random(seed)
    solver_rng = random.Random("solver-%d" % seed)
    rows = []
    for idx in range(1, instances + 1):
        a, b, w = GA.gen_instance(n, K, rng, weights=WEIGHTS, proc_hi=PROC_HI)
        kwargs = {"N": run_small.population_of(n),
                  "G": run_small.generations_of(n),
                  "lam": run_small.offspring_of(n),
                  "pc": run_small.crossover_probability_of(n),
                  "pm": run_small.mutation_probability_of(n),
                  "k": run_small.descent_width_of(n)}
        for flag, key in (("ga_pop", "N"), ("ga_gen", "G"),
                          ("ga_pc", "pc"), ("ga_pm", "pm")):
            if getattr(args, flag) is not None:
                kwargs[key] = getattr(args, flag)
        t0 = time.perf_counter()
        obj, _seconds, _proved, base = GA.solve_instance(
            a, b, w, seed=solver_rng.randrange(1 << 30), **kwargs)
        seconds = time.perf_counter() - t0
        # The reference of the instance is the optimum of the model where the
        # records hold it, and the lower bound LB of the instance otherwise.
        optima_of_instance = optima.get((n, PROC_HI, idx))
        reference = optima_of_instance if optima_of_instance else base
        rows.append((obj, reference, optima_of_instance is not None, seconds))
        GA.record_instance(record_path, n, WEIGHTS, seed, idx, instances, obj,
                           seconds, K, proc_hi=PROC_HI, base=reference,
                           label="GA")
    return rows


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        print("the small scale, the sizes %s, weights %s, range [1,%d]"
              % (", ".join(str(n) for n in SIZES), WEIGHTS, PROC_HI))
        print("reference: the recorded optimum of the model, LB where a "
              "configuration has none")
        return 0

    optima = recorded_optima()
    started = datetime.now()
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    record_path = RECORD_DIR / ("实例记录_small_ga_%s.txt"
                                % started.strftime("%Y-%m-%d_%H-%M-%S"))

    print("F2 || WCmax -- the genetic algorithm on the small scale")
    print("  sizes     : %s" % ", ".join(str(n) for n in SIZES))
    print("  instances : %d per configuration" % args.instances)
    print("  weights   : %s, both processing times on [1,%d], drawn normally"
          % (WEIGHTS, PROC_HI))
    print("  reference : the optimum of the model, read from the records "
          "where they hold it, LB otherwise")
    print("  records   : %s" % record_path)
    print()
    print("%4s %6s %8s %10s %10s %10s %9s %6s" %
          ("n", "in.", "ref.", "gap", "min", "max", "time", "MILP"))
    print("-" * 72)

    total = []
    for n in SIZES:
        rows = solve_config(n, args.instances, args, optima, record_path)
        gaps = [100.0 * (o - r) / r for o, r, _opt, _s in rows]
        times = [s for _o, _r, _opt, s in rows]
        with_opt = sum(1 for _o, _r, opt, _s in rows if opt)
        total.extend((n, g) for g in gaps)
        print("%4d %6d %8s %9.2f%% %9.2f%% %9.2f%% %8.2f s %6s" %
              (n, len(rows), "optimum" if with_opt == len(rows) else
               ("LB" if not with_opt else "mixed"),
               statistics.fmean(gaps), min(gaps), max(gaps),
               statistics.fmean(times), "%d/%d" % (with_opt, len(rows))))
        sys.stdout.flush()

    print()
    means = [statistics.fmean([g for nn, g in total if nn == n]) for n in SIZES]
    print("mean gap over the six sizes: %s"
          % ", ".join("%d: %.2f%%" % (n, g) for n, g in zip(SIZES, means)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
