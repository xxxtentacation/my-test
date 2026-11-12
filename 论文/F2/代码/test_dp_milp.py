# -*- coding: utf-8 -*-
"""
Cross-check of the two exact methods on the same small instances.

For every instance the dynamic program of Section 4 and the MILP model of
Section 3 are run on the very same data, and their objective values are
compared.  The dynamic program is taken as the reference: it is exact and its
state space is small at these sizes, while the model may stop on its time limit
without closing the instance.  A model value below the DP value therefore means
that one of the two is wrong; a model value above it means that the model's
search was cut short.

The instances are drawn from the generator of the study (``NEH.gen_instance``),
with the geometric ladder of weights used by the dynamic-program group, so that
the comparison covers exactly the instances of paper Section 6.

Usage
-----
    python test_dp_milp.py                        # the default grid
    python test_dp_milp.py --sizes 5 6 7 8        # other sizes
    python test_dp_milp.py --weights 2 3 --instances 5
    python test_dp_milp.py --time-limit 300       # longer model limit

The exit status is 0 when every instance agrees and 1 otherwise, so the script
can be used as a check before a long run.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import NEH      # noqa: E402
import dp       # noqa: E402
import milp     # noqa: E402

#: tolerance on the comparison of two objective values, which are numbers of
#: the same order as the data and are computed by two different methods
TOL = 1e-6


def instance_of(n, K, idx, seed, proc_hi):
    """The idx-th instance of the (n, K) configuration, from the study's stream."""
    rng = random.Random(seed + 1000 * n + 100 * K + idx)
    return NEH.gen_instance(n, K, rng, geometric=True, weights="narrow",
                            proc_hi=proc_hi)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the DP and the MILP model on the same small instances "
                    "and compare their objective values.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 6, 8, 10],
                        help="instance sizes to test (default: 5 6 8 10)")
    parser.add_argument("--weights", type=int, nargs="+", default=[2, 3],
                        help="numbers of weight values of the geometric ladder "
                             "(default: 2 3)")
    parser.add_argument("--instances", type=int, default=3,
                        help="instances per configuration (default: 3)")
    parser.add_argument("--proc-hi", type=int, default=40,
                        help="upper end of the processing-time range (default: 40)")
    parser.add_argument("--time-limit", type=float, default=120.0,
                        help="time limit of the model, in seconds (default: 120)")
    parser.add_argument("--seed", type=int, default=42,
                        help="base random seed (default: 42)")
    args = parser.parse_args(argv)

    print("DP vs MILP on the same instances")
    print("  sizes      : %s" % " ".join(str(n) for n in args.sizes))
    print("  weights    : %s" % " ".join("{%s}" % ",".join(
        str(10 ** i) for i in range(K)) for K in args.weights))
    print("  instances  : %d per configuration" % args.instances)
    print("  proc range : [1,%d]" % args.proc_hi)
    print("  model limit: %g s" % args.time_limit)
    print()

    total = agreed = below = above = unproven = 0
    for n in args.sizes:
        for K in args.weights:
            for i in range(args.instances):
                a, b, w = instance_of(n, K, i, args.seed, args.proc_hi)
                d, d_time, d_proved, _ = dp.solve_instance(a, b, w)
                m, m_time, m_proved, _ = milp.solve_instance(
                    a, b, w, time_limit=args.time_limit)
                total += 1
                if not m_proved:
                    unproven += 1

                if d is None:
                    verdict = "DP returned no value"
                elif m is None:
                    verdict = "model returned no value"
                elif abs(d - m) <= TOL:
                    agreed += 1
                    verdict = "ok"
                elif m < d:
                    below += 1
                    verdict = "WRONG: model below the DP"
                else:
                    above += 1
                    verdict = "model above the DP (not closed)"

                print("  n=%-3d K=%d  inst %d  DP %-12s %8.2f ms   "
                      "MILP %-12s %9.2f ms   %s%s"
                      % (n, K, i + 1,
                         "none" if d is None else "%.1f" % d, 1000.0 * d_time,
                         "none" if m is None else "%.1f" % m, 1000.0 * m_time,
                         verdict, "" if m_proved else "  [no proof]"),
                      flush=True)

    print()
    print("instances : %d" % total)
    print("agree     : %d" % agreed)
    print("model below the DP (wrong): %d" % below)
    print("model above the DP (not closed): %d" % above)
    print("model without a proof of optimality: %d" % unproven)
    if agreed == total:
        print("result    : all instances agree")
        return 0
    print("result    : MISMATCH")
    return 1


if __name__ == "__main__":
    sys.exit(main())
