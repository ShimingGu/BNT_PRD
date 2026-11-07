"""Multi-CPU likelihood throughput benchmark.

Runs ``M`` likelihood evaluations across ``P`` processes (spawned fresh, each
paying the module setup once) and reports wall time and per-point throughput.

    PYTHONPATH=<pkg>/src BNT_DATA_DIR=... \\
        python benchmarks/bench_parallel.py --points 8 --workers 4

Use it against two checkouts (e.g. a git worktree of a previous commit) with
the same data and environment to compare before/after.
"""

import argparse
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BASE = {
    "omega_c": 0.153, "omega_b": 0.0225, "h": 0.6898,
    "n_s": 0.969, "S_8": 0.797, "T_AGN": 7.8, "A_IA": 0.0,
    **{f"ranz{i}": 0.0 for i in range(1, 7)},
    **{f"m{i}": value for i, value in enumerate(
        [-0.023, -0.016, -0.011, 0.020, 0.030, 0.045], 1)},
}
_STATE = {}


def _init(mode, data_dir, output_dir):
    os.environ["BNT_DATA_DIR"] = data_dir
    os.environ["BNT_OUTPUT_DIR"] = output_dir
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.argv = ["bnt_nautilus.sampler", mode]
    # The sampler module requires __name__ == "__main__" (it reads MODE and
    # would otherwise raise during import).  Run it as __main__ and intercept
    # the Sampler constructor to capture the configured likelihood.
    import runpy

    import nautilus

    def _capture(prior, likelihood, **kwargs):
        _STATE["loglike"] = likelihood
        raise SystemExit(0)

    nautilus.Sampler = _capture
    try:
        runpy.run_module("bnt_nautilus.sampler", run_name="__main__")
    except SystemExit:
        pass
    if "loglike" not in _STATE:
        raise RuntimeError("failed to capture loglike_cosmo")


def _evaluate(params):
    started = time.perf_counter()
    result = _STATE["loglike"](params)
    return time.perf_counter() - started, float(result[0])


def make_points(count):
    points = []
    for index in range(count):
        point = dict(BASE)
        point["omega_c"] = BASE["omega_c"] + 0.004 * (index - count // 2)
        point["h"] = BASE["h"] + 0.002 * ((index % 3) - 1)
        point["n_s"] = BASE["n_s"] + 0.002 * ((index % 4) - 1.5)
        point["S_8"] = BASE["S_8"] + 0.006 * ((index % 5) - 2)
        point["T_AGN"] = BASE["T_AGN"] + 0.2 * (index % 3)
        point["A_IA"] = 0.1 * (index % 2)
        points.append(point)
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mode", default="080002")
    parser.add_argument("--data-dir", default=os.environ.get("BNT_DATA_DIR"))
    parser.add_argument("--output-dir", default=os.environ.get("BNT_OUTPUT_DIR", "/tmp/bnt_bench"))
    args = parser.parse_args()
    if not args.data_dir:
        raise SystemExit("Set BNT_DATA_DIR or pass --data-dir")

    points = make_points(args.points)
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_init,
        initargs=(args.mode, args.data_dir, args.output_dir),
    ) as pool:
        results = list(pool.map(_evaluate, points, chunksize=1))
    wall = time.perf_counter() - started

    times = [entry[0] for entry in results]
    print(f"package: {Path(__import__('bnt_nautilus').__file__).resolve().parent}")
    print(f"points {len(points)}  workers {args.workers}")
    print(f"wall {wall:.2f} s  per-point {wall / len(points):.3f} s  "
          f"throughput {len(points) / wall:.3f} /s")
    print(f"sum of evaluations {sum(times):.2f} s  "
          f"parallel efficiency {sum(times) / (wall * args.workers):.2f}")
    print(f"first loglike {results[0][1]:.6f}")


if __name__ == "__main__":
    main()
