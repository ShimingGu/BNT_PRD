"""Warm per-point likelihood timing (and optional cProfile) for BNT.

    BNT_DATA_DIR=/arc/home/gsm/KiDS_BNT_Nautilus \
    BNT_OUTPUT_DIR=/tmp/bnt_profile OMP_NUM_THREADS=1 \
        .pixi/envs/default/bin/python -u benchmarks/profile_likelihood.py [--profile]

Runs module setup, one warm-up evaluation, then four *different* parameter
points (the MCMC proxy).  `--profile` additionally profiles one warm point and
prints self-time and cumulative rankings.
"""

import argparse
import cProfile
import os
import pstats
import runpy
import sys
import time

import nautilus

BASE = {
    "omega_c": 0.153, "omega_b": 0.0225, "h": 0.6898,
    "n_s": 0.969, "S_8": 0.797, "T_AGN": 7.8, "A_IA": 0.0,
    **{f"ranz{i}": 0.0 for i in range(1, 7)},
    **{f"m{i}": v for i, v in enumerate(
        [-0.023, -0.016, -0.011, 0.020, 0.030, 0.045], 1)},
}
POINTS = [
    dict(BASE),
    dict(BASE, omega_c=0.145, h=0.695, n_s=0.972, S_8=0.805),
    dict(BASE, omega_c=0.162, h=0.684, n_s=0.965, S_8=0.790),
    dict(BASE, omega_c=0.138, h=0.701, n_s=0.975, S_8=0.812),
]
PROFILE = False


def _sampler(prior, likelihood, **kwargs):
    likelihood(POINTS[0])  # one-time warm-up: imports, Legendre roots, CCL init
    for index, point in enumerate(POINTS):
        started = time.perf_counter()
        likelihood(point)
        print(f"point {index}: {time.perf_counter() - started:.3f} s", flush=True)
    if PROFILE:
        profiler = cProfile.Profile()
        profiler.runcall(likelihood, POINTS[1])
        stats = pstats.Stats(profiler).strip_dirs()
        print("=== self time ===")
        stats.sort_stats("tottime").print_stats(25)
        print("=== cumulative ===")
        stats.sort_stats("cumulative").print_stats(20)
    raise SystemExit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--mode", default="080002")
    args = parser.parse_args()
    PROFILE = args.profile
    if "BNT_DATA_DIR" not in os.environ:
        raise SystemExit("Set BNT_DATA_DIR to the directory containing KL_new/")
    os.environ.setdefault("BNT_OUTPUT_DIR", "/tmp/bnt_profile")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    nautilus.Sampler = _sampler
    sys.argv = ["bnt_nautilus.sampler", args.mode]
    runpy.run_module("bnt_nautilus.sampler", run_name="__main__")
