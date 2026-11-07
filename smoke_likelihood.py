"""Run the KiDS initialization and one likelihood evaluation, without sampling."""
import cProfile
import os
import pstats
import runpy
import sys
import time

import nautilus
import numpy as np


def smoke_sampler(prior, likelihood, **kwargs):
    params = {
        "omega_c": 0.153, "omega_b": 0.0225, "h": 0.6898,
        "n_s": 0.969, "S_8": 0.797, "T_AGN": 7.8, "A_IA": 0.0,
        **{f"ranz{i}": 0.0 for i in range(1, 7)},
        **{f"m{i}": v for i, v in enumerate(
            [-0.023, -0.016, -0.011, 0.020, 0.030, 0.045], 1)},
    }
    start = time.perf_counter()
    result = likelihood(params)
    print(f"Likelihood seconds: {time.perf_counter() - start:.3f}", flush=True)
    print(f"Likelihood result: {result}", flush=True)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Likelihood returned a non-finite value")
    raise SystemExit(0)


if __name__ == "__main__":
    if "BNT_DATA_DIR" not in os.environ:
        raise SystemExit("Set BNT_DATA_DIR to the directory containing KL_new/")
    os.environ.setdefault("BNT_OUTPUT_DIR", "/tmp/bnt_prd_smoke")
    nautilus.Sampler = smoke_sampler
    sys.argv = ["bnt_nautilus.sampler", "080002"]
    profiler = cProfile.Profile()
    try:
        profiler.runcall(runpy.run_module, "bnt_nautilus.sampler", run_name="__main__")
    finally:
        pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(30)
