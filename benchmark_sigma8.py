"""Compare a matter-only CAMB sigma8 estimate to the PyCCL/CAMB baseline."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyccl as ccl
from scipy.stats import qmc
from bnt_nautilus.calibration import matter_only_sigma8


def compare(omega_c, omega_b, h, n_s):
    common = dict(Omega_c=omega_c / h**2, Omega_b=omega_b / h**2,
                  h=h, n_s=n_s, A_s=2.1e-9, Neff=3.046, m_nu=0.06,
                  w0=-1.0, wa=0.0)
    start = time.perf_counter()
    cosmo = ccl.Cosmology(**common, transfer_function="boltzmann_camb",
                          matter_power_spectrum="camb",
                          extra_parameters={"camb": {"halofit_version": "mead2020_feedback"}})
    sigma_ref = ccl.sigma8(cosmo)
    reference_s = time.perf_counter() - start

    start = time.perf_counter()
    sigma_fast = matter_only_sigma8(omega_c, omega_b, h, n_s, common["A_s"])
    fast_s = time.perf_counter() - start
    return dict(omega_c=omega_c, omega_b=omega_b, h=h, n_s=n_s,
                sigma_ref=float(sigma_ref), sigma_fast=sigma_fast,
                relative_s8_error=float(sigma_ref / sigma_fast - 1),
                reference_seconds=reference_s, fast_seconds=fast_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--output", type=Path, default=Path("sigma8_benchmark.json"))
    args = parser.parse_args()
    bounds_low = np.array([0.051, 0.019, 0.63, 0.83])
    bounds_high = np.array([0.355, 0.026, 0.83, 1.11])
    draws = qmc.scale(qmc.LatinHypercube(d=4, seed=args.seed).random(args.samples),
                      bounds_low, bounds_high)
    fiducial = np.array([[0.153, 0.0225, 0.6898, 0.969]])
    points = np.concatenate([fiducial, draws])
    rows = []
    for point in points:
        row = compare(*map(float, point))
        rows.append(row)
        print(json.dumps(row), flush=True)
    errors = np.abs([x["relative_s8_error"] for x in rows])
    summary = dict(seed=args.seed, prior_samples=args.samples,
                   max_abs_relative_error=float(np.max(errors)),
                   median_abs_relative_error=float(np.median(errors)),
                   n_over_0p005_relative=int(np.count_nonzero(errors > 0.005)),
                   median_reference_seconds=float(np.median([x["reference_seconds"] for x in rows])),
                   median_fast_seconds=float(np.median([x["fast_seconds"] for x in rows])))
    args.output.write_text(json.dumps(dict(summary=summary, points=rows), indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
