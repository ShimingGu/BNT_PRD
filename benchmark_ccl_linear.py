"""Compare CCL/CAMB amplitude calibration with and without nonlinear HMcode."""
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import camb
import pyccl as ccl


def run(omega_c, omega_b, h, ns, nonlinear):
    start = time.perf_counter()
    cosmo = ccl.Cosmology(
        Omega_c=omega_c / h**2, Omega_b=omega_b / h**2,
        h=h, n_s=ns, A_s=2.1e-9, Neff=3.046, m_nu=0.06, w0=-1.0, wa=0.0,
        transfer_function="boltzmann_camb", matter_power_spectrum="camb" if nonlinear else "linear",
        extra_parameters={"camb": {"halofit_version": "mead2020_feedback", "HMCode_logT_AGN": 7.8}},
    )
    if nonlinear:
        value = float(ccl.sigma8(cosmo))
    else:
        # CCL 3.2.1's linear-only CAMB wrapper asserts NonLinear_none, but
        # CAMB 2.0.5's set_for_lmax enables lensing nonlinear mode. Limit the
        # workaround to this benchmark and restore the original method after.
        original = camb.CAMBparams.set_for_lmax

        def linear_set_for_lmax(params, *args, **kwargs):
            result = original(params, *args, **kwargs)
            params.NonLinear = camb.model.NonLinear_none
            return result

        with patch.object(camb.CAMBparams, "set_for_lmax", linear_set_for_lmax):
            value = float(ccl.sigma8(cosmo))
    return value, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("ccl_linear_benchmark.json"))
    args = parser.parse_args()
    points = [(0.153, 0.0225, 0.6898, 0.969), (0.078, 0.0236, 0.6528, 0.975),
              (0.328, 0.0215, 0.6858, 1.019)]
    run(*points[0], False)  # CAMB initialization is excluded from measurements.
    rows = []
    for point in points:
        reference, baseline_seconds = run(*point, True)
        linear, linear_seconds = run(*point, False)
        row = dict(parameters=point, sigma8_baseline=reference, sigma8_linear=linear,
                   relative_difference=linear / reference - 1,
                   baseline_seconds=baseline_seconds, linear_seconds=linear_seconds)
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
