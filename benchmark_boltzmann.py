"""Time CAMB transfer and power-spectrum stages with representative grids."""

import json
import statistics
import time
from pathlib import Path

import camb
import numpy as np


def params_for(redshifts, kmax):
    params = camb.CAMBparams()
    params.set_cosmology(H0=68.98, ombh2=0.0225, omch2=0.153, mnu=0.06, nnu=3.046)
    params.InitPower.set_params(As=2.1e-9, ns=0.969)
    params.WantCls = False
    params.DoLensing = False
    params.set_matter_power(redshifts=redshifts, kmax=kmax, nonlinear=False, silent=True)
    params.NonLinear = camb.model.NonLinear_none
    return params


def measure(redshifts, kmax, repeats=3):
    camb.get_transfer_functions(params_for(redshifts, kmax))
    transfer_times, power_times, grid_sizes = [], [], []
    for _ in range(repeats):
        params = params_for(redshifts, kmax)
        start = time.perf_counter()
        result = camb.get_transfer_functions(params)
        transfer_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        result.calc_power_spectra()
        power_times.append(time.perf_counter() - start)
        grid_sizes.append(result.get_matter_transfer_data().nq)
    return dict(redshifts=len(redshifts), kmax=kmax, grid_sizes=grid_sizes,
                median_transfer_seconds=statistics.median(transfer_times),
                median_power_seconds=statistics.median(power_times),
                transfer_seconds=transfer_times, power_seconds=power_times)


def main():
    configs = [([0.0], 2.0), (np.linspace(0, 3, 64).tolist(), 10.0)]
    results = []
    for redshifts, kmax in configs:
        row = measure(redshifts, kmax)
        results.append(row)
        print(json.dumps(row), flush=True)
    Path("BNT_PRD/boltzmann_benchmark.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
