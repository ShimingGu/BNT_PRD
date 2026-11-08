"""Compare GPU linear + CCL halofit against CPU CAMB mead2020_feedback.

This is the nonlinear-path prototype.  CCL's apply_halofit is *not* CAMB's
HMcode-2020 with baryon feedback, so the measurement is the point: if the pC_l
and chi2 differences are small, the route is usable; otherwise it is recorded
as a known mismatch.
"""

import os
import runpy
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("BNT_DATA_DIR", "/arc/home/gsm/KiDS_BNT_Nautilus")
os.environ.setdefault("BNT_OUTPUT_DIR", "/tmp/bnt_gpu_nonlinear")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/scratch/pyccl-tensor/camb-tensor")

import nautilus  # noqa: E402

SOURCE = Path("/scratch/pyccl-tensor/camb-tensor/tensor_prototypes/"
              "camb-segment-export-_d74o2gh/source")
PARAMS = {
    "omega_c": 0.153, "omega_b": 0.0225, "h": 0.6898,
    "n_s": 0.969, "S_8": 0.797, "T_AGN": 7.8, "A_IA": 0.0,
    **{f"ranz{i}": 0.0 for i in range(1, 7)},
    **{f"m{i}": value for i, value in enumerate(
        [-0.023, -0.016, -0.011, 0.020, 0.030, 0.045], 1)},
}
CAPTURED = {}


def _capture(prior, likelihood, **kwargs):
    CAPTURED["globals"] = likelihood.__globals__
    raise SystemExit(0)


def main():
    nautilus.Sampler = _capture
    sys.argv = ["bnt_nautilus.sampler", "080002"]
    try:
        runpy.run_module("bnt_nautilus.sampler", run_name="__main__")
    except SystemExit:
        pass
    g = CAPTURED["globals"]

    cdz = g["mu_dz"]
    common = dict(PARAMS)
    common.update(
        ell_arr=g["cell_arr"],
        sigma8_normalization=g["sigma8_normalization"],
        **{f"dz{i + 1}": cdz[i] for i in range(6)},
        pCl_dir=g["KL_dir"], Omega_h=True,
        NL_code=g["NL_code"], NL_recipe=g["NL_recipe"],
        Modified_Gravity=g["MG_switch"],
        dzmode="additive", m_bias=True, kmin=0.0025, kmax=100,
    )

    cpu = g["bnt_core"](g["nz_dt"], A_NL=1, **common)

    from camb_gpu.api import bnt_pka_halofit

    cosmology = {key: PARAMS[key] for key in ("omega_c", "omega_b", "h", "n_s")}
    pka_nl = bnt_pka_halofit(SOURCE, cosmology, PARAMS["S_8"],
                             cpu.ccl_cosmo, cpu.aarr2)
    gpu = g["bnt_core"](g["nz_dt"], A_NL=1, external_pka=pka_nl, **common)
    cpu.pCl_vec()
    gpu.pCl_vec()

    ratio = gpu.pka / np.maximum(cpu.pka, 1e-300)
    print("pka ratio min/median/max",
          float(np.nanmin(ratio)), float(np.nanmedian(ratio)),
          float(np.nanmax(ratio)))

    cpu_pcl = np.asarray(cpu.pCl_deorg, dtype=np.float64)
    gpu_pcl = np.asarray(gpu.pCl_deorg, dtype=np.float64)
    mask = np.abs(cpu_pcl) > 1e-2 * np.max(np.abs(cpu_pcl))
    relative = float(np.max(np.abs(cpu_pcl[mask] - gpu_pcl[mask])
                            / np.abs(cpu_pcl[mask])))
    print(f"pC_l max relative difference (above 1% of max): {relative:.3e}")

    standard = g["pCl_std"] if g["theoretical_fid"] else g["pCl_kls"]

    def chi2(bnt):
        delta = bnt.Cl_vector_to_bnt(bnt.pCl_deorg - standard)[g["aCl_diff"]]
        return float(delta @ g["aCl_icov_sc"] @ delta)

    chi2_cpu, chi2_gpu = chi2(cpu), chi2(gpu)
    print(f"chi2 CPU (CAMB mead2020_feedback) {chi2_cpu:.6f}")
    print(f"chi2 GPU (linear + CCL halofit)   {chi2_gpu:.6f}")
    print(f"delta chi2                        {chi2_gpu - chi2_cpu:.6f}")
    assert np.all(np.isfinite([chi2_cpu, chi2_gpu]))
    print("OK (measurement, not an equivalence claim)")


if __name__ == "__main__":
    main()
