# Runtime data

The full KiDS/BNT input data are not distributed with this repository. If you
need to use the full data, please ask **Shiming Gu** or **Ziang Yan** at the
email addresses indicated in the paper.

After receiving the data, set `BNT_DATA_DIR` to the directory containing `KL_new/`.
For fiducial mode `05`, the legacy `Cov_pCl_Additive.npy` must be placed directly
inside `BNT_DATA_DIR`; other fiducial modes load the covariance from their
selected `KL_*` directory.
