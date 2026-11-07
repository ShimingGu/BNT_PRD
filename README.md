# BNT Nautilus KiDS likelihood

This package runs the BNT-transformed pseudo-`C_ell` KiDS likelihood using
Nautilus and MPI. It has been extracted from the original HPC working
directory, so paths are relative to the checkout by default.

The accompanying paper is available as [arXiv:2607.04384](https://arxiv.org/abs/2607.04384).

The code responsible for [arXiv:2511.09544](https://arxiv.org/abs/2511.09544)
and [arXiv:2412.14704](https://arxiv.org/abs/2412.14704) is on the
[`euclid-mock`](https://github.com/ShimingGu/BNT_PRD/tree/euclid-mock) branch.

## Setup

Create an environment that provides a C/C++ toolchain and MPI implementation,
then install the project:

```bash
python -m pip install -e .
```

`pyccl` and `mpi4py` may need to be built against the local scientific/MPI
stack.

An optional matter-only CAMB estimate can speed up the single S8 amplitude
correction. Enable it with `BNT_SIGMA8_NORMALIZATION=camb_matter`; the default
`ccl` method is unchanged. The approximation is currently restricted to
`w0=-1`, `wa=0`, and `m_nu=0.06 eV`, and its accuracy has only been checked
at a small set of prior points. See `GPU_ACCELERATION_DRAFT.md` and
`benchmark_sigma8.py` for details. Fast-path output and checkpoints have a
`fastsigma8_` prefix to keep them separate from the default method.

## Data access

The full KiDS/BNT input data are not distributed with this repository. If you
need to use the full data, please ask **Shiming Gu** or **Ziang Yan** at the
email addresses indicated in the paper. See [data/README.md](data/README.md)
for the expected data location.

## Run

```bash
./run_bnt.sh 080002
```

Set `BNT_MPI_WORKERS` to change the process count. Set `BNT_DATA_DIR` to the
directory containing the supplied data, and `BNT_OUTPUT_DIR` to change where
checkpoints and chains are written (default: `outputs/`).
