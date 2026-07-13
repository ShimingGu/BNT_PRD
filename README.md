# BNT Nautilus KiDS likelihood

This package runs the BNT-transformed pseudo-`C_ell` KiDS likelihood using
Nautilus and MPI. It has been extracted from the original HPC working
directory, so paths are relative to the checkout by default.

## Setup

Create an environment that provides a C/C++ toolchain and MPI implementation,
then install the project:

```bash
python -m pip install -e .
```

`pyccl` and `mpi4py` may need to be built against the local scientific/MPI
stack.

## Data access

The full KiDS/BNT input data are not distributed with this repository. If you
need to use the full data, please ask **Shiming Gu** or **Ziang Yan** at the
email addresses indicated in the paper. See [data/README.md](data/README.md)
for the expected data location.

## Run

```bash
./run_bnt.sh 000000
```

Set `BNT_MPI_WORKERS` to change the process count. Set `BNT_DATA_DIR` to the
directory containing the supplied data, and `BNT_OUTPUT_DIR` to change where
checkpoints and chains are written (default: `outputs/`).
