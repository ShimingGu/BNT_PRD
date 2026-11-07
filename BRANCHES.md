# Branch layout

- `main` — original BNT Nautilus KiDS likelihood pipeline for
  [arXiv:2607.04384](https://arxiv.org/abs/2607.04384), restored to its
  pre-acceleration state (`6149c3f`, tag `paper-2607.04384`).  Keep this branch
  faithful to the published pipeline; acceleration work does not belong here.
- `accelerated` — acceleration work only: GPU CAMB engine integration, CPU
  likelihood optimizations, benchmarks.  Branched from `main` at `6149c3f`.
- `euclid-mock` — code for arXiv:2511.09544 and arXiv:2412.14704, as noted in
  the README.
