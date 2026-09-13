# Numerical experiments

Programs that run the FBR (or a brute-force comparison) to *learn* something:
method comparisons, benchmarks, tuning drivers and negative results. The
baselines they are measured against live in [`../test/ref/`](../test/ref/readme.txt)
(`chain_*` is the ground truth, `star_*` the full-length star); nothing here is
read by the test suite.

- **Build:** configure with `-DFBR_APP=ON`; every `*.cpp` here becomes one executable in `build/app/`.
- **Run from the repository root:** the programs write `app/output/<program>_L<L>_U<U>.dat`
  (U exactly as typed on the command line).
- **Versioned:** only the `.dat` files. Logs and scratch output in `app/output/`, and
  everything under `app/plot/figures/`, are ignored by git.
- **Plots:** `python3 app/plot/report.py` regenerates every figure and writes
  `app/plot/report.html`. Each `app/plot/<study>.py` also runs on its own.
  `app/plot/common.py` has the loader, which reads any data file in the repo by column name.

## Green functions

| Program | Question | Data | Plot |
|---|---|---|---|
| `fbr_green_siam` | SIAM impurity G00 from one shared frame (`Fbr_dyn_shared`), at large L | `fbr_green_siam_L{100,200,500,1000}_U0.1`, `_L{100,200,500}_U0.025` | `green_siam.py` |
| `fbr_green_sep_irlm` | IRLM G00, G01 with each state in its own frame (`green_overlap.h`) | `fbr_green_sep_irlm_L100_U{0.1,0.2}` | `green_irlm.py` |
| `fbr_green_shared_irlm` | The same IRLM run in one shared frame, same parameters | `fbr_green_shared_irlm_L100_U{0.1,0.2}` | `green_irlm.py` |
| `bench_green_cost` | IRLM: few-body separate frames vs the full star, time per sweep against L | `bench_green_cost_L{20,40,80,160}_U0.2` | `bench_green_cost.py` |
| `bench_green_cost_siam` | SIAM: the same, split into evolution and overlap, up to t=L/2 | `bench_green_cost_siam_L{40,80,160}_U0.025` | `bench_green_cost.py` |

Findings so far:
- **SIAM, shared frame:** G00 agrees with the chain baseline to 2e-5 at L=100 (the star baseline only to 3e-4).
  But the shared window grows to the whole chain (n_active reaches L), so the method does
  not scale.
- **IRLM, L=100:** the shared frame agrees with the chain to 7e-5 up to t=20. The committed
  separate-frame data (cutoff 1e-4) is off by 7–9e-3.
- **The separate-frame data here predates the band fix in `green_overlap.h` (f14cd59).**
  The band left out mismatched columns, so the `|dG|` column of `bench_green_cost_siam_*`
  exceeds 1 by t≈0.3 and the few-body overlap timings are for a band that was too small.
- **Provenance of the SIAM data (going by file dates):** the U=0.025 `fbr_green_siam` data
  was produced with a master-slave `Fbr_dyn_shared` (basis from the first state only), which
  is not in the library yet. The U=0.1 data predates that change and used the averaged
  correlation matrix that is in the library now.

## Cost of one evolution

| Program | Question | Data | Plot |
|---|---|---|---|
| `quench_vs_excitation_siam` | Window and time of one `Fbr_dyn` for the paper's quench, the excitation c₀†\|gs⟩, and \|gs⟩ itself | `{quench,excitation,gs}_siam_L{100,500}_U0.025` | `quench_vs_excitation.py` |
| `gs_frame_vs_ip_siam` | Does a co-moving frame (`fbr_dyn_frame.h`) keep the \|gs⟩ window smaller than the interaction picture? | `gs_{ip,frame}_siam_L{20,40}_U0` | `quench_vs_excitation.py` |
| `bench_buildK` | Per-step wall time against L of the spin dynamics (the O(L²) orbital bookkeeping) | prints | — |

Finding: the co-moving frame is a **negative result**. The bath propagator couples empty and
full Slater orbitals, so the window grows and the eigenstate drifts. The ~20-orbital
ground-state window of `Fbr_dyn` is physical (see `include/fbr/fbr_dyn_frame.h`).

## Tuning and exploration

| Program | What it is |
|---|---|
| `star_dyn_tune` + `plot/tune_compare.py` | Sweep of the TDVP subspace-expansion parameters of the star SIAM run, compared with the chain reference. Produced the tuned set in `test/ref/star_dyn_siam_center.cpp` |
| `fbr_dyn_tune` | The same for the FBR SIAM dynamics (`err_goal`, `n_iter_diag`) |
| `star_dyn_siam_center_ip`, `star_dyn_siam_center_ipc` | Star SIAM in the interaction picture of the bath (analytic bath phases; accumulated bath unitary + Trotter split). They print only |
| `star_gs_siam`, `star_dyn_siam`, `star_dyn_siam_spin` | Checks that building the state through the library types reproduces the raw-ITensor star runs. They print only |
