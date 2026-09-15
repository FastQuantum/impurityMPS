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
- **IRLM, L=100:** separate frames (measurement cutoff 1e-4) agree with the chain to 1e-4
  up to t=20, the same as the shared frame (7e-5). The loose cutoff only skips Givens gates;
  the aligned MPS is truncated at cutoff² (f14cd59). Before that fix the separate-frame
  error was 7–9e-3.
- **The separate-frame overlap is not O(n_active).** The band where the two frames differ
  covers the whole chain by t≈1, so each measurement aligns all L orbitals. For the SIAM
  (U=0.025, to t=L/2, cutoff 1e-3), few-body vs star: 4.1× faster at L=40 and L=80 but only
  1.4× at L=160, where the overlap takes 1293 s of the few-body's 1437 s. max|dG| vs the star
  is 6e-4 / 2e-3 / 5e-3 at L=40 / 80 / 160. The short IRLM benchmark (t≤2) still shows
  5× (L=20) to 23× (L=160). Wall times depend on the machine and its load.
- **Master-slave vs the average, measured on two-state runs, L=100, t≤5:** same
  accuracy (IRLM U=0.2: 7.0e-5 from the chain either way). The IRLM window is 82 orbitals
  (bond 38, 127 s) with the excitation as master vs 23 (bond 26, 22 s) with the average
  of the two correlation matrices. For the SIAM (U=0.1) both fill the chain (n_active=100)
  by t≈3. The library keeps master-slave because the basis is then simple to reason about.
  In the committed `fbr_green_shared_irlm` runs (master-slave, to t=20) the IRLM window fills
  the whole chain (n_active=100) by t=5 (U=0.1) and t=10 (U=0.2).
- **Provenance of the SIAM data (going by file dates):** the U=0.025 `fbr_green_siam` data
  was produced with the master-slave `Fbr_dyn_shared` that the library now uses (basis from
  the first state, the excitation). The U=0.1 data predates it and used the average of the
  states' correlation matrices.

## Cost of one evolution

| Program | Question | Data | Plot |
|---|---|---|---|
| `quench_vs_excitation_siam` | Window and time of one `Fbr_dyn` for the paper's quench, the excitation c₀†\|gs⟩ (`spin_block`), and \|gs⟩ itself | `{quench,excitation,gs}_siam_L{100,500}_U0.025` | `quench_vs_excitation.py` |
| `gs_frame_vs_ip_siam` | Does a co-moving frame (`fbr_dyn_frame.h`) keep the \|gs⟩ window smaller than the interaction picture? | `gs_{ip,frame}_siam_L{20,40}_U0` | `quench_vs_excitation.py` |
| `bench_buildK` | Per-step wall time against L of the spin dynamics (the O(L²) orbital bookkeeping) | prints | — |

Finding: the co-moving frame is a **negative result**. The bath propagator couples empty and
full Slater orbitals, so the window grows and the eigenstate drifts. The ~20-orbital
ground-state window of `Fbr_dyn` is physical (see `include/fbr/fbr_dyn_frame.h`).

## Bond dimension in time: few-body vs full MPS

| Program | Question | Data | Plot |
|---|---|---|---|
| `quench_bond_irlm` | Paper Fig. 2: χmax and N_active of the IRLM quench (U=0.2, L=200), FBR vs the full chain and full star | `quench_bond_irlm_{fbr,chain,star}_L200_U0.2` | `bond_dynamics.py` |
| `excitation_bond_siam` | For which U does the full star MPS of the Green-function excitation c₀↑†\|gs⟩ grow, and what does the FBR do on the same evolution? | `excitation_bond_siam_{star,fbr}_L{100,200}_U{0,0.025,0.05,0.1,0.2,0.4}` (+ `_cut1e-12`, `star_gs` checks at L=100, U=0.1) | `bond_dynamics.py` |

Both truncate every MPS at 1e-10 (the paper's ε). The full-MPS programs share `full_mps.h`, and
the star is built from the same `ImpurityParam` the FBR uses, so it is exactly the FBR's star.

Findings (2026-09-14):
- **IRLM quench (Fig. 2) is reproduced:** the FBR saturates at χmax≈70–77 and N_active=23 by
  t≈50 (paper ≈70, ≈25), at 0.7 s/step. The full star grows linearly without bound (χ=261 at
  t=100, 78 s/step). The chain sits on the bath Fermi-sea plateau (χ≈115) until t≈18, then grows
  linearly; it was stopped at χ=400, t=76.6 (105 s/step). The paper's Trotter chain climbs faster
  (400 by t≈33), presumably from a different truncation convention; the shape is the same.
  All three agree on n0(t): chain–FBR ≲1e-4, star–FBR ≲3e-4 (the FBR's dt=0.1 Trotter error).
- **SIAM excitation, star:** flat for U≤0.05 (χ≈19–25 up to t=100 at L=200). It starts to grow
  at **U≈0.1** (24→42 at L=200, from t≈35) and grows clearly for U=0.2, 0.4. |gs⟩ itself stays flat
  under the same TDVP (`star_gs`), so the growth is the excitation's.
- **SIAM excitation, FBR (spin_block):** agrees with the star on n0↑(t) to ≲4e-4, runs 5× faster
  at L=200 (window ≈25 sites instead of 200), and is trivial at U=0 (χ=2, N_active=4). But its
  bond dimension grows **as fast or faster** than the star's for U≥0.1 (U=0.1: 46 vs 42 at t=100;
  U=0.4: 85 vs 58 at t≈55). N_active saturates (25–32); the growth is inside the window. Unlike
  the quench, the few-body frame does not bound the excitation's entanglement.

  | L | U | star χ, t=10 → t=L/2 | FBR χ, t=10 → t=L/2 | FBR N_active at t=L/2 | wall star / FBR (s) | max\|Δn0↑\| |
  |---|---|---|---|---|---|---|
  | 100 | 0 | 15 → 15 | 2 → 2 | 4 | 236 / 35 | 3.3e-4 |
  | 100 | 0.025 | 16 → 16 | 10 → 11 | 16 | 283 / 41 | 3.9e-4 |
  | 100 | 0.05 | 17 → 17 | 11 → 17 | 18 | 267 / 46 | 3.2e-4 |
  | 100 | 0.1 | 18 → 23 | 14 → 26 | 20 | 276 / 56 | 1.7e-4 |
  | 100 | 0.2 | 19 → 33 | 17 → 38 | 24 | 322 / 119 | 2.0e-4 |
  | 100 | 0.4 | 20 → 48 | 22 → 66 | 28 | 339 / 157 | 2.5e-4 |
  | 200 | 0 | 19 → 19 | 2 → 2 | 4 | 1089 / 156 | 3.0e-4 |
  | 200 | 0.025 | 19 → 22 | 12 → 15 | 19 | 1130 / 204 | 2.6e-4 |
  | 200 | 0.05 | 22 → 25 | 14 → 24 | 23 | 1260 / 207 | 3.2e-4 |
  | 200 | 0.1 | 24 → 42 | 18 → 46 | 25 | 1381 / 279 | 2.3e-4 |
  | 200 | 0.2 | 24 → 64 | 23 → 75 | 30 | 1250 / 448 | 2.5e-4 |
  | 200 | 0.4 | 24 → 93 | 29 → 119 | 36 | 1640 / 1342 | 2.4e-4 |

  (Wall time is the evolution only, one core each, 16–20 runs sharing the machine.)
- **spin_symmetric is wrong for the excitation:** `Fb_mps::apply` mirrors the down block of `cc`
  onto the up block, so n0↑ reads n0↓ after one step (0.651 instead of 0.996). The MPS drifts
  too: the window and the rotations come from the down sector only, so the window stays at 12
  instead of 14–15 (L=40, U=0.1) and the up electron's spread is cut off. n0↑ read straight
  from the MPS is off by 3e-4 at t=3, ten times the spin_block–star agreement. `Fbr_dyn` and
  `Fbr_dyn_shared` now throw on a state that is not its own mirror image under
  `spin_symmetric`; use `spin_block` for spin-polarized states.

## Tuning and exploration

| Program | What it is |
|---|---|
| `star_dyn_tune` + `plot/tune_compare.py` | Sweep of the TDVP subspace-expansion parameters of the star SIAM run, compared with the chain reference. Produced the tuned set in `test/ref/star_dyn_siam_center.cpp` |
| `fbr_dyn_tune` | The same for the FBR SIAM dynamics (`err_goal`, `n_iter_diag`) |
| `star_dyn_siam_center_ip`, `star_dyn_siam_center_ipc` | Star SIAM in the interaction picture of the bath (analytic bath phases; accumulated bath unitary + Trotter split). They print only |
| `star_gs_siam`, `star_dyn_siam`, `star_dyn_siam_spin` | Checks that building the state through the library types reproduces the raw-ITensor star runs. They print only |
