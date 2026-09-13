Reference programs
==================

These standalone executables produce the *reference data* in output/: the
baselines that new code and new numerical experiments are measured against.
They are built only when CMake is configured with -DFBR_EXAMPLE_REF=ON.

What belongs here, by prefix:
  chain_  trusted baseline: full MPS in the real-space chain (the ground truth)
  star_   full MPS in the star geometry (whole chain active, no window)
  fbr_    the FBR's own trajectory or state, only when a test replays it
Experiments, benchmarks and tuning drivers go to app/ instead, with their data
in app/output/ (see app/README.md).

The dynamics programs take U as an optional first argument, e.g.
./chain_dyn_siam_center 0.1


Programs
--------
- chain_dyn_siam_center.cpp
    Full two-site TDVP in the CHAIN (real-space) geometry: spin-up and spin-down
    are two nearest-neighbour chains and the impurity cluster sits contiguously at
    the centre. This basis is MPS-friendly, so it is the *gold-standard* baseline
    (it agrees with all three FBR variants to ~1e-4). Writes ni + full correlation
    matrix <c_i^dag c_j> snapshots to
        output/chain_dyn_siam_center_U<U>_ref.txt
    at t = 0, 0.1, 5, 10, 20 (dt=0.1). The run stops early, keeping the snapshots
    collected so far, if the bond dimension exceeds 1024.

- chain_green_irlm.cpp
    The impurity Green functions G(0,0) and G(0,1) of the SPINLESS IRLM (a
    different model from the SIAM used by everything else here: L=100 chain of
    hopping 0.5, impurity sites 0 and 1, V=0.1, e_imp=-U/2, Hubbard U between the
    two impurity sites -- the model of example/fbr_dyn_irlm.cpp).

        G(i,j,t) = -i <psi0| c_i(t) c_j^dag(0) |psi0>

    is a matrix element between two states, so |psi0>, c_0^dag|psi0> and
    c_1^dag|psi0> are all evolved and G = -i <c_i^dag A(t) | B_j(t)>. Everything
    is done in the real-space chain (DMRG + two-site TDVP, no orbital rotation
    and no interaction picture), which makes this the trusted baseline the same
    way chain_dyn_siam_center is for the correlators. Writes one row per time
    step, t = 0 .. 20 at dt=0.1, to
        output/chain_green_irlm_U<U>_ref.txt
    rewriting the file every step, and stops early if the bond dimension of any
    of the three states reaches 1024. At U=0 the run also compares itself with
    the analytic free-fermion Green function: it agrees to 4.3e-8, which is what
    says the baseline is right. Used by test/test_ref_green.cpp.

- fbr_green_gs.cpp
    The FBR ground states test_ref_green.cpp starts from (L=100, U=0.1 and 0.2),
    saved once because computing them is the slow part of the test:
        fbr_green_gs irlm  ->  output/fbr_green_gs_L100_U<U>.dat
        fbr_green_gs siam  ->  output/fbr_green_gs_siam_L100_U<U>.dat
    (written to the working directory; copy them into output/). The test still
    checks what it loads, against the chain baseline at t=0. Fbr_gs is not
    bit-reproducible, so regenerate only the model you changed.

- star_dyn_siam_center.cpp
    Full TDVP in the STAR geometry (each spin's bath is diagonalised into energy
    eigenmodes). computeKstar also returns the rotation, so the correlator is
    rotated back to the original real-space basis before being written to
        output/star_dyn_siam_center_U<U>_ref.txt
    in the same format and at the same times as the chain program. The impurity
    couples to all bath eigenmodes (long-range), so the subspace expansion must be
    resolved well. TDVP params were tuned (app/star_dyn_tune.cpp) to match the earlier
    overkill run (n_krylov=15, err_goal=1e-8, epsilon_M=1e-7, epsilon_K=1e-8) at
    minimum cost. Findings, at U=0.2:
      - n_krylov is the cheap knob: 15 -> 2 cuts runtime ~4x, leaves t=20 unchanged
        (dcc 1.09e-3 vs 1.10e-3) and t<=10 within the same order (~5e-5 vs ~3e-5).
      - err_goal and epsilon_M/epsilon_K are sensitive: err_goal 1e-8 -> 1e-7 is fine
        (dcc 6.4e-5 at t=5) but 1e-6 breaks (4.9e-4); the expansion cutoffs tolerate
        ~3x loosening (6.7e-5) but 10x breaks (8.8e-4).
    The program now uses the tuned set n_krylov=2, err_goal=1e-7, epsilon_M=3e-7,
    epsilon_K=3e-8, which tracks the chain baseline as well as the overkill run.
    (The tuning drivers star_dyn_tune.cpp / fbr_dyn_tune.cpp are in app/.)
    Kept as a record only, not used in the tests. (Much coarser expansion cutoffs
    give a ~1e-2 agreement.)

- chain_green_siam.cpp, star_green_siam.cpp
    The impurity Green function G00(t) = -i <c_0^dag A(t)|B(t)> of the SPINFUL
    SIAM (impurity coupled by V=0.1 to a bath of hopping 0.5, Hubbard U), the
    model of app/fbr_green_siam.cpp. Both keep the whole L-site MPS (DMRG + two-
    site TDVP, no orbital rotation): the chain in the MPS-friendly centre layout
    of chain_dyn_siam_center, the star with each spin's bath diagonalised (the
    basis the FBR evolves in; its bond dimension climbs fast, which is the cost
    the active window removes). Both stop when a bond dimension reaches 1024.
    Usage: chain_green_siam [L] [tmax] [U] [dt]  (defaults 100, L/2, 0.1, 0.1).
    Run from the repository root; they write
        output/{chain,star}_green_siam_L<L>_U<U>.dat
    with columns "t bond_dim wall_s ReG00 ImG00 n0 ReC0n ImC0n" (n0 the impurity
    occupation, C0n the impurity-to-first-bath-site correlator). Committed: chain
    at L=100 for U=0.1 (to t=50) and U=0.2 (to t=20), which test_ref_green.cpp
    compares the FBR G00 against; star at L=100,200,500(,1000) for U=0.1 and
    0.025, up to t=L/2 or the bond-dimension stop, for the record.


Output reference files
----------------------
output/{chain,star}_dyn_siam_center_U<U>_ref.txt  (format tag
chain_dyn_siam_center_ref_v1) hold ni + correlation-matrix snapshots.

- test/test_ref_{fbr,block,ns}.cpp compare each FBR variant against the CHAIN
  baseline only (the trusted standard), at every snapshot present in the file.
- output/chain_green_irlm_U<U>_ref.txt (format tag chain_green_irlm_ref_v1) is a
  different, denser format: one row per time step, "t ReG00 ImG00 ReG01 ImG01 m".
  test/test_ref_green.cpp compares the FBR Green functions against it at every
  step (up to t=2 by default, t=20 with -DFBR_ENABLE_LONG_TEST=ON).
- The star baseline is kept for the record but is NOT used in the tests. Its
  agreement with the chain baseline is documented, once, in
  output/star_vs_chain.txt, regenerated by compare_star_vs_chain.py.
- output/chain_green_siam_L100_U<U>.dat is a plain table ('#' comment lines, then
  "t bond_dim wall_s ReG00 ImG00 n0 ReC0n ImC0n"). test/test_ref_green.cpp
  compares the FBR SIAM G00 against it the same way (t<=2, or t<=20 long).
