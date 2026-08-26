Reference programs
==================

These standalone executables produce *trusted* baselines for the SIAM dynamics,
used to validate the few-body (active-window) FBR solvers in ../../test/. They are
built only when CMake is configured with -DFBR_EXAMPLE_REF=ON.

The model is the same throughout: spinful SIAM, L=100, nImp=4, hybridization
V=0.1, Hubbard U (default 0.2). The dynamics programs take U as an optional first
argument, e.g.  ./chain_dyn_siam_center 0.1


Pure-ITensor programs (raw itensor::MPS, no few-body state classes)
------------------------------------------------------------------
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

- fbr_green_irlm_L1000.cpp
    The SAME Green functions at L=1000, but computed by the FBR itself. This is
    NOT a trusted baseline: at L=1000 there is nothing to check it against, since
    a real-space chain TDVP of three states on 1000 sites is out of reach -- that
    is the point of the active-window method. What it records is the solver's own
    trajectory, so a later change of behaviour at a size the L=100 tests never
    reach shows up as a difference.

    Two things make it usable as a baseline. The reference state is a Slater
    determinant with BOTH impurity orbitals empty (from_slater, not Fbr_gs), so
    the run is deterministic: two runs give byte-identical files. An Fbr_gs
    ground state does not -- two identical runs of it gave bond dimensions 126
    and 144 and Im G(0,0) differing by 5e-4, the DMRG choosing between
    near-degenerate orbital sets. And both impurity orbitals have to be empty for
    c_j^dag|psi> to be non-zero at all on a determinant.

    So these are the Green functions of that quench, not equilibrium ones. Rows
    carry the largest bond dimension over the three states and the width of the
    active window; being integers, the replay in test_ref_green.cpp has to land
    on them exactly. Writes
        output/fbr_green_irlm_L1000_U<U>.txt
    Note the name: fbr_ marks a self-reference, chain_ a trusted baseline.

- star_dyn_siam_center.cpp
    Full TDVP in the STAR geometry (each spin's bath is diagonalised into energy
    eigenmodes). computeKstar also returns the rotation, so the correlator is
    rotated back to the original real-space basis before being written to
        output/star_dyn_siam_center_U<U>_ref.txt
    in the same format and at the same times as the chain program. The impurity
    couples to all bath eigenmodes (long-range), so the subspace expansion must be
    resolved well. TDVP params were tuned (star_dyn_tune.cpp) to match the earlier
    overkill run (nKrylov=15, err_goal=1e-8, epsilonM=1e-7, epsilonK=1e-8) at
    minimum cost. Findings, at U=0.2:
      - nKrylov is the cheap knob: 15 -> 2 cuts runtime ~4x, leaves t=20 unchanged
        (dcc 1.09e-3 vs 1.10e-3) and t<=10 within the same order (~5e-5 vs ~3e-5).
      - err_goal and epsilonM/epsilonK are sensitive: err_goal 1e-8 -> 1e-7 is fine
        (dcc 6.4e-5 at t=5) but 1e-6 breaks (4.9e-4); the expansion cutoffs tolerate
        ~3x loosening (6.7e-5) but 10x breaks (8.8e-4).
    The program now uses the tuned set nKrylov=2, err_goal=1e-7, epsilonM=3e-7,
    epsilonK=3e-8, which tracks the chain baseline as well as the overkill run.
    Kept as a record only, not used in the tests. (Much coarser expansion cutoffs
    give a ~1e-2 agreement.)

- star_dyn_siam_center_ip.cpp
    Star geometry in the interaction picture of the bath: the bath phases are
    advanced analytically (expBath) and only the interaction-picture Hamiltonian
    Kip_n is applied to the MPS via TDVP.

- star_dyn_siam_center_ipc.cpp
    Star geometry, interaction picture with an accumulated bath unitary Ubath and
    a Trotter split (exp(-iH dt) = exp(-iH_bath dt) exp(-iH^(2) dt) + O(dt^3)):
    exp(-iH_bath dt) is tracked in Ubath and never touches the MPS. Structured so
    an intra-bath rotation R can be inserted between steps (Ubath <- R Ubath).


Programs bridging to the few-body state classes (Fb_mps / Impurity[Spin])
-------------------------------------------------------------------------
- star_gs_siam.cpp
    Ground state (DMRG), spinless: Impurity + Fb_mps<double>.

- star_dyn_siam.cpp
    Dynamics, spinless: Impurity + Fb_mps<cmpx>. Interleaved up/down layout with
    the impurity at sites 0..3.

- star_dyn_siam_spin.cpp
    Dynamics, spinful: ImpuritySpin + Fb_mps<cmpx> (spin_block). Block layout, constructing
    the star model directly (bypassing toStar) to check that building the state
    through the library types reproduces the raw-ITensor result.


Output reference files
----------------------
output/{chain,star}_dyn_siam_center_U<U>_ref.txt  (format tag
chain_dyn_siam_center_ref_v1) hold ni + correlation-matrix snapshots.

- test/test_ref_{fbr,block,ns}.cpp compare each FBR variant against the CHAIN
  baseline only (the trusted standard), at every snapshot present in the file.
- output/fbr_green_irlm_L1000_U<U>.txt (format tag fbr_green_irlm_ref_v1) holds
  "t ReG00 ImG00 ReG01 ImG01 maxBondDim nActive" per step. test_ref_green.cpp
  replays the first 20 steps and requires the two integers back exactly.
- output/chain_green_irlm_U<U>_ref.txt (format tag chain_green_irlm_ref_v1) is a
  different, denser format: one row per time step, "t ReG00 ImG00 ReG01 ImG01 m".
  test/test_ref_green.cpp compares the FBR Green functions against it at every
  step in the file (up to t=5 by default, the whole file with
  -DFBR_ENABLE_LONG_TEST=ON).
- The star baseline is kept for the record but is NOT used in the tests. Its
  agreement with the chain baseline is documented, once, in
  output/star_vs_chain.txt, regenerated by compare_star_vs_chain.py.
