# TODO

## Tests / references

- [x] **Improve star_vs_chain precision.** Done: it was the TDVP subspace
  expansion, not the ordering. With epsilonM=1e-7, epsilonK=1e-8 the star baseline
  now tracks chain to ~3e-5 (see `example/ref/output/star_vs_chain.txt`), on par
  with FBR-vs-chain.

- [ ] **Use the default `TdvpParam` in the ref tests.** `test/test_ref_*.cpp`
  currently pass per-variant `iterate({...})` overrides. Switch them to the
  default and re-calibrate `chainTol()` in `test/test_ref_common.h` to the new
  agreement.

- [ ] **Use the default `TdvpParam` everywhere.** Audit the codebase/examples for
  hand-tuned TDVP parameters and prefer the default unless an override is truly
  needed (and, if so, document why). Motivated by the finding that custom
  expansion cutoffs *hurt* the star run.

- [ ] **Make `test_ref*` obviously correct by reading it.** A test should be easy
  to be convinced of by inspection. The current shared header
  (`test_ref_common.h`) is DRY but templated (`compareTrajectory`, lambdas);
  revisit whether more explicit, linear per-variant code would read better and be
  easier to trust, even at the cost of some duplication.
