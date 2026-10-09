# Codex adversarial-review brief (local, untracked)

Challenge the UNCOMMITTED changes on branch `fix/scale-equivariance-fmpe-npse`
(`git diff` shows them) against BOTH (1) the PR goal and (2) maintainer janfb's
review. Be adversarial: find correctness regressions, missed review points, and
any place a fix only appears to satisfy janfb.

## PR goal
Make FMPE/NPSE amortized posteriors scale-equivariant under theta -> c*theta via
an OPT-IN per-dim composed standardization (`compose_standardization=True`). The
estimator trains/samples in z-space; affine `theta = shift + scale*z` is applied
only at boundaries (loss input, sample output, log_prob Jacobian). Flag OFF must
be byte-identical to current sbi. Single-obs sample + log_prob supported;
iid/guided/MAP/sample_batched raise NotImplementedError (documented follow-up).

## janfb's blocking review items (CHANGES_REQUESTED)
1. **Pickle backward-compat (real bug).** Posteriors pickled by sbi <= 0.27 have
   no `_compose_standardization` buffer and unpickling skips `__init__`, so the
   old `_compose_enabled` mirror attribute is absent -> `AttributeError` on first
   `.sample()`. Fix janfb suggested: a safe `@property compose_enabled` returning
   `bool(getattr(self, "_compose_standardization", False))`, drop the mirror, use
   the property at every check site.
2. **z_score_x guard.** Under compose, the entire `z_score_x` argument is ignored
   (incl. `'none'`), so a user disabling z-scoring still gets standardized. Make
   it one rule: compose=True allows only `'independent'`; everything else raises
   with a one-line reason. Also branch on compose first to avoid computing
   `z_standardization` twice.
3. **Move per-step invariant checks to build/load.** `_check_compose_baseline_compatible`
   and `_check_compose_internal_stats_unit` ran on every ODE/SDE step (in
   `forward`/`loss`). Call them once: in `_wire_compose` (build) and at the end of
   `_load_from_state_dict` (checkpoint). Keep the two methods.
4. **Delete the unreachable iid backstop** in `vector_field_potential.__call__`
   (only reachable by mutating `_x_is_iid` directly; guarded upstream). GSOC is
   refactoring this file — touch minimally.
5. **Reduce test volume** to ~10 fast tests in one `compose_standardization_test.py`
   + trimmed slow file; rename `test_*.py` -> `*_test.py`; unify `_fit` build route
   (both models via `posterior_flow_nn`); `@pytest.mark.gpu` not skipif; drop seed
   parametrization and the `test_default_collapses_without_compose` tests; add
   pickle-regression + z_score_x guard tests.
6. **Reword PR description** so it doesn't read as "#1752 failed" (it delivered
   training stability on shifted data; this adds scale equivariance).

## What to scrutinize hardest
- Does the `compose_enabled` property actually fix the pickle path, or just the
  `state_dict` path? Any remaining raw `_compose_standardization` truth-tests that
  reintroduce a CUDA host/device sync on a hot path?
- Moving checks to build/load: is there ANY runtime path (manual assembly, load
  of a compose+baseline or non-unit-stats checkpoint) that now silently produces a
  wrong model because the per-step guard is gone?
- z_score_x guard: does treating `None` vs `'independent'` differ from the default
  path? Could a legitimate default call now raise?
- log_prob Jacobian sign/magnitude: `log p_theta = log p_z - log_abs_det` with
  `log_abs_det = sum(log scale)`. Verify direction.
- Deleting the backstop: is the public API still fully guarded for compose+iid
  log_prob?
- Do the consolidated tests actually exercise the fixed behavior, or assert
  weaker properties (isfinite-only, implementation-vs-itself)?
