<!-- LOCAL DRAFT — not posted. Reworded per janfb review item 6:
     frame prior work (training stability on shifted data) positively;
     this PR adds the stronger property of scale equivariance. -->

## What does this PR do?

FMPE and NPSE amortized posteriors are not equivariant under the exact
reparameterization theta -> c*theta when parameters are far from O(1) scale.
The flow / SDE integrate against a unit base, so calibration degrades for
small-scale parameters even with the default `z_score_theta`.

Prior work established time-dependent z-scoring, which delivers training
stability when the **data** is shifted. This PR adds the stronger property of
**scale equivariance**: an opt-in per-dimension composed standardization
(`compose_standardization=True`). The estimator is trained and sampled in a
standardized z-space, with an invertible affine map theta = shift + scale * z
composed at the boundaries:
- loss input standardized;
- sample output unstandardized back to theta;
- log_prob corrected by the affine Jacobian (-sum log scale);
- prior-support rejection performed in original theta-space.

Per-dimension scaling handles heterogeneous parameter scales (mixed O(1) and
O(1e-6) dimensions). The flag defaults to off; with it off, behavior is
byte-identical to current sbi, and legacy checkpoints load via identity buffers.

Scope of this PR is single-observation sampling and log_prob, which are fully
supported. iid (x with batch>1), guided sampling, and MAP raise
`NotImplementedError` under compose and are left as a documented follow-up:
- iid/guided combine the theta-space prior with the z-space score, which is not
  yet transformed;
- MAP computes the potential gradient in z-space, so theta-space gradient ascent
  would need a chain-rule rescaling.

## Changes

- `neural_nets/estimators/base.py`: standardization buffers and helpers, the
  `compose_enabled` property, and an atomic legacy-checkpoint shim.
- `neural_nets/estimators/flowmatching_estimator.py`, `score_estimator.py`:
  standardize loss input; invariant checks moved to build/load time.
- `inference/posteriors/vector_field_posterior.py`: unstandardize samples;
  guard MAP/iid/guided under compose.
- `inference/potentials/vector_field_potential.py`: affine Jacobian in log_prob.
- `neural_nets/net_builders/vector_field_nets.py`: opt-in builder kwarg;
  single source of truth for the standardization stats; `z_score_x` must be
  `'independent'` under compose.
- `tests/compose_standardization_test.py`, `tests/scale_equivariance_test.py`:
  build/affine/checkpoint/guard coverage and the equivariance regression tests.

## Validation

- FMPE and NPSE recover equivariance at scale s = 1e-5 (both SDE and ODE paths),
  covered by the scale-equivariance tests (homogeneous, heterogeneous).
- Single-observation log_prob matches the analytic affine-Jacobian correction.
- Legacy (pre-compose) checkpoints load as compose-off; pickled posteriors from
  older sbi remain loadable.
- Full non-slow test suite: passing.

## Does this close any issues?

Relates to #1680. Prior work delivered training stability on shifted data; this
PR completes the default-path case by adding scale equivariance for FMPE/NPSE.

## Checklist

- [x] Tests added (scale-equivariance, heterogeneous, affine contract, guards,
      checkpoint/pickle regression)
- [x] Slow tests marked
- [x] Code commented where non-obvious
- [x] Default behavior unchanged (flag off = byte-identical)
