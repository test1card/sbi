# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Training-free unit tests for the opt-in ``compose_standardization`` feature.

Consolidated fast coverage for FMPE / NPSE composed standardization: build-time
wiring and guards, the boundary affine contract, checkpoint / state_dict loading
across sbi versions, the single-observation scope guards, and the exact
single-obs log_prob Jacobian correction. All tests here build/wire only (no
training) and run in well under a second.

The slow, training-based scale-equivariance behavior lives in
``tests/scale_equivariance_test.py``.
"""

import pickle

import pytest
import torch
from torch.distributions import Independent, Normal

from sbi.inference.posteriors.vector_field_posterior import VectorFieldPosterior
from sbi.inference.potentials.vector_field_potential import VectorFieldBasedPotential
from sbi.neural_nets.factory import posterior_flow_nn, posterior_score_nn
from sbi.neural_nets.net_builders.vector_field_nets import (
    build_vector_field_estimator,
)
from sbi.utils.sbiutils import z_standardization

NUM_DIM = 2
_COMPOSE_KEYS = ("_theta_shift", "_theta_scale", "_compose_standardization")


def _batches(dim=NUM_DIM, n=64):
    """Non-unit theta stats on purpose (mean ~100, std ~5): exercises the build
    path that zeroes the internal stats and moves the scale into the boundary
    affine, and puts from_z-corrected samples far from the z-space origin."""
    torch.manual_seed(0)
    batch_x = 100.0 + 5.0 * torch.randn(n, dim)
    batch_y = torch.randn(n, dim)
    return batch_x, batch_y


def _build(estimator_type="flow", compose=True, **kwargs):
    batch_x, batch_y = _batches()
    return build_vector_field_estimator(
        batch_x,
        batch_y,
        estimator_type=estimator_type,
        z_score_x="independent",
        compose_standardization=compose,
        **kwargs,
    )


def _wide_prior(dim=NUM_DIM):
    # Wide Normal so theta=0 (and untrained samples) lie in support.
    return Independent(Normal(torch.zeros(dim), 200.0 * torch.ones(dim)), 1)


def _posterior(estimator_type="flow", sample_with="ode", **kwargs):
    est = _build(estimator_type=estimator_type, compose=True, **kwargs)
    return VectorFieldPosterior(est, prior=_wide_prior(), sample_with=sample_with)


# --------------------------------------------------------------------------
# Build-time wiring and guards (public API)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "estimator_type,kwargs", [("flow", {}), ("score", {"sde_type": "ve"})]
)
def test_build_compose_on_unit_stats_and_affine(estimator_type, kwargs):
    """compose ON => unit internal stats (mean_0=0/std_0=1) and the per-dim
    boundary affine (_theta_shift/_theta_scale) matches z_standardization."""
    batch_x, _ = _batches()
    est = _build(estimator_type=estimator_type, compose=True, **kwargs)
    exp_mean, exp_std = z_standardization(batch_x, structured_dims=False)
    assert est.compose_enabled is True
    assert torch.allclose(est.mean_0, torch.zeros_like(est.mean_0))
    assert torch.allclose(est.std_0, torch.ones_like(est.std_0))
    assert torch.allclose(est._theta_shift.reshape(-1), exp_mean.float(), atol=1e-5)
    assert torch.allclose(
        est._theta_scale.reshape(-1), exp_std.clamp_min(1e-20).float(), atol=1e-5
    )


@pytest.mark.parametrize(
    "estimator_type,kwargs", [("flow", {}), ("score", {"sde_type": "ve"})]
)
def test_build_compose_off_identity(estimator_type, kwargs):
    """compose OFF => flag False, identity theta affine (byte-identical OFF path)."""
    est = _build(estimator_type=estimator_type, compose=False, **kwargs)
    assert est.compose_enabled is False
    assert torch.allclose(est._theta_shift, torch.zeros_like(est._theta_shift))
    assert torch.allclose(est._theta_scale, torch.ones_like(est._theta_scale))


def test_build_rejects_compose_plus_baseline():
    """Build path refuses compose_standardization + gaussian_baseline (mutually
    exclusive: the analytical baseline velocity is derived from original-space
    mean_0/std_0, composition feeds the network standardized z)."""
    batch_x, batch_y = _batches()
    with pytest.raises(
        ValueError, match="gaussian_baseline and compose_standardization"
    ):
        build_vector_field_estimator(
            batch_x,
            batch_y,
            estimator_type="flow",
            z_score_x="independent",
            gaussian_baseline=True,
            compose_standardization=True,
        )


def test_factory_rejects_compose_plus_non_independent_z_score_theta():
    """The compose/z-score-theta guard lives at the factory layer, where
    ``z_score_theta`` is unambiguous. Both FMPE (``posterior_flow_nn``) and NPSE
    (``posterior_score_nn``) must reject ``z_score_theta='none'`` under compose.

    The NPSE case is the previously-broken one: its factory maps
    ``z_score_theta`` onto the builder's ``z_score_y`` (theta-scoring lives in
    ``z_score_y`` for NPSE, ``z_score_x`` for FMPE), so a builder-level guard on
    ``z_score_x`` checked the wrong knob and silently standardized theta.
    """
    for factory in (posterior_flow_nn, posterior_score_nn):
        for bad in ("none", "structured", None):
            with pytest.raises(ValueError, match="z_score_theta"):
                factory(z_score_theta=bad, compose_standardization=True)


def test_factory_compose_ignores_z_score_x():
    """``z_score_x`` is observation-scoring and is irrelevant to compose (which
    standardizes theta). ``posterior_score_nn(z_score_x='none', compose=True)``
    must NOT raise, and the built estimator has compose enabled. Previously this
    wrongly raised because the NPSE factory routed ``z_score_x`` into the
    builder's compose guard."""
    build_fn = posterior_score_nn(
        z_score_x="none", compose_standardization=True, sde_type="ve"
    )
    batch_theta, batch_x = _batches()
    est = build_fn(batch_theta, batch_x)  # must not raise
    assert est.compose_enabled is True


# --------------------------------------------------------------------------
# Boundary affine contract — roundtrip + known-scale log_abs_det
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_affine_contract(dtype):
    """from_z(to_z(theta)) == theta in float32/float64, and log_abs_det() equals
    sum(log scale) for a hand-set (non-tautological) known scale."""
    est = _build(compose=True)
    if dtype == torch.float64:
        est = est.double()
    theta = (100.0 + 5.0 * torch.randn(10, NUM_DIM)).to(dtype)
    recon = est.from_z(est.to_z(theta))
    assert recon.dtype == dtype
    assert torch.allclose(recon, theta, atol=1e-5 if dtype == torch.float32 else 1e-10)

    # Non-tautological Jacobian check: hand-set scale to [2, 3].
    known = torch.tensor([2.0, 3.0]).to(dtype)
    est._theta_scale.copy_(known.reshape(1, NUM_DIM))
    expected = torch.log(known).sum()
    assert torch.allclose(est.log_abs_det(), expected)


@pytest.mark.gpu
def test_affine_roundtrip_cuda():
    """from_z(to_z(theta)) == theta on CUDA (buffers move with the module)."""
    est = _build(compose=True).cuda()
    theta = (100.0 + 5.0 * torch.randn(10, NUM_DIM)).cuda()
    recon = est.from_z(est.to_z(theta))
    assert recon.is_cuda
    assert torch.allclose(recon, theta, atol=1e-5)


# --------------------------------------------------------------------------
# Checkpoint / state_dict loading across sbi versions
# --------------------------------------------------------------------------


class _WrappedEstimator(torch.nn.Module):
    """Minimal wrapper that places the estimator at a non-root prefix ('est.')."""

    def __init__(self, est):
        super().__init__()
        self.est = est


def _mutate_nonunit_stats(sd):
    """Force non-unit internal stats into a compose-ON checkpoint (inconsistent)."""
    sd["mean_0"] = torch.full((NUM_DIM,), 100.0)
    sd["std_0"] = torch.full((NUM_DIM,), 5.0)


# id, src_compose, dst_compose, dst_baseline, strip_suffixes, mutate, prefixed, expect
_CHECKPOINT_CASES = [
    ("legacy_off", False, False, False, _COMPOSE_KEYS, None, False, ("off",)),
    ("full_on", True, False, False, (), None, False, ("on",)),
    ("partial_raises", True, False, False, ("_theta_scale",), None, False,
     ("raise", "_theta_scale")),
    ("prefixed_legacy_off", False, False, False, _COMPOSE_KEYS, None, True, ("off",)),
    ("baseline_reject", True, False, True, (), None, False,
     ("raise", "cannot be used together")),
    ("nonunit_reject", True, False, False, (), _mutate_nonunit_stats, False,
     ("raise", "unit")),
]


@pytest.mark.parametrize(
    "src_compose,dst_compose,dst_baseline,strip,mutate,prefixed,expect",
    [c[1:] for c in _CHECKPOINT_CASES],
    ids=[c[0] for c in _CHECKPOINT_CASES],
)
def test_checkpoint_load(
    src_compose, dst_compose, dst_baseline, strip, mutate, prefixed, expect
):
    """State_dict loading is all-or-nothing on the three compose buffers: none
    present -> loads OFF; all present -> loads ON; partial -> RuntimeError. Load
    also re-validates compose invariants (baseline-incompatible, unit stats)."""
    src_est = _build(compose=src_compose)
    src = _WrappedEstimator(src_est) if prefixed else src_est
    sd = src.state_dict()
    if strip:
        for k in list(sd.keys()):
            if any(k.endswith(s) for s in strip):
                del sd[k]
    if mutate is not None:
        mutate(sd)

    dst_est = _build(compose=dst_compose, gaussian_baseline=dst_baseline)
    dst = _WrappedEstimator(dst_est) if prefixed else dst_est

    if expect[0] == "raise":
        with pytest.raises((RuntimeError, ValueError), match=expect[1]):
            dst.load_state_dict(sd)
    else:
        dst.load_state_dict(sd)  # must not raise
        assert dst_est.compose_enabled is (expect[0] == "on")


# --------------------------------------------------------------------------
# Single-observation scope guards (unsupported paths raise NotImplementedError)
# --------------------------------------------------------------------------

# Structurally identical guards: each entry point rejects the unsupported combo
# under compose. (posterior, potential) are wired from the same estimator.
_GUARD_ACTIONS = [
    ("sample_iid",
     lambda post, pot: post.sample(
         (2,), x=torch.randn(3, NUM_DIM), show_progress_bars=False),
     "iid"),
    ("sample_guidance",
     lambda post, pot: post.sample(
         (2,), x=torch.zeros(1, NUM_DIM), guidance_method="classifier_free",
         show_progress_bars=False),
     "guided"),
    ("log_prob_iid",
     lambda post, pot: post.log_prob(
         torch.zeros(1, NUM_DIM), x=torch.randn(3, NUM_DIM)),
     "iid"),
    ("map", lambda post, pot: post.map(show_progress_bars=False), "MAP"),
    ("set_x_iid",
     lambda post, pot: pot.set_x(torch.randn(2, NUM_DIM), x_is_iid=True), "iid"),
    ("set_x_guidance",
     lambda post, pot: pot.set_x(
         torch.zeros(1, NUM_DIM), x_is_iid=False,
         guidance_method="classifier_free"),
     "guided"),
    ("sample_batched",
     lambda post, pot: post.sample_batched(
         torch.Size([3]), x=torch.randn(2, NUM_DIM)),
     "sample_batched"),
]


@pytest.mark.parametrize(
    "action", _GUARD_ACTIONS, ids=[a[0] for a in _GUARD_ACTIONS]
)
def test_compose_scope_guard(action):
    """Each unsupported entry point raises NotImplementedError up front under
    compose (single observation is the only supported path)."""
    _id, fn, match = action
    est = _build(compose=True)
    posterior = VectorFieldPosterior(est, prior=_wide_prior(), sample_with="ode")
    potential = VectorFieldBasedPotential(
        est, prior=_wide_prior(), x_o=None, device="cpu"
    )
    with pytest.raises(NotImplementedError, match=match):
        fn(posterior, potential)


# --------------------------------------------------------------------------
# from_z sample-output boundary — samples returned in ORIGINAL theta-space
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "estimator_type,sample_with,kwargs",
    [("flow", "ode", {}), ("score", "sde", {"sde_type": "ve"})],
)
def test_sample_returns_theta_space_under_compose(
    estimator_type, sample_with, kwargs
):
    """Single-obs sample under compose applies from_z on both the ODE (FMPE) and
    SDE (NPSE) paths: output lives in original theta-space (shift ~100), not
    z-space. Untrained -> only coordinate space + finiteness asserted."""
    posterior = _posterior(
        estimator_type=estimator_type, sample_with=sample_with, **kwargs
    )
    samples = posterior.sample(
        (5,),
        x=torch.zeros(1, NUM_DIM),
        reject_outside_prior=False,
        show_progress_bars=False,
    )
    assert samples.shape == (5, NUM_DIM)
    assert torch.isfinite(samples).all()
    # from_z applied => samples carry the ~100 shift; z-space samples would be O(1).
    assert samples.abs().mean() > 10.0


# --------------------------------------------------------------------------
# Pickle regression (sbi <= 0.27 posteriors have no compose buffer)
# --------------------------------------------------------------------------


def test_pickle_pre_compose_loads_as_off():
    """A posterior pickled by sbi <= 0.27 has no ``_compose_standardization``
    buffer, and unpickling bypasses ``__init__``. ``compose_enabled`` must still
    return False (via getattr default) so sampling works.

    We simulate a genuine pre-compose posterior by building without compose,
    deleting the three compose buffers from the module, and then performing a
    REAL ``pickle.dumps``/``pickle.loads`` round-trip (unpickling bypasses
    ``__init__``, exactly like loading an old artifact). The restored estimator
    must report ``compose_enabled is False`` and still sample."""
    est = _build(compose=False)
    for name in _COMPOSE_KEYS:
        if name in est._buffers:
            delattr(est, name)
    assert est.compose_enabled is False

    posterior = VectorFieldPosterior(est, prior=_wide_prior(), sample_with="ode")

    restored = pickle.loads(pickle.dumps(posterior))
    restored_est = restored.vector_field_estimator
    assert not any(name in restored_est._buffers for name in _COMPOSE_KEYS)
    assert restored_est.compose_enabled is False

    samples = restored.sample(
        (3,),
        x=torch.zeros(1, NUM_DIM),
        reject_outside_prior=False,
        show_progress_bars=False,
    )
    assert samples.shape == (3, NUM_DIM)
    assert torch.isfinite(samples).all()


# --------------------------------------------------------------------------
# Exact single-obs log_prob Jacobian correction (deterministic, no training)
# --------------------------------------------------------------------------


def test_single_obs_log_prob_jacobian_exact():
    """Deterministic pin of the affine-Jacobian log_prob correction in the
    single-obs branch of ``VectorFieldBasedPotential.__call__``.

    Wire a tiny compose estimator into the potential, set a single observation
    (builds the neural ODE -- no training), set ``_theta_scale`` to a known
    [2.0, 3.0], and monkeypatch the underlying ``flow.log_prob`` to return the
    SUM of its input coordinates. Because the stub depends on its input, the test
    is sensitive to the ``to_z()`` boundary call: if it were removed, the flow
    would see original-space theta (0) instead of ``z = to_z(0)`` and the result
    would change. This exercises the REAL ``to_z()``, ``log_abs_det()`` and the
    REAL subtraction ``log_probs = log_probs - log_abs_det`` (not a
    reimplementation), so the returned potential must equal
    ``sum(z) - log(2) - log(3)``.
    """
    est = _build(compose=True)
    # Wide prior so theta=0 lies in support (within_support uses original theta).
    potential = VectorFieldBasedPotential(
        est, prior=_wide_prior(), x_o=None, device="cpu"
    )
    potential.set_x(torch.zeros(1, NUM_DIM), x_is_iid=False)  # builds flow, no training

    # Known per-dim scale -> log_abs_det = log(2) + log(3).
    est._theta_scale.copy_(torch.tensor([[2.0, 3.0]]))

    theta = torch.zeros(1, NUM_DIM)
    # Input-dependent stub: sensitive to whether to_z() is applied to the input.
    potential.flow.log_prob = lambda inp: inp.sum(-1)

    out = potential(theta)
    # Expected uses the REAL boundary map: z = to_z(theta), then Jacobian subtract.
    expected = est.to_z(theta).sum(-1) - est.log_abs_det()
    assert torch.allclose(out.reshape(-1), expected.reshape(-1), atol=1e-6), (
        f"log_prob Jacobian/to_z correction off: got {out.reshape(-1)}, expected "
        f"{expected.reshape(-1)}"
    )
