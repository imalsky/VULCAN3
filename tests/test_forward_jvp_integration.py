"""Full adaptive-integration Kzz tangent against independently converged FD.

Uses the paper's HD189 photo-off comparison, without saved output or sibling
dependencies. A subprocess isolates the import-frozen chemistry network.
"""

import pytest


def _check_gradient():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vulcan3.config import load_config
    from vulcan3.legacy_io import Output
    from vulcan3.op_jax import Ros2JAX
    from vulcan3.outer_loop import OuterLoop
    from vulcan3.state import RunState

    # nega_cut: photo off, the zero CO cell at layer 148 falls ~0.08 cm^-3/s, so the shipped -1 rejects every step past
    # dt ~14 s and the column never leaves t ~1e5 s (notes "Forward mode"). At -inf every negative clips, the map this
    # bar was measured on; the column certifies at 793 steps with no rejection.
    cfg = load_config(
        "default",
        use_photo=False,
        use_print_prog=False,
        yconv_cri=1e-3,
        yconv_min=1e-3,
        nega_cut=float("-inf"),
    )
    integ = OuterLoop(Ros2JAX(), Output(), cfg=cfg)
    state0, static = integ.prepare_runstate(RunState.with_pre_loop_setup(cfg))

    def run(kzz):
        final = integ._runner(state0, static._replace(Kzz=kzz))
        return final.y / jnp.sum(final.y, axis=1, keepdims=True), final

    _, tangent, nominal = jax.jvp(run, (static.Kzz,), (static.Kzz,), has_aux=True)
    eps = 1e-3
    plus, state_plus = run((1 + eps) * static.Kzz)
    minus, state_minus = run((1 - eps) * static.Kzz)
    for final in (nominal, state_plus, state_minus):
        assert integ._classify_end_case(final) == 1, "FD/JVP column did not converge"
    fd = np.asarray((plus - minus) / (2 * eps))
    tangent = np.asarray(tangent)
    assert np.isfinite(fd).all()
    assert np.isfinite(tangent).all()
    high = np.abs(fd) >= np.quantile(np.abs(fd), 0.99)
    assert np.min(np.abs(fd[high])) > 0
    relative = np.abs((tangent[high] - fd[high]) / fd[high])
    print(f"Kzz JVP/FD: median={np.median(relative):.6g}, max={relative.max():.6g}, cells={high.sum()}", flush=True)
    # Paper comparison is below 0.1% on these cells; cutting the adaptive
    # controller's tangent produces roughly 40% error despite identical primals.
    assert relative.max() < 1e-3


@pytest.mark.slow
def test_kzz_jvp_matches_reconverged_finite_difference(tmp_path):
    from _helpers import run_self

    run_self(__file__, network="thermo/NCHO_photo_network.txt", atom_list="H,O,C,N", cwd=tmp_path)


if __name__ == "__main__":
    _check_gradient()
