import numpy as np

from mytorch.layers import linear, linear_lmap, linear_lmap_adjoint


# ============================================================
# Linear function finite diff test: f(x, W) = W x
# ============================================================
def _test_linear_lmap(f, f_lmap):
    """Test the linear map using finite differences."""

    lmap_exact = None
    lmap_finite_diff = None

    np.testing.assert_allclose(lmap_exact, lmap_finite_diff, rtol=1e-5, atol=1e-8)


def _test_linear_lmap_adjoint(f, f_lmap_adjoint):
    """Test the adjoint map using finite differences."""

    lmap_adjoint_exact = None
    lmap_adjoint_finite_diff = None

    np.testing.assert_allclose(
        lmap_adjoint_exact, lmap_adjoint_finite_diff, rtol=1e-5, atol=1e-8
    )


# ============================================================
# Run all tests
# ============================================================
def test_lmaps():
    for i in range(10):
        np.random.seed(i)
        _test_linear_lmap(linear, linear_lmap)


def test_lmaps_adjoint():
    for i in range(10):
        np.random.seed(i)
        _test_linear_lmap_adjoint(linear, linear_lmap_adjoint)


if __name__ == "__main__":
    test_lmaps()
    test_lmaps_adjoint()
