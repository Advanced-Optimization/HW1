import numpy as np


# ============================================================
# Linear function: f(x, W) = W x
# ============================================================
def linear(x, W):
    """Linear function f(W) = Wx."""

    return W @ x


def linear_lmap(x, W, dx=None):
    """Differential of linear map at x along dx."""

    return None


def linear_lmap_adjoint(x, W, u):
    """Adjoint differential of linear map at x along output direction u."""

    return None
