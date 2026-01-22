import numpy as np


# ============================================================
# Linear function: f(x, W) = W x
# ============================================================
def linear(x, W):
    """Linear function f(W) = Wx."""

    return W @ x


def linear_lmap(x, W, dx=None, dW=None):
    """Differential of linear map at x along dx."""

    # first notebook
    # return None

    # second notebook
    lmap_x = None
    lmap_W = None
    return lmap_x, lmap_W


def linear_lmap_adjoint(x, W, u):
    """Adjoint differential of linear map at x along output direction u."""

    # first notebook
    # return None

    # second notebook
    lmap_adjoint_x = None
    lmap_adjoint_W = None
    return lmap_adjoint_x, lmap_adjoint_W


# ============================================================
# ReLU function: f(x) = max(0, x)
# ============================================================
def relu(x):
    """ReLU of x."""

    output = np.zeros_like(x)
    output[x >= 0] = x[x >= 0]
    return output


def relu_lmap(x, v):
    """linear map for ReLU at x in the direction v."""

    return None


def relu_lmap_adjoint(x, u):
    """Adjoint map for ReLU at x in the direction u."""

    return None


# ============================================================
# sigmoid function: f(x) = sigmoid(x)
# ============================================================
def sigmoid(x):
    """Sigmoid of x."""

    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_lmap(x, v):
    """linear map for sigmoid at x in the direction v."""

    return None


def sigmoid_lmap_adjoint(x, u):
    """Adjoint map for sigmoid at x in the direction u."""

    return None


# ============================================================
# square error function: f(y,z) = ||y - z||^2
# ============================================================
def sqe(y, z):
    """Square error between y and z."""

    return np.sum((y - z) ** 2)


def sqe_lmap(y, z, dy):
    """Directional derivative of the loss at y in the direction dy."""

    return None


def sqe_lmap_adjoint(y, z, u):
    """Adjoint directional derivative of the loss at y in the direction u."""

    return None
