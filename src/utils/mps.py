import jax
import jax.numpy as jnp

from mujoco.mjx._src.math import quat_sub, quat_mul, quat_inv

# ============================================================================
# JAX MATH UTILS (QUAT, ROTATIONS, ETC.)
# ============================================================================

def quat_rotate(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate 3D vector v by quaternion q (wxyz format).

    Args:
        q: quaternion [w, x, y, z]
        v: 3D vector [vx, vy, vz]

    Returns:
        Rotated vector (3,)
    """
    w, x, y, z = q
    vx, vy, vz = v

    # Quaternion * vector as quaternion (0, v)
    # Compute q * (0, v)
    qw = -x * vx - y * vy - z * vz
    qx =  w * vx + y * vz - z * vy
    qy =  w * vy + z * vx - x * vz
    qz =  w * vz + x * vy - y * vx

    # Multiply result by q conjugate
    return jnp.array([
        -qw * x + qx * w - qy * z + qz * y,
        -qw * y + qy * w - qz * x + qx * z,
        -qw * z + qz * w - qx * y + qy * x
    ])

def quat_to_yaw_only(q):
    """
    Extract yaw from a quaternion (w, x, y, z) and return
    a quaternion that represents ONLY that yaw rotation.
    
    yaw = rotation around global Z axis
    pitch = rotation around Y axis
    roll = rotation around X axis
    
    Returned quaternion keeps yaw and sets pitch=roll=0.
    """
    w, x, y, z = q

    # Convert quat → yaw using standard ZYX convention
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    # Build quaternion with ONLY yaw rotation about Z:
    # q_yaw = [cos(yaw/2), 0, 0, sin(yaw/2)]
    half = yaw * 0.5
    cy = jnp.cos(half)
    sy = jnp.sin(half)

    return jnp.array([cy, 0.0, 0.0, sy])

def subtract_frame_transforms(
    pos1: jnp.ndarray,
    quat1: jnp.ndarray, 
    pos2: jnp.ndarray,
    quat2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Transform from frame 1 to frame 2."""
    pos_rel = pos2 - pos1
    pos_in_frame1 = quat_rotate(quat_inv(quat1), pos_rel)
    quat_rel = quat_mul(quat2, quat_inv(quat1))
    return pos_in_frame1, quat_rel

def quat_to_rot6(q):
    return rotation_matrix_first_two_cols(q).reshape(6,)

def rotation_matrix_first_two_cols(q):
    """
    Compute first 2 columns of rotation matrix from quaternion q (wxyz).
    Returns shape: (N, 3, 2) where columns are r0, r1.
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Rotation matrix rows
    r00 = 1 - 2*(y*y + z*z)
    r01 = 2*(x*y - w*z)
    r02 = 2*(x*z + w*y)

    r10 = 2*(x*y + w*z)
    r11 = 1 - 2*(x*x + z*z)
    r12 = 2*(y*z - w*x)

    r20 = 2*(x*z - w*y)
    r21 = 2*(y*z + w*x)
    r22 = 1 - 2*(x*x + y*y)

    col0 = jnp.stack([r00, r10, r20], axis=-1)
    col1 = jnp.stack([r01, r11, r21], axis=-1)
    return jnp.stack([col0, col1], axis=-1)  # (.., 3, 2)

def quat_rotate_inv(q, v):
    """Rotate vector v by the inverse of quaternion q."""
    return quat_rotate(quat_inv(q), v)


# Spline interpolation utilities

from functools import partial
from typing import Callable, Literal

from interpax import interp1d
from jax import vmap

InterpMethodType = Literal["zero", "linear", "cubic"]
InterpFuncType = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


"""We define the interpolation functions here so they're picklable for async."""


@partial(vmap, in_axes=(None, None, 0))
def interp_zero(tq: jax.Array, tk: jax.Array, knots: jax.Array) -> jax.Array:
    """Zero-order spline interpolation."""
    # for a zero-order spline, take the "next" knot as the control
    # ex: tq = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    #     tk = [0.0, 0.25, 0.5]
    #     inds = [0, 0, 0, 1, 1, 2]  # searchsorted trick does this
    #     interp_func(tq, tk, knots) = knots[:, inds]
    return knots[jnp.searchsorted(tk, tq, side="right") - 1]


@partial(vmap, in_axes=(None, None, 0))
def interp_linear(tq: jax.Array, tk: jax.Array, knots: jax.Array) -> jax.Array:
    """Linear spline interpolation."""
    return interp1d(tq, tk, knots, method="linear", extrap=True)


@partial(vmap, in_axes=(None, None, 0))
def interp_cubic(tq: jax.Array, tk: jax.Array, knots: jax.Array) -> jax.Array:
    """Cubic spline interpolation."""
    return interp1d(tq, tk, knots, method="cubic2", extrap=True)


def get_interp_func(method: InterpMethodType) -> InterpFuncType:
    """Get the 1D interpolation function based on the specified method.

    In particular, the function will have signature
        u_traj = interp_func(tq, tk, knots),
    where
        * tq is a 1D array of query times of shape (H,)
        * tk is a 1D array of knot times of shape (num_knots,),
        * knots is an array of shape (num_rollouts, num_knots), and
        * u_traj is the batch of interpolated trajectories of shape
            (num_rollouts, H).
    Here, we expect H to be the number of control time steps over some horizon T
    in seconds.

    Args:
        method: The interpolation method to use. Can be "zero", "linear", or
            "cubic".

    Returns:
        interp_func: The interpolation function.
    """
    if method == "zero":
        interp_func = interp_zero
    elif method == "linear":
        interp_func = interp_linear
    elif method == "cubic":
        interp_func = interp_cubic
    else:
        raise ValueError(
            f"Unknown interpolation method: {method}. "
            "Expected one of ['zero', 'linear', 'cubic']."
        )
    return interp_func


