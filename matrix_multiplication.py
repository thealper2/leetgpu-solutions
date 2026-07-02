import jax
import jax.numpy as jnp


@jax.jit
def solve(A: jax.Array, B: jax.Array, M: int, N: int, K: int) -> jax.Array:
    return A @ B
