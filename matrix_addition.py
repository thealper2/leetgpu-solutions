import jax
import jax.numpy as jnp


@jax.jit
def solve(A: jax.Array, B: jax.Array, N: int) -> jax.Array:
    return A + B
