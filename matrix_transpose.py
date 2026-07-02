import jax
import jax.numpy as jnp


@jax.jit
def solve(input: jax.Array, rows: int, cols: int) -> jax.Array:
    return input.T
