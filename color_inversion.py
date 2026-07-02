import jax
import jax.numpy as jnp


@jax.jit
def solve(image: jax.Array, width: int, height: int) -> jax.Array:
    image = image.at[0::4].set(255 - image[0::4])
    image = image.at[1::4].set(255 - image[1::4])
    image = image.at[2::4].set(255 - image[2::4])
    return image
