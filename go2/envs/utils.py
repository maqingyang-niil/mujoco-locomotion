import numpy as np


def _backend(x):
    if type(x).__module__.startswith("jaxlib") or type(x).__module__.startswith("jax"):
        import jax.numpy as jnp
        return jnp
    return np