import jax
import jax.numpy as jnp
from jax import random

# Jax needs manual initializing params and keys
def init_network_params(sizes, key):
    def random_layer_params(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (m, n)), scale * random.normal(b_key, (n,))
    
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

# No softplus because training data has negative derivates
def forward(params, x):
    inputs = jnp.atleast_1d(x)
    *hidden, last = params
    for W, b in hidden:
        inputs = jax.nn.tanh(inputs @ W + b)
    W_last, b_last = last
    return inputs @ W_last + b_last

# Numerically solve what is the next window step (3 days)
def rk4_step(params, y, dt):
    k1 = forward(params, y)
    k2 = forward(params, y + 0.5 * dt * k1)
    k3 = forward(params, y + 0.5 * dt * k2)
    k4 = forward(params, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)