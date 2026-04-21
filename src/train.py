import jax
import jax.numpy as jnp
import optax
from model import rk4_step

@jax.jit
def mse_loss(true, pred):
    return jnp.mean((true - pred)**2)

def data_driven_loss(params, t_data, true_data, window_size=3):
    dt = t_data[1] - t_data[0] # Always 0.1
    total_loss = 0.0
    for i in range(window_size):
        y_curr = true_data[:-window_size] 
        y_next_true = true_data[i+1 : len(true_data) - window_size + i + 1] # Look at what is the next point on the time series
        
        if i == 0:
            y_pred = jax.vmap(rk4_step, in_axes=(None, 0, None))(params, y_curr, dt) # Derive all 6 variables
        else:
            y_pred = jax.vmap(rk4_step, in_axes=(None, 0, None))(params, y_pred, dt)
            
        total_loss += mse_loss(y_next_true, y_pred)
    return total_loss / window_size

# Fix newer JAX error for optimizer not being an array but rather python functions
def make_update_fn(optimizer):
    @jax.jit
    def update(params, opt_state, t_data, true_data):
        loss, grads = jax.value_and_grad(data_driven_loss)(params, t_data, true_data)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    return update