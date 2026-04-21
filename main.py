import jax
import optax
import diffrax
import matplotlib.pyplot as plt
from model import init_network_params, forward
from train import make_update_fn
from data_utils import fetch_and_clean_data, derive_latent_states

# 1. Setup Data
C_obs, D_obs = fetch_and_clean_data()
t_data, true_data, N = derive_latent_states(C_obs, D_obs)

# 2. Initializing the model
params = init_network_params([6, 64, 64, 64, 6], jax.random.PRNGKey(0))
lr_schedule = optax.exponential_decay(1e-3, transition_steps=500, decay_rate=0.9)
optimizer = optax.adam(learning_rate=lr_schedule)
opt_state = optimizer.init(params)
update = make_update_fn(optimizer)

# 3. Training Loop
epochs = 10000
loss_history = []
print("Starting Training...")
for epoch in range(epochs):
    # Newer JAX does not like functions inside of jit
    params, opt_state, loss = update(params, opt_state, t_data, true_data)
    loss_history.append(loss)
    if epoch % 500 == 0:
        print(f"Epoch {epoch:04d} | Loss: {loss:.3e}")

# 4. Inference using Diffrax
def learned_field(t, y, args): return forward(args, y)
term = diffrax.ODETerm(learned_field)
sol = diffrax.diffeqsolve(term, diffrax.Dopri5(), t0=0.0, t1=float(len(t_data)), 
                          dt0=0.1, y0=true_data[0], args=params, 
                          saveat=diffrax.SaveAt(ts=t_data.flatten()))
pred_data = sol.ys

# 5. Final Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
labels = ['Susceptible', 'Exposed', 'Infectious', 'Recovered', 'Dead', 'Confirmed']

for i in range(6):
    ax1.plot(t_data, true_data[:, i] * N, label=f'True {labels[i]}', linestyle='--')
    ax1.plot(t_data, pred_data[:, i] * N, label=f'Learned {labels[i]}', alpha=0.7)
ax1.set_title("SEIRD Model: True vs PINN Prediction")
ax1.legend()

#Save to an image
ax2.plot(loss_history, color='purple')
ax2.set_yscale('log')
ax2.set_title("Training Convergence")

plt.savefig('pinn_results.png')
print("Results saved to pinn_results.png")

fig2, ax = plt.subplots(figsize=(16, 6))
ax.plot(t_data, true_data[:, 1] * N, label='True Exposed', linestyle='--')
ax.plot(t_data, pred_data[:, 1] * N, label='Learned Exposed', alpha=0.7)
ax.set_xlabel("Days from 2020-2021")
ax.set_ylabel("2020 Population in millions")
ax.set_title("Exposed")
ax.legend()
plt.savefig('pinn_results_close-up.png')
print("Close up saved to pinn_results_close-up.png")