import time

import diffrax as dfx
import distrax as dsx
import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import optax
from superiorflows import Flow


class MLPVelocity(eqx.Module):
    """MLP velocity field with time conditioning."""

    mlp: eqx.nn.MLP

    def __init__(self, input_dim: int, width: int, depth: int, *, key):
        self.mlp = eqx.nn.MLP(
            in_size=input_dim + 1,
            out_size=input_dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, x, args):
        t_feat = jnp.broadcast_to(t, x.shape[:-1] + (1,))
        return self.mlp(jnp.concatenate([x, t_feat], axis=-1))


class MaximumLikelihoodLoss(eqx.Module):
    """Computes the Negative Log Likelihood (NLL) loss for a Flow model."""

    base_distribution: dsx.Distribution
    flow_kwargs: dict = eqx.field(static=True)

    def __init__(self, base_distribution, **flow_kwargs):
        self.base_distribution = base_distribution
        self.flow_kwargs = flow_kwargs

    def __call__(self, velocity_field, batch, key=None):
        flow = Flow(
            velocity_field=velocity_field,
            base_distribution=self.base_distribution,
            **self.flow_kwargs,
        )
        return -jnp.mean(flow.log_prob(batch))


@eqx.filter_jit
def training_step(model, opt_state, batch, key, loss_module, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_module)(model, batch, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def main():
    print("Initializing...")
    key = jax.random.key(0)
    d = 2

    # 1. Setup Model
    key, subkey = jax.random.split(key)
    model = MLPVelocity(input_dim=d, width=64, depth=3, key=subkey)

    # 2. Setup Base & Target
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(d), jnp.ones(d))

    angles = jnp.arange(8) * (jnp.pi / 4)
    locs = 10.0 * jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=1)
    target_dist = dsx.MixtureSameFamily(
        mixture_distribution=dsx.Categorical(probs=jnp.ones(8) / 8),
        components_distribution=dsx.MultivariateNormalDiag(loc=locs, scale_diag=jnp.full((8, 2), 0.7)),
    )

    # 3. Setup Training
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    loss_fn = MaximumLikelihoodLoss(
        base_distribution=base_dist,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
    )

    # 4. Prepare Data with Grain
    # We use a simple pre-sampling strategy for this example.
    # In a real scenario, this source would stream from disk or generate on-the-fly.
    print("Preparing data...")
    batch_size = 256
    n_steps = 1000

    key, subkey = jax.random.split(key)
    training_data = target_dist.sample(seed=subkey, sample_shape=(n_steps * batch_size,))

    dataset = grain.MapDataset.source(training_data)
    dataset = dataset.batch(batch_size)
    train_loader = iter(dataset)

    print("Starting training loop...")

    # JIT Compilation Step
    key, subkey = jax.random.split(key)
    first_batch = next(train_loader)  # Helper to trigger compilation

    t0_compile = time.time()
    model, opt_state, loss = training_step(model, opt_state, first_batch, subkey, loss_fn, optimizer)
    loss.block_until_ready()
    print(f"Compilation + Step 0: {time.time() - t0_compile:.4f}s, Loss: {loss:.4f}")

    # Training Loop
    t0_train = time.time()

    # We explicitly number the steps for logging, continuing from 1
    for step, batch in enumerate(train_loader, start=1):
        key, subkey = jax.random.split(key)
        model, opt_state, loss = training_step(model, opt_state, batch, subkey, loss_fn, optimizer)

        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss:.4f}")

    total_time = time.time() - t0_train
    print(f"Training finished in {total_time:.2f}s ({total_time/(n_steps-1)*1000:.1f} ms/step)")
    print(f"Final Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
