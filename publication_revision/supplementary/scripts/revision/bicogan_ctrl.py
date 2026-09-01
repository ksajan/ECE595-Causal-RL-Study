"""Bidirectional conditional GAN used by the corrected CTRL reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class BiCoGANConfig:
    """Training configuration for a transition-level BiCoGAN."""

    latent_dim: int = 4
    batch_size: int = 256
    pretrain_steps: int = 2_000
    adversarial_steps: int = 5_000
    learning_rate: float = 2e-4
    reconstruction_weight: float = 10.0
    extrinsic_weight: float = 1.0
    latent_cycle_weight: float = 1.0
    validation_fraction: float = 0.2
    generator_kind: str = "triangular"


def _mlp(input_dim: int, widths: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in widths:
        layers.extend([nn.Linear(previous, width), nn.BatchNorm1d(width), nn.ReLU()])
        previous = width
    return nn.Sequential(*layers)


class UnconstrainedGenerator(nn.Module):
    """Map normalized ``(state, commanded_action, latent)`` to next state."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.body = _mlp(5 + latent_dim, (200, 400, 600, 600))
        self.output = nn.Linear(600, 4)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.cat([state, action, latent], dim=1)
        return self.output(self.body(values))


class PositiveLinear(nn.Module):
    """Linear layer with strictly positive, exponentially parameterized weights."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.bias = nn.Parameter(torch.empty(output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize effective weights on the usual linear-layer scale."""
        effective = torch.empty_like(self.raw_weight)
        nn.init.kaiming_uniform_(effective, a=np.sqrt(5.0))
        effective = effective.abs().clamp_min(1e-3)
        with torch.no_grad():
            self.raw_weight.copy_(effective.log())
        bound = 1.0 / np.sqrt(self.raw_weight.shape[1])
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, self.raw_weight.exp(), self.bias)


class MonotonicBiCoGANGenerator(nn.Module):
    """BiCoGAN generator that is coordinate-wise nondecreasing in latent noise.

    Unconstrained context projections condition every layer on ``(s, a)``.
    All paths from ``u`` to the output use exponentially positive weights.
    Batch normalization has no learned affine scale, preserving the sign of
    those paths at evaluation time.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        widths = (200, 400, 600, 600)
        self.positive_layers = nn.ModuleList()
        self.context_layers = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        previous = latent_dim
        for width in widths:
            self.positive_layers.append(PositiveLinear(previous, width))
            self.context_layers.append(nn.Linear(5, width, bias=False))
            self.normalizations.append(nn.BatchNorm1d(width, affine=False))
            previous = width
        self.positive_output = PositiveLinear(previous, 4)
        self.context_output = nn.Linear(5, 4)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        context = torch.cat([state, action], dim=1)
        features = latent
        for positive, conditional, normalization in zip(
            self.positive_layers,
            self.context_layers,
            self.normalizations,
        ):
            features = F.relu(normalization(positive(features) + conditional(context)))
        return self.positive_output(features) + self.context_output(context)


class TriangularMonotoneGenerator(nn.Module):
    """Conditional affine triangular SCM, invertible in ``u`` by construction."""

    state_dim = 4

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        if latent_dim != self.state_dim:
            raise ValueError("Triangular generator requires latent_dim == state_dim.")
        self.body = _mlp(5, (200, 400, 600, 600))
        self.location = nn.Linear(600, self.state_dim)
        self.lower_triangle = nn.Linear(600, 10)
        row, column = torch.tril_indices(self.state_dim, self.state_dim)
        self.register_buffer("row_indices", row)
        self.register_buffer("column_indices", column)

    def parameters_for(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return conditional location and lower-triangular scale matrices."""
        features = self.body(torch.cat([state, action], dim=1))
        location = self.location(features)
        triangle_values = self.lower_triangle(features)
        matrix = torch.zeros(
            state.shape[0],
            self.state_dim,
            self.state_dim,
            dtype=state.dtype,
            device=state.device,
        )
        matrix[:, self.row_indices, self.column_indices] = F.softplus(triangle_values)
        diagonal = torch.arange(self.state_dim, device=state.device)
        matrix[:, diagonal, diagonal] += 1e-3
        return location, matrix

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        location, matrix = self.parameters_for(state, action)
        return location + torch.bmm(matrix, latent.unsqueeze(2)).squeeze(2)

    def inverse(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Exactly abduct ``u`` from an observed transition under the learned SCM."""
        location, matrix = self.parameters_for(state, action)
        return torch.linalg.solve_triangular(
            matrix,
            (next_state - location).unsqueeze(2),
            upper=False,
        ).squeeze(2)


class Encoder(nn.Module):
    """Infer factual state, commanded action, and exogenous latent from next state."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.body = _mlp(4, (600, 600, 400, 200))
        self.state_head = nn.Linear(200, 4)
        self.action_head = nn.Linear(200, 1)
        self.latent_head = nn.Linear(200, latent_dim)

    def forward(
        self, next_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.body(next_state)
        state = self.state_head(features)
        action = torch.sigmoid(self.action_head(features))
        latent = self.latent_head(features)
        return state, action, latent


class JointDiscriminator(nn.Module):
    """Distinguish encoded and generated ``(s', s, a, u)`` joint samples."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.body = _mlp(9 + latent_dim, (600, 600, 400, 200))
        self.output = nn.Linear(200, 1)

    def forward(
        self,
        next_state: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.cat([next_state, state, action, latent], dim=1)
        return self.output(self.body(values)).squeeze(1)


class CTRLBiCoGAN:
    """Train and apply the bidirectional SCM used for CTRL augmentation."""

    def __init__(self, config: BiCoGANConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        if config.generator_kind == "triangular":
            generator: nn.Module = TriangularMonotoneGenerator(config.latent_dim)
        elif config.generator_kind == "monotonic_bicogan":
            generator = MonotonicBiCoGANGenerator(config.latent_dim)
        elif config.generator_kind == "unconstrained":
            generator = UnconstrainedGenerator(config.latent_dim)
        else:
            raise ValueError(f"Unknown generator kind: {config.generator_kind}")
        self.generator = generator.to(device)
        self.encoder = Encoder(config.latent_dim).to(device)
        self.discriminator = JointDiscriminator(config.latent_dim).to(device)

    def _batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = states.shape[0]
        indices = torch.randint(
            count,
            (self.config.batch_size,),
            generator=generator,
            device=self.device,
        )
        return states[indices], actions[indices], next_states[indices]

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        trial_ids: np.ndarray,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        seed: int,
        validation_states: np.ndarray | None = None,
        validation_actions: np.ndarray | None = None,
        validation_next_states: np.ndarray | None = None,
        validation_trial_ids: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Fit BiCoGAN and return independent one-step diagnostics when supplied."""
        rng = np.random.default_rng(seed + 70_000)
        external_values = (
            validation_states,
            validation_actions,
            validation_next_states,
            validation_trial_ids,
        )
        if any(value is not None for value in external_values) and not all(
            value is not None for value in external_values
        ):
            raise ValueError("External validation arrays must be supplied together.")

        if validation_states is not None:
            overlap = np.intersect1d(trial_ids, validation_trial_ids)
            if len(overlap):
                raise ValueError(
                    "External validation trial IDs overlap training trial IDs: "
                    f"{overlap.tolist()}"
                )
            training_indices = np.arange(len(states))
            validation_trials = np.unique(validation_trial_ids)
            validation_trial_count = len(validation_trials)
            raw_validation_states = validation_states
            raw_validation_actions = validation_actions
            raw_validation_next_states = validation_next_states
            validation_source = "independent_dataset"
        else:
            unique_trials = np.unique(trial_ids)
            shuffled_trials = rng.permutation(unique_trials)
            validation_trial_count = max(
                1,
                int(len(unique_trials) * self.config.validation_fraction),
            )
            validation_trials = shuffled_trials[:validation_trial_count]
            validation_mask = np.isin(trial_ids, validation_trials)
            validation_indices = np.flatnonzero(validation_mask)
            training_indices = np.flatnonzero(~validation_mask)
            raw_validation_states = states[validation_indices]
            raw_validation_actions = actions[validation_indices]
            raw_validation_next_states = next_states[validation_indices]
            validation_source = "training_trial_holdout"
        if len(training_indices) < self.config.batch_size:
            raise ValueError("BiCoGAN training split is smaller than one batch.")

        normalized_states = (states - state_mean) / state_std
        normalized_next_states = (next_states - state_mean) / state_std
        action_values = actions.astype(np.float32)[:, None] / 10.0
        normalized_validation_states = (raw_validation_states - state_mean) / state_std
        normalized_validation_next_states = (
            raw_validation_next_states - state_mean
        ) / state_std
        validation_action_values = (
            raw_validation_actions.astype(np.float32)[:, None] / 10.0
        )
        train_states = torch.as_tensor(
            normalized_states[training_indices], device=self.device
        )
        train_actions = torch.as_tensor(
            action_values[training_indices], device=self.device
        )
        train_next_states = torch.as_tensor(
            normalized_next_states[training_indices], device=self.device
        )
        validation_states = torch.as_tensor(
            normalized_validation_states, device=self.device
        )
        validation_actions = torch.as_tensor(
            validation_action_values, device=self.device
        )
        validation_next_states = torch.as_tensor(
            normalized_validation_next_states, device=self.device
        )

        torch_generator = torch.Generator(device=self.device)
        torch_generator.manual_seed(seed + 80_000)
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.learning_rate,
            betas=(0.5, 0.999),
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.learning_rate,
            betas=(0.5, 0.999),
        )
        joint_optimizer = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.encoder.parameters()),
            lr=self.config.learning_rate,
            betas=(0.5, 0.999),
        )
        logs: dict[str, list[float]] = {
            "pretrain_mse": [],
            "flow_nll": [],
            "discriminator": [],
            "adversarial": [],
            "reconstruction": [],
            "extrinsic": [],
            "latent_cycle": [],
        }

        self.generator.train()
        for step in range(self.config.pretrain_steps):
            state, action, next_state = self._batch(
                train_states,
                train_actions,
                train_next_states,
                torch_generator,
            )
            latent = torch.zeros(
                self.config.batch_size,
                self.config.latent_dim,
                device=self.device,
            )
            prediction = self.generator(state, action, latent)
            loss = F.mse_loss(prediction, next_state)
            generator_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.generator.parameters(), 10.0)
            generator_optimizer.step()
            if step % 100 == 0 or step + 1 == self.config.pretrain_steps:
                logs["pretrain_mse"].append(float(loss.item()))

        if isinstance(self.generator, TriangularMonotoneGenerator):
            flow_optimizer = torch.optim.Adam(
                self.generator.parameters(),
                lr=self.config.learning_rate,
            )
            for step in range(self.config.adversarial_steps):
                state, action, next_state = self._batch(
                    train_states,
                    train_actions,
                    train_next_states,
                    torch_generator,
                )
                latent = self.generator.inverse(state, action, next_state)
                _, scale = self.generator.parameters_for(state, action)
                diagonal = torch.diagonal(scale, dim1=1, dim2=2)
                flow_nll = (
                    0.5 * latent.square().sum(dim=1) + torch.log(diagonal).sum(dim=1)
                ).mean()
                flow_optimizer.zero_grad()
                flow_nll.backward()
                nn.utils.clip_grad_norm_(self.generator.parameters(), 10.0)
                flow_optimizer.step()
                if step % 100 == 0 or step + 1 == self.config.adversarial_steps:
                    logs["flow_nll"].append(float(flow_nll.item()))

        gan_steps = (
            0
            if isinstance(self.generator, TriangularMonotoneGenerator)
            else self.config.adversarial_steps
        )
        for step in range(gan_steps):
            state, action, next_state = self._batch(
                train_states,
                train_actions,
                train_next_states,
                torch_generator,
            )
            prior_latent = torch.randn(
                self.config.batch_size,
                self.config.latent_dim,
                generator=torch_generator,
                device=self.device,
            )

            with torch.no_grad():
                encoded_state, encoded_action, encoded_latent = self.encoder(next_state)
                generated_next_state = self.generator(state, action, prior_latent)
            encoded_logits = self.discriminator(
                next_state,
                encoded_state,
                encoded_action,
                encoded_latent,
            )
            generated_logits = self.discriminator(
                generated_next_state,
                state,
                action,
                prior_latent,
            )
            discriminator_loss = F.binary_cross_entropy_with_logits(
                encoded_logits,
                torch.ones_like(encoded_logits),
            ) + F.binary_cross_entropy_with_logits(
                generated_logits,
                torch.zeros_like(generated_logits),
            )
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0)
            discriminator_optimizer.step()

            for parameter in self.discriminator.parameters():
                parameter.requires_grad_(False)
            encoded_state, encoded_action, encoded_latent = self.encoder(next_state)
            reconstructed_next_state = self.generator(
                state,
                action,
                encoded_latent,
            )
            generated_next_state = self.generator(state, action, prior_latent)
            _, _, cycled_latent = self.encoder(generated_next_state)
            encoded_logits = self.discriminator(
                next_state,
                encoded_state,
                encoded_action,
                encoded_latent,
            )
            generated_logits = self.discriminator(
                generated_next_state,
                state,
                action,
                prior_latent,
            )
            adversarial_loss = F.binary_cross_entropy_with_logits(
                encoded_logits,
                torch.zeros_like(encoded_logits),
            ) + F.binary_cross_entropy_with_logits(
                generated_logits,
                torch.ones_like(generated_logits),
            )
            reconstruction_loss = F.mse_loss(reconstructed_next_state, next_state)
            extrinsic_loss = F.mse_loss(encoded_state, state) + F.mse_loss(
                encoded_action, action
            )
            latent_cycle_loss = F.mse_loss(cycled_latent, prior_latent)
            joint_loss = (
                adversarial_loss
                + self.config.reconstruction_weight * reconstruction_loss
                + self.config.extrinsic_weight * extrinsic_loss
                + self.config.latent_cycle_weight * latent_cycle_loss
            )
            joint_optimizer.zero_grad()
            joint_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.generator.parameters()) + list(self.encoder.parameters()),
                10.0,
            )
            joint_optimizer.step()
            for parameter in self.discriminator.parameters():
                parameter.requires_grad_(True)

            if step % 100 == 0 or step + 1 == self.config.adversarial_steps:
                logs["discriminator"].append(float(discriminator_loss.item()))
                logs["adversarial"].append(float(adversarial_loss.item()))
                logs["reconstruction"].append(float(reconstruction_loss.item()))
                logs["extrinsic"].append(float(extrinsic_loss.item()))
                logs["latent_cycle"].append(float(latent_cycle_loss.item()))

        self.generator.eval()
        self.encoder.eval()
        with torch.no_grad():
            inferred_state, inferred_action, encoder_latent = self.encoder(
                validation_next_states
            )
            if isinstance(self.generator, TriangularMonotoneGenerator):
                inferred_latent = self.generator.inverse(
                    validation_states,
                    validation_actions,
                    validation_next_states,
                )
                abduction_method = "analytic_triangular_inverse"
            else:
                inferred_latent = encoder_latent
                abduction_method = "encoder"
            validation_prediction = self.generator(
                validation_states,
                validation_actions,
                inferred_latent,
            )
            normalized_error = F.mse_loss(
                validation_prediction, validation_next_states
            ).item()
            physical_prediction = (
                validation_prediction.cpu().numpy() * state_std + state_mean
            )
            physical_target = raw_validation_next_states
            physical_rmse = np.sqrt(
                np.mean((physical_prediction - physical_target) ** 2, axis=0)
            )
            if isinstance(self.generator, TriangularMonotoneGenerator):
                state_reconstruction_mse = None
                action_reconstruction_mse = None
            else:
                state_reconstruction_mse = float(
                    F.mse_loss(inferred_state, validation_states).item()
                )
                action_reconstruction_mse = float(
                    F.mse_loss(inferred_action, validation_actions).item()
                )

        return {
            "config": asdict(self.config),
            "train_count": len(training_indices),
            "validation_count": len(raw_validation_states),
            "validation_source": validation_source,
            "validation_trial_count": int(validation_trial_count),
            "validation_trial_ids": [
                int(value) for value in np.sort(validation_trials)
            ],
            "normalized_next_state_mse": float(normalized_error),
            "next_state_rmse_by_dimension": [float(value) for value in physical_rmse],
            "state_reconstruction_mse": state_reconstruction_mse,
            "action_reconstruction_mse": action_reconstruction_mse,
            "central_action_baseline_mse": float(
                F.mse_loss(
                    torch.full_like(validation_actions, 0.5),
                    validation_actions,
                ).item()
            ),
            "latent_mean": float(inferred_latent.mean().item()),
            "latent_std": float(inferred_latent.std().item()),
            "latent_mean_by_dimension": [
                float(value) for value in inferred_latent.mean(dim=0)
            ],
            "latent_std_by_dimension": [
                float(value) for value in inferred_latent.std(dim=0)
            ],
            "abduction_method": abduction_method,
            "logs": logs,
        }

    def infer_latent(
        self,
        normalized_states: np.ndarray,
        action_values: np.ndarray,
        normalized_next_states: np.ndarray,
    ) -> np.ndarray:
        """Infer transition-specific latent values from normalized next states."""
        with torch.no_grad():
            next_state = torch.as_tensor(normalized_next_states, device=self.device)
            if isinstance(self.generator, TriangularMonotoneGenerator):
                self.generator.eval()
                latent = self.generator.inverse(
                    torch.as_tensor(normalized_states, device=self.device),
                    torch.as_tensor(action_values, device=self.device),
                    next_state,
                )
            else:
                self.encoder.eval()
                _, _, latent = self.encoder(next_state)
        return latent.cpu().numpy()

    def predict(
        self,
        normalized_states: np.ndarray,
        action_values: np.ndarray,
        latent_values: np.ndarray,
    ) -> np.ndarray:
        """Predict normalized next states for supplied interventions and latents."""
        self.generator.eval()
        with torch.no_grad():
            result = self.generator(
                torch.as_tensor(normalized_states, device=self.device),
                torch.as_tensor(action_values, device=self.device),
                torch.as_tensor(latent_values, device=self.device),
            )
        return result.cpu().numpy()

    def save(self, path: Path) -> None:
        """Save learned weights and serialized training configuration."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "encoder": self.encoder.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "config": asdict(self.config),
            },
            path,
        )
