"""Flow matching for motion windows conditioned on history and committed references."""

import torch
from torch import nn

from omg.generation.conditions.position import SinusoidalPositionEncoding
from omg.generation.denoisers.transformer import MotionTransformerBlock, TimestepEmbedding


class MotionFlow(nn.Module):
    """Model a normalized motion suffix, keeping its committed prefix as clean input."""

    def __init__(
        self,
        feature_dim: int = 38,
        hidden_dim: int = 512,
        num_layers: int = 10,
        num_heads: int = 8,
        dropout: float = 0.1,
        text_dim: int | None = None,
        max_prefix_length: int | None = None,
    ):
        super().__init__()
        if max_prefix_length is not None and max_prefix_length < 0:
            raise ValueError("max_prefix_length must be nonnegative")
        self.feature_dim = feature_dim
        self.max_prefix_length = max_prefix_length
        self.time_embedding = TimestepEmbedding(hidden_dim)
        # An explicit per-token indicator distinguishes committed frames even at t=1.
        self.input_proj = nn.Linear(feature_dim + hidden_dim + 1, hidden_dim)
        self.history_proj = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.history_position = SinusoidalPositionEncoding(hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim) if text_dim is not None else None
        self.layers = nn.ModuleList(
            MotionTransformerBlock(hidden_dim, num_heads, dropout=dropout, use_frame_film=False)
            for _ in range(num_layers)
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, feature_dim))

    def _context(self, history, text_tokens, text_mask):
        if history.ndim != 3 or history.shape[1] < 1 or history.shape[2] != self.feature_dim:
            raise ValueError(f"history must have shape (B, H>=1, {self.feature_dim})")
        context = self.history_proj(history)
        context = context + self.history_position(context)
        mask = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
        if (text_tokens is None) != (text_mask is None):
            raise ValueError("text_tokens and text_mask must be supplied together")
        if self.text_proj is not None:
            if text_tokens is None:
                # Empty text provides a motion-only context to the same model.
                text_tokens = history.new_empty((history.size(0), 0, self.text_proj.in_features))
                text_mask = mask[:, :0]
            if (text_tokens.ndim != 3 or text_tokens.shape[0] != history.size(0)
                    or text_tokens.shape[-1] != self.text_proj.in_features
                    or text_tokens.shape[:2] != text_mask.shape or text_mask.dtype != torch.bool):
                raise ValueError("text tokens and their boolean mask must agree with batch size and text_dim")
            context = torch.cat([context, self.text_proj(text_tokens.to(history))], dim=1)
            mask = torch.cat([mask, text_mask.to(history.device)], dim=1)
        elif text_tokens is not None:
            raise ValueError("set text_dim when constructing a text-conditioned model")
        return context, mask

    def _velocity(self, state, time, committed_mask, context, context_mask):
        # The path runs from noise (0) to data (1). Use the clean endpoint on
        # committed tokens and the current integration time on generated tokens.
        time_embedding = torch.where(
            committed_mask[..., None],
            self.time_embedding(time.new_ones(1) * 1000.0),
            self.time_embedding(time * 1000.0)[:, None, :],
        )
        hidden = self.input_proj(torch.cat([
            state, time_embedding.to(state), committed_mask[..., None].to(state.dtype),
        ], dim=-1))
        for layer in self.layers:
            hidden = layer(hidden, context, context_mask, key_padding_mask=None)
        return self.output(hidden)

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        *,
        prefix_lengths: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Average per-window suffix velocity MSE; both inputs are full normalized windows."""
        context, context_mask = self._context(history, text_tokens, text_mask)
        if future.ndim != 3 or future.shape[0] != history.size(0) or future.size(-1) != self.feature_dim:
            raise ValueError("future must have shape (B, T, feature_dim) matching history")
        batch, horizon, _ = future.shape
        if horizon < 1:
            raise ValueError("future must contain at least one frame")
        if prefix_lengths is None:
            limit = horizon - 1 if self.max_prefix_length is None else min(horizon - 1, self.max_prefix_length)
            prefix_lengths = torch.randint(limit + 1, (batch,), device=future.device)
        elif (prefix_lengths.shape != (batch,) or prefix_lengths.dtype != torch.long
                or torch.any((prefix_lengths < 0) | (prefix_lengths >= horizon))):
            raise ValueError("prefix_lengths must be a long tensor of shape (B,) with values in [0, T-1]")
        if time is None:
            time = torch.rand(batch, device=future.device)
        elif (time.shape != (batch,) or not time.is_floating_point()
                or not torch.all((time >= 0) & (time <= 1))):
            raise ValueError("time must contain one float in [0, 1] per example")
        if noise is None:
            noise = torch.randn_like(future)
        elif noise.shape != future.shape:
            raise ValueError("training noise must have the same shape as future")

        committed = torch.arange(horizon, device=future.device)[None, :] < prefix_lengths[:, None]
        state = torch.where(committed[..., None], future, torch.lerp(noise, future, time[:, None, None].to(future)))
        velocity = self._velocity(state, time, committed, context, context_mask)
        error = (velocity.float() - (future - noise).float()).masked_fill(committed[..., None], 0)
        # Each sampled prefix length receives equal weight regardless of suffix length.
        return (error.square().sum(dim=(1, 2)) / ((horizon - prefix_lengths) * self.feature_dim)).mean()

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        committed: torch.Tensor,
        horizon: int,
        *,
        steps: int = 15,
        noise: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate only the unknown suffix. Returns the full normalized future (B,T,D).

        committed is (B,P,D); an empty prefix starts a fresh plan. Optional noise
        has shape (B,T,D); only its uncommitted coordinates are used. Call eval()
        to disable transformer dropout for inference.
        """
        context, context_mask = self._context(history, text_tokens, text_mask)
        if (committed.ndim != 3 or committed.size(0) != history.size(0)
                or committed.size(-1) != self.feature_dim):
            raise ValueError("committed must have shape (B, P, feature_dim) matching history")
        batch, prefix, _ = committed.shape
        if horizon < 1 or prefix > horizon or steps < 1:
            raise ValueError("require horizon >= 1, prefix <= horizon and steps >= 1")
        full_shape = (batch, horizon, self.feature_dim)
        if noise is not None and noise.shape != full_shape:
            raise ValueError(f"sampling noise must have shape {full_shape}")
        if prefix == horizon:
            return committed

        suffix = (torch.randn((batch, horizon - prefix, self.feature_dim),
                              dtype=committed.dtype, device=committed.device)
                  if noise is None else noise[:, prefix:])
        committed_mask = (torch.arange(horizon, device=committed.device) < prefix).expand(batch, -1)
        for step in range(steps):
            state = torch.cat([committed, suffix], dim=1)
            time = torch.full((batch,), step / steps, device=committed.device, dtype=torch.float32)
            velocity = self._velocity(state, time, committed_mask, context, context_mask)
            suffix = suffix + velocity[:, prefix:].to(suffix.dtype) / steps
        return torch.cat([committed, suffix], dim=1)
