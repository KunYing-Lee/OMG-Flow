"""Continue a committed native-50 Hz G1 world reference from an OMG-Flow checkpoint."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from omg.data.bones import encode_bones_window
from omg.generation.flow import MotionFlow
from omg.motion.feature_codec import heading_quaternions
from omg.utils.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_multiply,
    rotation_6d_to_matrix,
)


class FlowReferenceGenerator(nn.Module):
    def __init__(self, model: MotionFlow, history: int, horizon: int, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        if history < 1 or horizon < 1 or model.feature_dim != 38:
            raise ValueError("A world-reference checkpoint requires native38, positive history and horizon")
        if (mean.shape != (38,) or std.shape != (38,) or not torch.isfinite(mean).all()
                or not (torch.isfinite(std) & (std > 0)).all()):
            raise ValueError("A world-reference checkpoint requires finite native38 means and positive finite scales")
        self.model = model
        self.history = history
        self.horizon = horizon
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu"):
        """Load a trusted training checkpoint, including its native38 normalization."""
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        model = MotionFlow(**saved["model_config"])
        model.load_state_dict(saved["model"], strict=True)
        return cls(
            model, saved["config"]["history"], saved["config"]["horizon"],
            torch.as_tensor(saved["mean"], dtype=torch.float32), torch.as_tensor(saved["std"], dtype=torch.float32),
        ).to(device)

    @torch.inference_mode()
    def generate_suffix(
        self,
        history_qpos: torch.Tensor,
        committed_qpos: torch.Tensor,
        noise: torch.Tensor,
        *,
        steps: int = 15,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return only editable native-50 Hz world qpos36 (xyz, wxyz, 29 G1 joints).

        history_qpos [B,H,36] immediately precedes committed_qpos [B,P,36].
        Pass the previously exposed world reference in its original world frame.
        For an initial plan use P=0. Noise is always full horizon [B,T,38].
        Text inputs are frozen T5 embeddings and their boolean attention mask.
        The caller retains the original P world frames in its reference buffer.
        """
        if history_qpos.ndim != 3 or history_qpos.shape[1:] != (self.history, 36):
            raise ValueError(f"history_qpos must have shape (B, {self.history}, 36)")
        batch = history_qpos.size(0)
        if (committed_qpos.ndim != 3 or committed_qpos.size(0) != batch
                or committed_qpos.size(-1) != 36 or committed_qpos.size(1) > self.horizon):
            raise ValueError(f"committed_qpos must have shape (B, P <= {self.horizon}, 36)")
        if noise.shape != (batch, self.horizon, 38):
            raise ValueError(f"noise must have shape (B, {self.horizon}, 38)")

        observed = torch.cat([history_qpos, committed_qpos], dim=1).to(self.mean)
        known = encode_bones_window(observed, self.history, self.mean, self.std)
        plan = self.model.sample(
            known[:, :self.history], known[:, self.history:], self.horizon,
            steps=steps, noise=noise.to(self.mean), text_tokens=text_tokens, text_mask=text_mask,
        )
        features = plan[:, committed_qpos.size(1):] * self.std + self.mean
        anchor = observed[:, self.history - 1:self.history]
        heading, _ = heading_quaternions(anchor[..., 3:7])
        orientation = F.normalize(matrix_to_quaternion(rotation_6d_to_matrix(features[..., 3:9])), dim=-1)
        return torch.cat([
            quaternion_apply(heading, features[..., :3]) + anchor[..., :3],
            F.normalize(quaternion_multiply(heading, orientation), dim=-1),
            features[..., 9:],
        ], dim=-1)
