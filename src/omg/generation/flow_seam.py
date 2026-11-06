"""On-policy two-plan seam measurements at the 25-frame native50 switch interval."""

import torch

from omg.generation.flow_reference import FlowReferenceGenerator


@torch.inference_mode()
def replan_seam_metrics(
    generator: FlowReferenceGenerator,
    world_qpos: torch.Tensor,
    first_noise: torch.Tensor,
    next_noise: torch.Tensor,
    *,
    steps: int,
    text_tokens: torch.Tensor,
    text_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare the regenerated seam, the old plan at that time, and held-out mocap.

    The second call sees only prior generated history and the previously exposed
    prefix. The ground-truth future is used exclusively to measure accelerations.
    Each acceleration uses a backward difference ending at the frame being scored.
    """
    interval = 25
    first_edit = 2 * interval
    if generator.history > interval or generator.horizon <= first_edit + 1:
        raise ValueError("P25 replan evaluation needs history <= 25 and horizon >= 52")
    if world_qpos.shape[1] != generator.history + generator.horizon:
        raise ValueError("Expected the held-out history and one full native50 future")
    history = world_qpos[:, :generator.history]
    conditions = dict(steps=steps, text_tokens=text_tokens, text_mask=text_mask)
    first = generator.generate_suffix(history, history[:, :0], first_noise, **conditions)
    suffix = generator.generate_suffix(
        first[:, interval-generator.history:interval], first[:, interval:first_edit], next_noise,
        **conditions,
    )
    # Match all trajectories over the same absolute timestamps. Frames 0..49
    # come from plan one. Only the second plan supplies frames from 50 onward.
    replanned = torch.cat([first[:, :first_edit], suffix[:, :generator.horizon-first_edit]], dim=1)
    acceleration = torch.stack([replanned, first, world_qpos[:, generator.history:]])[..., 7:]
    acceleration = acceleration.float().diff(n=2, dim=2).abs() * 50**2
    # Stencils ending on frames 50 and 51 both cross the splice. Exclude both
    # when comparing with within-clip acceleration at other timestamps.
    seam = first_edit - 2
    interior = torch.cat([acceleration[:, :, :seam], acceleration[:, :, seam+2:]], dim=2)
    return {
        "replan_seam_joint_accel_abs_rad_s2": acceleration[0, :, seam].mean(),
        "original_same_frame_joint_accel_abs_rad_s2": acceleration[1, :, seam].mean(),
        "gt_same_frame_joint_accel_abs_rad_s2": acceleration[2, :, seam].mean(),
        "replan_interior_joint_accel_abs_rad_s2": interior[0].mean(),
        "gt_interior_joint_accel_abs_rad_s2": interior[2].mean(),
    }
