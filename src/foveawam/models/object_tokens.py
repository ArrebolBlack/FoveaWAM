"""Bounded oracle pose/kinematics tokens for the common world conditioning path.

Only raw poses inside one prediction window enter this module. No derivative or
encoded state survives the call. Numeric statistics are supplied by a train-only
preparation pass and travel in the resolved model configuration and state.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from foveawam.models.io import (
    _predicate_results,
    _require_predicate_results,
    _unit_quaternion_rotation_code,
)

_OBJECT_FEATURES = 25


def _object_motion_descriptors(
    position: Tensor,
    quaternion: Tensor,
    timestamps: Tensor,
    valid: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return values, numeric validity, token validity, and relative token time.

    Numeric order: p(3), logR(3), dp(3), log(dR)(3), dt(1), v(3),
    omega(3), a(3), alpha(3). Interval i occupies slot i and describes
    observations i -> i+1; the final slot is reserved for the current pose.
    Acceleration uses two adjacent valid intervals, never a predecessor outside
    this window. Rotations and vectors use the fixed robot-base coordinate frame.
    """
    if valid.ndim != 2 or not 1 <= valid.shape[1] <= 16 or valid.dtype != torch.bool:
        raise ValueError("object validity must be bool [B,T], 1 <= T <= 16")
    if (
        position.shape != (*valid.shape, 3)
        or quaternion.shape != (*valid.shape, 4)
        or timestamps.shape != valid.shape
        or timestamps.dtype != torch.float64
    ):
        raise ValueError("object descriptors require aligned pose and float64 timestamps")
    checks = [
        (
            torch.isfinite(position).all()
            & torch.isfinite(quaternion).all()
            & torch.isfinite(timestamps).all(),
            "object descriptor inputs must be finite",
        )
    ]

    def rotation_code(value: Tensor) -> Tensor:
        # Keep the original normalization, including the second normalization
        # of pose quaternions, but read all numeric predicates only once.
        rotation_norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        checks.append(((rotation_norm > 1e-8).all(), "rotation quaternion must be nonzero"))
        return _unit_quaternion_rotation_code(torch.div(value, rotation_norm), "rotvec")

    batch, length = valid.shape
    values = position.new_zeros(batch, length, _OBJECT_FEATURES)
    features = torch.zeros_like(values, dtype=torch.bool)
    relative = timestamps - timestamps[:, -1:]
    checks.append((~((relative > 0) & valid).any(), "object window must remain causal"))
    checks.append((~((relative < -0.750001) & valid).any(), "object history exceeds 750 ms"))
    token_time = relative.clone()
    identity = torch.zeros_like(quaternion)
    identity[..., 3] = 1
    quaternion = torch.where(valid[..., None], quaternion, identity)
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    checks.append((~(norm <= 0).any(), "valid object quaternions must be nonzero"))
    quaternion = quaternion / norm
    pose_valid = valid.clone()
    pose_valid[:, :-1] = False
    values[..., :3] = position
    values[..., 3:6] = rotation_code(quaternion)
    features[..., :6] = pose_valid[..., None]
    if length > 1:
        interval_valid = valid[:, :-1] & valid[:, 1:]
        dt = torch.diff(timestamps, dim=1)
        checks.append(
            (
                ~((dt <= 0) & interval_valid).any(),
                "valid object intervals require strictly increasing timestamps",
            )
        )
        safe_dt = torch.where(interval_valid, dt, torch.ones_like(dt)).to(position.dtype)
        dp = position[:, 1:] - position[:, :-1]
        # q_i * conjugate(q_{i-1}) represents R_i R_{i-1}^T in base coordinates.
        left, right = quaternion[:, 1:], quaternion[:, :-1]
        xyz = (
            -left[..., 3:] * right[..., :3]
            + right[..., 3:] * left[..., :3]
            - torch.linalg.cross(left[..., :3], right[..., :3], dim=-1)
        )
        scalar = left[..., 3:] * right[..., 3:] + (left[..., :3] * right[..., :3]).sum(
            -1, keepdim=True
        )
        dr = rotation_code(torch.cat((xyz, scalar), dim=-1))
        velocity, omega = dp / safe_dt[..., None], dr / safe_dt[..., None]
        values[:, :-1, 6:19] = torch.cat((dp, dr, safe_dt[..., None], velocity, omega), dim=-1)
        features[:, :-1, 6:19] = interval_valid[..., None]
        # Form centers after subtracting the current clock origin in float64.
        centers = (relative[:, :-1] + relative[:, 1:]) * 0.5
        token_time[:, :-1] = centers
        if length > 2:
            acceleration_valid = interval_valid[:, :-1] & interval_valid[:, 1:]
            center_dt = torch.diff(centers, dim=1)
            safe_center_dt = torch.where(acceleration_valid, center_dt, torch.ones_like(center_dt))
            acceleration = torch.diff(velocity, dim=1) / safe_center_dt[..., None].to(
                position.dtype
            )
            angular_acceleration = torch.diff(omega, dim=1) / safe_center_dt[..., None].to(
                position.dtype
            )
            values[:, 1:-1, 19:25] = torch.cat((acceleration, angular_acceleration), dim=-1)
            features[:, 1:-1, 19:25] = acceleration_valid[..., None]
    values = values.masked_fill(~features, 0)
    tokens = features.any(dim=-1)
    _require_predicate_results(
        _predicate_results(*(check for check, _ in checks)),
        tuple(message for _, message in checks),
    )
    return values, features, tokens, token_time.masked_fill(~tokens, 0).to(torch.float32)


class _ObjectMotionTokens(nn.Module):
    """Embed the pose and finite-difference motion features at each timestamp."""

    mean: Tensor
    std: Tensor

    def __init__(self, width: int, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        # Validate the small configuration constants on CPU even during a meta
        # parameter-count construction. Normal model.to(...) moves the buffers.
        mean_t = torch.tensor(mean, device="cpu")
        std_t = torch.tensor(std, device="cpu")
        if (
            mean_t.shape != (_OBJECT_FEATURES,)
            or std_t.shape != (_OBJECT_FEATURES,)
            or not bool(torch.isfinite(mean_t).all() & torch.isfinite(std_t).all())
            or not bool((std_t > 0).all())
        ):
            raise ValueError("object tokens require explicit finite 25D train mean/positive std")
        self.register_buffer("mean", mean_t.to(torch.float32))
        self.register_buffer("std", std_t.to(torch.float32))
        self.mlp = nn.Sequential(nn.Linear(28, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, values: Tensor, features: Tensor) -> Tensor:
        normalized = ((values - self.mean) / self.std).masked_fill(~features, 0)
        flags = torch.stack((features[..., 0], features[..., 6], features[..., 19]), dim=-1)
        encoded: Tensor = self.mlp(torch.cat((normalized, flags.to(normalized.dtype)), dim=-1))
        return encoded.to(torch.float32)
