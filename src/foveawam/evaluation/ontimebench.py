"""Convert a capture-aligned OnTimeBench frame into FoveaWAM's public policy input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from ontimebench.api import ActionSpec, BenchmarkAction, BenchmarkObservation, validate_action_spec
from torch import Tensor

from foveawam.api import ActionSpec as ModelActionSpec
from foveawam.api import FoveaWAMPolicy, PolicyObservation

def benchmark_action_spec(model_spec: ModelActionSpec) -> ActionSpec:
    if model_spec.mode != "E7" or model_spec.gripper_mode != "discrete":
        raise ValueError("FoveaWAM requires E7 actions with a discrete gripper")
    return ActionSpec(model_spec.mode, model_spec.representation, tuple(model_spec.motion_limits))


def benchmark_action(value: Tensor, spec: ActionSpec) -> BenchmarkAction:
    """Convert one physical command from the model to a simulator action."""
    if value.device.type != "cpu" or value.dtype != torch.float32 or value.shape != (7,):
        raise ValueError("FoveaWAM commands must be CPU float32 tensors of shape (7,)")
    action = np.array(value.numpy(), dtype=np.float64, copy=True, order="C")
    action.setflags(write=False)
    result = BenchmarkAction(spec.mode, spec.representation, action)
    validate_action_spec(result, spec)
    return result


def _tensor(array: npt.NDArray[np.generic], *, dtype: npt.DTypeLike = None) -> Tensor:
    return torch.from_numpy(np.array(array, dtype=dtype, copy=True, order="C"))


@dataclass
class _FoveaWAMAdapter:
    policy: FoveaWAMPolicy
    action_spec: ActionSpec
    previous_quaternion: npt.NDArray[np.float64] | None = None

    def reset(self, task_id: str) -> None:
        self.previous_quaternion = None
        self.policy.reset(task_id)

    def _observation(self, frame: BenchmarkObservation) -> PolicyObservation:
        position, quaternion = frame.target_position, frame.target_quaternion_xyzw
        if position is None or quaternion is None:
            raise ValueError(
                "FoveaWAM needs the target pose; evaluate with target_pose_source='simulator'"
            )
        # Keep the quaternion sign continuous across ticks, as in the training data.
        if self.previous_quaternion is not None and quaternion @ self.previous_quaternion < 0:
            quaternion = -quaternion
        self.previous_quaternion = quaternion.copy()
        camera_to_base = frame.T_robot_base_from_world @ frame.world_from_camera_cv
        depth = frame.depth_m[:, None, :, :]
        opening = float(frame.gripper_position[0] - frame.gripper_position[1]) / 0.08
        return PolicyObservation(
            view_names=frame.view_names,
            rgb=_tensor(frame.rgb.transpose(0, 3, 1, 2)),
            timestamp_s=torch.tensor(frame.timestamp_s, dtype=torch.float64, device="cpu"),
            eef_position_robot_base_m=_tensor(frame.eef_position),
            eef_quaternion_robot_base_xyzw=_tensor(frame.eef_quaternion_xyzw),
            gripper_open_fraction=torch.tensor(opening, dtype=torch.float32, device="cpu"),
            camera_intrinsics=_tensor(frame.camera_intrinsics, dtype=np.float32),
            T_robot_base_from_camera=_tensor(camera_to_base, dtype=np.float32),
            depth_m=_tensor(depth),
            depth_valid=_tensor(depth > 0),
            target_position_m=_tensor(position, dtype=np.float32),
            target_quaternion_xyzw=_tensor(quaternion, dtype=np.float32),
            target_valid=torch.tensor(True, device="cpu"),
        )

    def observe(self, observation: BenchmarkObservation) -> None:
        self.policy.observe(self._observation(observation))

    def query_actions(self, observation: BenchmarkObservation) -> tuple[BenchmarkAction, ...]:
        chunk = self.policy.predict_action_chunk(self._observation(observation))
        return tuple(benchmark_action(value, self.action_spec) for value in chunk.actions)


def bind_foveawam(policy: FoveaWAMPolicy) -> _FoveaWAMAdapter:
    return _FoveaWAMAdapter(policy, benchmark_action_spec(policy.action_spec))


__all__ = ["benchmark_action", "benchmark_action_spec", "bind_foveawam"]
