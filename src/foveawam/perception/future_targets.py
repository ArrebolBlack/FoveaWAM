"""Compute DINOv3 training targets from future frames."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from foveawam.perception.vision import (
    _IMAGE_MEAN,
    _IMAGE_STD,
    _INPUT_SIZE,
    _PATCH_COUNT,
    _PATCH_SIZE,
    _PATCHES_PER_AXIS,
    _require,
)

if TYPE_CHECKING:
    from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel

_HIDDEN_SIZE = 1024
_REGISTER_TOKENS = 4


def _frozen_config() -> DINOv3ViTConfig:
    from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig

    return DINOv3ViTConfig(
        architectures=["DINOv3ViTModel"],
        image_size=_INPUT_SIZE,
        patch_size=_PATCH_SIZE,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        hidden_act="gelu",
        attention_dropout=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        rope_theta=100.0,
        num_channels=3,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        proj_bias=True,
        mlp_bias=True,
        layerscale_value=1.0,
        drop_path_rate=0.0,
        use_gated_mlp=False,
        num_register_tokens=_REGISTER_TOKENS,
        pos_embed_shift=None,
        pos_embed_jitter=None,
        pos_embed_rescale=2.0,
        apply_layernorm=True,
        reshape_hidden_states=True,
        dtype="float32",
    )


def _load_frozen_backbone(weights_path: str) -> DINOv3ViTModel:
    from safetensors.torch import load_file
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel

    path = Path(weights_path).expanduser()
    if path.suffix != ".safetensors":
        raise ValueError("DINOv3 weights_path must name one safetensors file")
    if not path.is_file():
        raise FileNotFoundError(f"DINOv3 weights file does not exist: {path}")
    backbone = DINOv3ViTModel(_frozen_config())
    stored_state = load_file(path, device="cpu")
    expected_state = backbone.state_dict()
    expected_stored_keys = {
        name.removeprefix("model.") if name.startswith("model.layer.") else name
        for name in expected_state
    }
    _require(
        set(stored_state) == expected_stored_keys,
        "DINOv3-L weight keys differ from the frozen teacher",
    )
    canonical_state = {
        f"model.{name}" if name.startswith("layer.") else name: tensor
        for name, tensor in stored_state.items()
    }
    for name, expected in expected_state.items():
        stored = canonical_state[name]
        _require(
            stored.shape == expected.shape,
            f"DINOv3 safetensors tensor shape differs for {name}",
        )
        _require(
            stored.dtype == expected.dtype,
            f"DINOv3 safetensors tensor dtype differs for {name}",
        )
    backbone.load_state_dict(canonical_state, strict=True)
    return backbone


class _DINOv3VisionEncoder(nn.Module):
    """Frozen DINOv3 with the common RGB preprocessing and spatial patch grid."""

    output_dimension = _HIDDEN_SIZE
    patch_count = _PATCH_COUNT
    patches_per_axis = _PATCHES_PER_AXIS

    def __init__(self, backbone: DINOv3ViTModel) -> None:
        super().__init__()
        config = backbone.config
        _require(config.image_size == _INPUT_SIZE, "DINOv3 image size differs from 224")
        _require(config.patch_size == _PATCH_SIZE, "DINOv3 patch size differs from 16")
        _require(
            config.hidden_size == _HIDDEN_SIZE, "DINOv3-L must provide 1024 features per patch"
        )
        self.output_dimension = config.hidden_size
        _require(
            config.num_register_tokens == _REGISTER_TOKENS,
            "DINOv3 register-token count differs from 4",
        )
        self.backbone = backbone.requires_grad_(False)
        self.backbone.train(False)
        self.image_mean: Tensor
        self.image_std: Tensor
        self.register_buffer(
            "image_mean",
            torch.tensor(_IMAGE_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(_IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.train(False)

    def train(self, mode: bool = True) -> Self:
        super().train(False)
        return self

    def _preprocess(self, rgb: Tensor) -> Tensor:
        _require(rgb.ndim == 4 and int(rgb.shape[1]) == 3, "RGB must have axes [N,3,H,W]")
        _require(rgb.dtype == torch.uint8, "RGB must remain uint8 until vision preprocessing")
        _require(rgb.is_contiguous(), "RGB input to DINOv3 must be contiguous")
        rescaled = rgb.to(torch.float32) * (1.0 / 255.0)
        resized = functional.interpolate(
            rescaled,
            size=(_INPUT_SIZE, _INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return ((resized - self.image_mean) / self.image_std).contiguous()

    def forward(self, rgb: Tensor) -> Tensor:
        _require(rgb.ndim == 4 and int(rgb.shape[1]) == 3, "RGB must have axes [N,3,H,W]")
        _require(int(rgb.shape[0]) > 0, "DINOv3 RGB batch must be non-empty")
        _require(rgb.dtype == torch.uint8, "RGB must remain uint8 until vision preprocessing")
        _require(rgb.is_contiguous(), "RGB input to DINOv3 must be contiguous")
        patch_rows: list[Tensor] = []
        batch_size = 1 if torch.are_deterministic_algorithms_enabled() else int(rgb.shape[0])
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            with torch.no_grad():
                for row in range(0, int(rgb.shape[0]), batch_size):
                    pixel_values = self._preprocess(rgb[row : row + batch_size])
                    output = self.backbone(pixel_values=pixel_values)
                    hidden = cast(Tensor, output.last_hidden_state)
                    expected = (
                        int(pixel_values.shape[0]),
                        1 + _REGISTER_TOKENS + _PATCH_COUNT,
                        self.output_dimension,
                    )
                    _require(
                        tuple(hidden.shape) == expected,
                        f"DINOv3 token shape must be {expected}",
                    )
                    patches = hidden[:, 1 + _REGISTER_TOKENS :]
                    _require(
                        tuple(patches.shape[1:]) == (_PATCH_COUNT, self.output_dimension),
                        "DINOv3 must yield exactly 196 spatial patch tokens",
                    )
                    patch_rows.append(patches.to(torch.float32).contiguous())
        return torch.cat(patch_rows).contiguous()


@torch.no_grad()
def encode_future_dino_targets(
    encoder: nn.Module,
    future_rgb: Tensor,
    prediction_mask: Tensor,
    *,
    future_record_valid: Tensor | None = None,
    future_frame_ids: Tensor | None = None,
) -> dict[str, Tensor]:
    """Extract the teacher's 14x14 patch grid, encoding each real episode/frame once."""
    rgb = future_rgb
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("future DINO teacher must be frozen")
    b, s = prediction_mask.shape
    real_horizons = prediction_mask[..., None].expand(b, s, 8)
    if future_record_valid is not None:
        real_horizons = real_horizons & future_record_valid
    real = real_horizons.reshape(-1)
    selected = real.nonzero(as_tuple=False).flatten()
    inverse = None
    if future_frame_ids is not None:
        if (
            future_frame_ids.shape != (b, s, 8, 2)
            or future_frame_ids.dtype != torch.int64
            or future_frame_ids.device != rgb.device
        ):
            raise ValueError(
                "future teacher identities require int64 [B,S,H,episode/frame] on the RGB device"
            )
        keys = future_frame_ids.reshape(-1, 2)[real]
        _, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
        order = torch.argsort(inverse, stable=True)
        ordered_groups = inverse[order]
        first = torch.ones_like(ordered_groups, dtype=torch.bool)
        first[1:] = ordered_groups[1:] != ordered_groups[:-1]
        selected = selected[order[first]]
    images = rgb.reshape(-1, 3, 224, 224)[selected].contiguous()
    if len(images):
        encoded = encoder(images)
        if encoded.shape != (len(images), 196, 1024) or not torch.isfinite(encoded).all():
            raise ValueError("future DINO teacher must return finite [N,196,1024]")
        features = encoded.new_zeros(b * s * 8, 196, 1024)
        features[real] = encoded if inverse is None else encoded[inverse]
    else:
        features = torch.zeros(b * s * 8, 196, 1024, dtype=torch.float32, device=rgb.device)
    features = features.reshape(b, s, 8, 196, 1024).contiguous()
    return {"future_dino": features}


def build_future_dino_teacher(weights_path: str) -> nn.Module:
    """Load the frozen DINOv3-L teacher that supplies appearance targets."""
    with torch.random.fork_rng(devices=[]):
        return _DINOv3VisionEncoder(_load_frozen_backbone(weights_path))
