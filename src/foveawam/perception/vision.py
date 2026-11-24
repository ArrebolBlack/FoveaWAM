"""Spatial patch encoders for RGB observations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

_INPUT_SIZE = 224
_PATCH_SIZE = 16
_PATCHES_PER_AXIS = _INPUT_SIZE // _PATCH_SIZE
_PATCH_COUNT = _PATCHES_PER_AXIS**2
_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_encoder_state(
    encoder: nn.Module,
    stored_state: Mapping[str, Tensor],
) -> None:
    expected_state = encoder.state_dict()
    _require(
        set(stored_state) == set(expected_state),
        "vision deployment state keys differ from the encoder",
    )
    for name, expected in expected_state.items():
        stored = stored_state[name]
        _require(
            stored.shape == expected.shape,
            f"vision tensor shape differs for {name}",
        )
        _require(
            stored.dtype == expected.dtype,
            f"vision tensor dtype differs for {name}",
        )


def _build_vision_encoder(*, weights_path: str) -> _MAEVisionEncoder:
    with torch.random.fork_rng(devices=[]):
        return _MAEVisionEncoder(_load_mae_backbone(weights_path))


def _build_deployment_vision_encoder(
    *,
    identity: str,
    input_size_px: int,
    trainable: bool,
    state_dict: Mapping[str, Tensor],
) -> _MAEVisionEncoder:
    _require(
        identity == "mae_vitb16" and input_size_px == _INPUT_SIZE and not trainable,
        "FoveaWAM uses the frozen MAE ViT-B/16 at 224 pixels",
    )
    with torch.random.fork_rng(devices=[]):
        mae_encoder = _MAEVisionEncoder(_MAEBackbone())
    _validate_encoder_state(mae_encoder, state_dict)
    mae_encoder.load_state_dict(state_dict, strict=True)
    return mae_encoder


class _MAEAttention(nn.Module):
    """MAE's native fused-QKV attention, retaining its checkpoint names."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(768, 3 * 768)
        self.proj = nn.Linear(768, 768)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, 12, 64).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        projected: Tensor = self.proj(x.transpose(1, 2).reshape(batch, tokens, width))
        return projected


class _MAEBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(768, eps=1e-6)
        self.attn = _MAEAttention()
        self.norm2 = nn.LayerNorm(768, eps=1e-6)
        self.mlp = nn.ModuleDict({"fc1": nn.Linear(768, 3072), "fc2": nn.Linear(3072, 768)})

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        result: Tensor = x + self.mlp["fc2"](functional.gelu(self.mlp["fc1"](self.norm2(x))))
        return result


class _MAEBackbone(nn.Module):
    """The encoder of the original MAE ViT-B/16; no decoder or masking.

    Architecture/state names match facebookresearch/mae models_mae.py and
    DreamVLA models/vit_mae.py. Keeping the original spatial order matters:
    native random_masking permutes patches even with mask_ratio=0.
    """

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.ModuleDict({"proj": nn.Conv2d(3, 768, 16, stride=16)})
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768), requires_grad=False)
        self.blocks = nn.ModuleList([_MAEBlock() for _ in range(12)])
        self.norm = nn.LayerNorm(768, eps=1e-6)

    def forward(self, pixels: Tensor) -> Tensor:
        patches = self.patch_embed["proj"](pixels).flatten(2).transpose(1, 2)
        patches = patches + self.pos_embed[:, 1:]
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(len(pixels), -1, -1)
        hidden = torch.cat((cls, patches), dim=1)
        for block in self.blocks:
            hidden = block(hidden)
        normalized: Tensor = self.norm(hidden)
        return normalized[:, 1:]


def _load_mae_backbone(weights_path: str) -> _MAEBackbone:
    path = Path(weights_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MAE-B weights file does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _require(
        isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict),
        "MAE-B checkpoint must contain its original model state dictionary",
    )
    stored = checkpoint["model"]
    backbone = _MAEBackbone()
    expected = backbone.state_dict()
    extras = set(stored) - set(expected)
    _require(
        all(name.startswith("decoder_") or name == "mask_token" for name in extras),
        "MAE-B checkpoint contains unexpected non-decoder state",
    )
    encoder_state = {name: value for name, value in stored.items() if name not in extras}
    _validate_encoder_state(backbone, encoder_state)
    backbone.load_state_dict(encoder_state, strict=True)
    return backbone


class _MAEVisionEncoder(nn.Module):
    """Frozen ImageNet-normalized MAE-B with row-major 14x14 spatial patches."""

    output_dimension = 768
    patch_count = _PATCH_COUNT
    patches_per_axis = _PATCHES_PER_AXIS

    def __init__(self, backbone: _MAEBackbone) -> None:
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        self.image_mean: Tensor
        self.image_std: Tensor
        self.register_buffer(
            "image_mean", torch.tensor(_IMAGE_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(_IMAGE_STD).view(1, 3, 1, 1), persistent=False
        )
        self.train(False)

    def train(self, mode: bool = True) -> Self:
        super().train(False)
        return self

    def _preprocess(self, rgb: Tensor) -> Tensor:
        pixels = functional.interpolate(
            rgb.float() / 255.0,
            (_INPUT_SIZE, _INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return ((pixels - self.image_mean) / self.image_std).contiguous()

    def forward(self, rgb: Tensor) -> Tensor:
        _require(
            rgb.ndim == 4 and rgb.shape[0] > 0 and rgb.shape[1] == 3,
            "RGB must have nonempty axes [N,3,H,W]",
        )
        _require(rgb.dtype == torch.uint8 and rgb.is_contiguous(), "RGB must be contiguous uint8")
        with torch.no_grad(), torch.autocast(device_type=rgb.device.type, enabled=False):
            pixels = self._preprocess(rgb)
            # Keep the frozen MAE, including its norms, in the stored precision.
            patches: Tensor = self.backbone(pixels.to(dtype=self.backbone.cls_token.dtype))
        _require(tuple(patches.shape) == (len(rgb), 196, 768), "MAE-B patch grid differs")
        return patches.float().contiguous()


def _apply_frozen_mae_precision(model: nn.Module, precision: str) -> None:
    """Apply the declared frozen precision after the training body becomes FP32."""
    encoders = [module for module in model.modules() if isinstance(module, _MAEVisionEncoder)]
    if len(encoders) != 1:
        raise ValueError("MAE observation configuration requires exactly one frozen MAE encoder")
    if precision == "bf16":
        encoders[0].backbone.bfloat16()
