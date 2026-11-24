"""Attention over scene, summary and action tokens."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from foveawam.models.common.rope import _apply_3d_rope


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_mask(mask: Tensor, *, batch: int, length: int, device: torch.device) -> None:
    _require(mask.dtype == torch.bool, "attention masks must be boolean")
    _require(tuple(mask.shape) == (batch, length), "attention mask must have axes [B,L]")
    _require(mask.device == device, "attention masks and tokens must share a device")


class _WorldActionAttention(nn.Module):
    """World tokens attend to observations. Action tokens can also attend to one another."""

    allowed_keys = ((0,), (0, 1), (0, 1, 2))

    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        _require(
            heads > 0 and dimension % heads == 0, "width must be divisible by the number of heads"
        )
        _require(dimension // heads >= 6, "3D RoPE needs at least six channels per head")
        self.dimension = dimension
        self._prepared_bf16_inference = False
        self.heads = heads
        self.head_dimension = dimension // heads
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.output = nn.Linear(dimension, dimension)

    def project(
        self,
        tokens: Tensor,
        *,
        query_mask: Tensor,
        key_mask: Tensor,
        geometry_mask: Tensor | None,
        position_m: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project one raw group and apply explicit robot-base geometry once.

        Geometry validity only selects rotation; it never removes a key. Padding
        is zeroed before projection/rotation so invalid values cannot pollute
        valid tokens. Position values outside valid geometry are not consumed.
        A pair of None geometry arguments explicitly denotes a nongeometric group;
        callers own checking that groups requiring geometry actually provide it.
        """

        _require(tokens.ndim == 3, "attention tokens must have axes [B,L,D]")
        batch, length, dimension = (int(axis) for axis in tokens.shape)
        _require(dimension == self.dimension, "attention token dimension does not match")
        _check_mask(query_mask, batch=batch, length=length, device=tokens.device)
        _check_mask(key_mask, batch=batch, length=length, device=tokens.device)
        token_valid = query_mask | key_mask
        clean_tokens = tokens.masked_fill(~token_valid[..., None], 0.0)
        projected = self.qkv(clean_tokens)
        projected = projected.reshape(batch, length, 3, self.heads, self.head_dimension)
        query, key, value = projected.permute(2, 0, 3, 1, 4).unbind(0)
        _require(
            (geometry_mask is None) == (position_m is None),
            "geometry_mask and position_m must be provided together",
        )
        if geometry_mask is not None:
            assert position_m is not None
            _check_mask(geometry_mask, batch=batch, length=length, device=tokens.device)
            _require(
                tuple(position_m.shape) == (batch, length, 3),
                "attention positions must have axes [B,L,3]",
            )
            _require(position_m.device == tokens.device, "attention positions must share device")
            rotate = geometry_mask & token_valid
            positions = position_m.masked_fill(~rotate[..., None], 0.0)
            query, key = _apply_3d_rope(query, key, positions)
            # RoPE may promote its math under autocast; cached Q/K/V share the
            # projection compute dtype at the attention boundary.
            query, key = query.to(value.dtype), key.to(value.dtype)
        query = query.masked_fill(~query_mask[:, None, :, None], 0.0)
        key = key.masked_fill(~key_mask[:, None, :, None], 0.0)
        value = value.masked_fill(~key_mask[:, None, :, None], 0.0)
        return query, key, value

    def project_kv(
        self,
        tokens: Tensor,
        key_mask: Tensor,
        position_m: Tensor | None,
        geometry_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return graph-preserving, post-RoPE K/V for a layer's raw input group."""

        _, key, value = self.project(
            tokens,
            query_mask=key_mask,
            key_mask=key_mask,
            geometry_mask=geometry_mask,
            position_m=position_m,
        )
        return key, value

    def project_query(
        self,
        tokens: Tensor,
        query_mask: Tensor,
        position_m: Tensor | None,
        geometry_mask: Tensor | None,
    ) -> Tensor:
        """Project only Q with the same QKV parameter when K/V are already cached."""

        _require(tokens.ndim == 3, "attention tokens must have axes [B,L,D]")
        batch, length, dimension = (int(axis) for axis in tokens.shape)
        _require(dimension == self.dimension, "attention token dimension does not match")
        _check_mask(query_mask, batch=batch, length=length, device=tokens.device)
        clean_tokens = tokens.masked_fill(~query_mask[..., None], 0.0)
        projection = self.qkv
        bias = projection.bias
        query = (
            functional.linear(
                clean_tokens,
                projection.weight[: self.dimension],
                None if bias is None else bias[: self.dimension],
            )
            .reshape(batch, length, self.heads, self.head_dimension)
            .transpose(1, 2)
        )
        _require(
            (geometry_mask is None) == (position_m is None),
            "geometry_mask and position_m must be provided together",
        )
        if geometry_mask is not None:
            assert position_m is not None
            _check_mask(geometry_mask, batch=batch, length=length, device=tokens.device)
            _require(
                tuple(position_m.shape) == (batch, length, 3),
                "attention positions must have axes [B,L,3]",
            )
            _require(position_m.device == tokens.device, "attention positions must share device")
            positions = position_m.masked_fill(~(geometry_mask & query_mask)[..., None], 0.0)
            projection_dtype = query.dtype
            query, _ = _apply_3d_rope(query, query, positions)
            query = query.to(projection_dtype)
        return query.masked_fill(~query_mask[:, None, :, None], 0.0)

    def _attention_values(
        self, query: Tensor, key: Tensor, value: Tensor, allowed: Tensor
    ) -> Tensor:
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed[:, None],
            dropout_p=0.0,
        )
        batch, _, length, _ = attended.shape
        return attended.transpose(1, 2).reshape(batch, length, self.dimension)

    def _project_output(self, attended: Tensor, allowed: Tensor) -> Tensor:
        if attended.shape[1] == 0:
            return attended
        projected: Tensor = self.output(attended)
        return projected.masked_fill(~allowed.any(dim=-1)[..., None], 0)

    def _attend_shared_noise_prefix(
        self,
        queries: tuple[Tensor, ...],
        keys: tuple[Tensor, ...],
        values: tuple[Tensor, ...],
        query_masks: tuple[Tensor, ...],
        key_masks: tuple[Tensor, ...],
        repeats: int,
    ) -> tuple[Tensor, ...]:
        """Pack independent noise queries; keep one differentiable Scene/Meta bank.

        The additional Action keys are block masked by noise identity. Every
        query still has one softmax over exactly its original allowed keys.
        Cached K already includes its original RoPE; it is never rotated again.
        """
        expanded, heads, horizon, width = (int(axis) for axis in queries[2].shape)
        _require(
            repeats > 1 and expanded % repeats == 0 and horizon > 0,
            "shared noise query axes differ",
        )
        base = expanded // repeats
        device, dtype = queries[2].device, queries[2].dtype
        for index, (q, k, v, qm, km) in enumerate(
            zip(queries, keys, values, query_masks, key_masks, strict=True)
        ):
            _require(
                q.shape[:2] == (expanded, heads) and q.shape[-1] == width,
                "shared noise query shape differs",
            )
            _require(
                k.shape == v.shape
                and k.shape[:2] == (base if index < 2 else expanded, heads)
                and k.shape[-1] == width,
                "shared noise key/value shape differs",
            )
            _require(
                all(t.device == device and t.dtype == dtype for t in (q, k, v)),
                "shared noise attention groups must share device/dtype",
            )
            _check_mask(qm, batch=expanded, length=int(q.shape[2]), device=device)
            _check_mask(km, batch=expanded, length=int(k.shape[2]), device=device)
        _require(
            all(q.shape[2] == 0 for q in queries[:2]),
            "shared noise prefix only consumes Action queries",
        )
        _require(keys[2].shape == queries[2].shape, "shared noise Action key/query axes differ")

        def pack(value: Tensor) -> Tensor:
            return (
                value.reshape(repeats, base, heads, horizon, width)
                .permute(1, 2, 0, 3, 4)
                .reshape(base, heads, repeats * horizon, width)
            )

        prefix_masks = [key_masks[i].reshape(repeats, base, -1)[0] for i in (0, 1)]
        prefix_keys = [keys[i].masked_fill(~prefix_masks[i][:, None, :, None], 0.0) for i in (0, 1)]
        prefix_values = [
            values[i].masked_fill(~prefix_masks[i][:, None, :, None], 0.0) for i in (0, 1)
        ]
        query = pack(queries[2].masked_fill(~query_masks[2][:, None, :, None], 0.0))
        key = pack(keys[2].masked_fill(~key_masks[2][:, None, :, None], 0.0))
        value = pack(values[2].masked_fill(~key_masks[2][:, None, :, None], 0.0))
        query_valid = (
            query_masks[2].reshape(repeats, base, horizon).permute(1, 0, 2).reshape(base, -1)
        )
        action_valid = (
            key_masks[2].reshape(repeats, base, horizon).permute(1, 0, 2).reshape(base, -1)
        )
        noise = torch.arange(repeats, device=device).repeat_interleave(horizon)
        own_noise = noise[:, None] == noise[None, :]
        prefix_valid = torch.cat(prefix_masks, dim=1)
        allowed = query_valid[:, :, None] & torch.cat(
            (
                prefix_valid[:, None, :].expand(-1, repeats * horizon, -1),
                own_noise[None] & action_valid[:, None, :],
            ),
            dim=-1,
        )
        attended = self._attention_values(
            query,
            torch.cat([*prefix_keys, key], dim=2),
            torch.cat([*prefix_values, value], dim=2),
            allowed,
        )
        attended = (
            attended.reshape(base, repeats, horizon, self.dimension)
            .permute(1, 0, 2, 3)
            .reshape(expanded, horizon, self.dimension)
        )
        valid_output = (
            allowed.any(dim=-1)
            .reshape(base, repeats, horizon)
            .permute(1, 0, 2)
            .reshape(expanded, horizon)
        )
        output = self.output(attended).masked_fill(~valid_output[..., None], 0)
        empty = tuple(q.transpose(1, 2).reshape(expanded, 0, self.dimension) for q in queries[:2])
        return (*empty, output)

    def attend(
        self,
        queries: tuple[Tensor, ...],
        keys: tuple[Tensor, ...],
        values: tuple[Tensor, ...],
        *,
        query_masks: tuple[Tensor, ...],
        key_masks: tuple[Tensor, ...],
        _skip_empty_queries: bool = False,
        _prefix_noise_repeats: int = 1,
    ) -> tuple[Tensor, ...]:
        if _prefix_noise_repeats != 1:
            return self._attend_shared_noise_prefix(
                queries, keys, values, query_masks, key_masks, _prefix_noise_repeats
            )
        batch = int(queries[0].shape[0])
        device, dtype = queries[0].device, queries[0].dtype
        skip_empty = not torch.is_grad_enabled() or _skip_empty_queries
        used_keys = (
            {
                source
                for query, sources in zip(queries, self.allowed_keys, strict=True)
                if query.ndim == 4 and query.shape[2] != 0
                for source in sources
            }
            if skip_empty
            else set(range(3))
        )
        clean_queries, clean_keys, clean_values = [], [], []
        for index, (query, key, value, query_mask, key_mask) in enumerate(
            zip(queries, keys, values, query_masks, key_masks, strict=True)
        ):
            # Keep all projected groups in the query compute dtype when an
            # explicit FP32 geometry boundary meets BF16 autocast.
            if (
                self._prepared_bf16_inference
                and not self.training
                and not torch.is_grad_enabled()
                and torch.is_autocast_enabled("cuda")
                and torch.get_autocast_dtype("cuda") == torch.bfloat16
                and any(tensor.dtype != dtype for tensor in (query, key, value))
            ):
                query, key, value = (tensor.to(dtype=dtype) for tensor in (query, key, value))
            for tensor in (query, key, value):
                _require(
                    tensor.ndim == 4
                    and tensor.shape[:2] == (batch, self.heads)
                    and tensor.shape[-1] == self.head_dimension,
                    "projected attention tokens must have axes [B,H,L,d]",
                )
                _require(
                    tensor.device == device and tensor.dtype == dtype,
                    "projected attention groups must share device and dtype",
                )
            _require(key.shape == value.shape, "attention K/V shapes must agree")
            _check_mask(query_mask, batch=batch, length=int(query.shape[2]), device=device)
            _check_mask(key_mask, batch=batch, length=int(key.shape[2]), device=device)
            clean_queries.append(
                query
                if skip_empty and query.shape[2] == 0
                else query.masked_fill(~query_mask[:, None, :, None], 0.0)
            )
            clean_keys.append(
                key if index not in used_keys else key.masked_fill(~key_mask[:, None, :, None], 0.0)
            )
            clean_values.append(
                value
                if index not in used_keys
                else value.masked_fill(~key_mask[:, None, :, None], 0.0)
            )

        results = []
        for index, (query, query_mask, allowed_keys) in enumerate(
            zip(clean_queries, query_masks, self.allowed_keys, strict=True)
        ):
            if skip_empty and query.shape[2] == 0:
                results.append(query.transpose(1, 2).reshape(batch, 0, self.dimension))
                continue
            key = torch.cat([clean_keys[source] for source in allowed_keys], dim=2)
            value = torch.cat([clean_values[source] for source in allowed_keys], dim=2)
            key_mask = torch.cat([key_masks[source] for source in allowed_keys], dim=1)
            allowed = query_mask[:, :, None] & key_mask[:, None, :]
            results.append(
                self._project_output(self._attention_values(query, key, value, allowed), allowed)
            )
        return tuple(results)
