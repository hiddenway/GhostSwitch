"""Image input for Qwen3.5 on MLX.

mlx-lm loads only the language model of a Qwen3.5 checkpoint. This module loads
the checkpoint's vision tower on the first image, turns each image into model
embeddings, and gives the full-attention layers Qwen's multimodal rotary
positions (t, h, w) while the prompt contains image tokens. LoRA adapters,
sampling and streaming stay on the usual mlx-lm path.
"""

from __future__ import annotations

import contextlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # GGUF-only installations have no MLX.
    mx = nn = None  # type: ignore[assignment]

VISION_START = "<|vision_start|>"
IMAGE_PAD = "<|image_pad|>"
VISION_END = "<|vision_end|>"
IMAGE_PLACEHOLDER = VISION_START + IMAGE_PAD + VISION_END


@dataclass(frozen=True)
class VisionSettings:
    hidden_size: int
    num_heads: int
    depth: int
    intermediate_size: int
    out_hidden_size: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    in_channels: int
    num_position_embeddings: int
    image_token_id: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    min_pixels: int
    max_pixels: int
    mrope_section: tuple[int, int, int]


def has_vision_weights(model_dir: str | Path) -> bool:
    directory = Path(model_dir)
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return "vision_config" in config and any(
        key.startswith("vision_tower.") for key in index.get("weight_map", {}))


def read_settings(model_dir: str | Path) -> VisionSettings:
    directory = Path(model_dir)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    vision = config["vision_config"]
    if vision.get("deepstack_visual_indexes"):
        raise ValueError("Vision checkpoints with deepstack features are not supported")
    processor: dict[str, Any] = {}
    for name in ("preprocessor_config.json", "processor_config.json"):
        path = directory / name
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            processor = {**loaded.get("image_processor", loaded), **processor}
    size = processor.get("size") or {}
    rope = config.get("text_config", {}).get("rope_parameters") or {}
    if rope.get("mrope_interleaved") is False:
        raise ValueError("Only interleaved multimodal RoPE is supported")
    return VisionSettings(
        hidden_size=int(vision["hidden_size"]), num_heads=int(vision["num_heads"]),
        depth=int(vision["depth"]), intermediate_size=int(vision["intermediate_size"]),
        out_hidden_size=int(vision["out_hidden_size"]), patch_size=int(vision["patch_size"]),
        temporal_patch_size=int(vision["temporal_patch_size"]),
        merge_size=int(vision["spatial_merge_size"]), in_channels=int(vision.get("in_channels", 3)),
        num_position_embeddings=int(vision["num_position_embeddings"]),
        image_token_id=int(config["image_token_id"]),
        image_mean=tuple(processor.get("image_mean", (0.5, 0.5, 0.5))),  # type: ignore[arg-type]
        image_std=tuple(processor.get("image_std", (0.5, 0.5, 0.5))),  # type: ignore[arg-type]
        min_pixels=int(size.get("shortest_edge") or processor.get("min_pixels") or 65536),
        max_pixels=int(size.get("longest_edge") or processor.get("max_pixels") or 16777216),
        mrope_section=tuple(rope.get("mrope_section", (11, 11, 10))),  # type: ignore[arg-type]
    )


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen's resize rule: both sides a multiple of factor, area within the pixel limits."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess(image: Any, settings: VisionSettings, max_pixels: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Patches of one RGB PIL image and its (t, h, w) patch grid.

    Each row holds one patch as (temporal, row, column, channel) values, the
    layout of the checkpoint's patch projection.
    """
    from PIL import Image

    patch, merge, temporal = settings.patch_size, settings.merge_size, settings.temporal_patch_size
    width, height = image.size
    resized_h, resized_w = smart_resize(height, width, patch * merge, settings.min_pixels,
                                        min(max_pixels, settings.max_pixels))
    if (resized_w, resized_h) != (width, height):
        image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - np.array(settings.image_mean, np.float32)) / np.array(settings.image_std, np.float32)
    grid_h, grid_w = resized_h // patch, resized_w // patch
    # One still frame fills the temporal patch by repetition.
    frames = np.repeat(pixels[None], temporal, axis=0)  # (T, H, W, C)
    patches = frames.reshape(temporal, grid_h // merge, merge, patch, grid_w // merge, merge, patch,
                             settings.in_channels)
    # Order patches by merge block (block row, block column, row, column), as the merger expects.
    patches = patches.transpose(1, 4, 2, 5, 0, 3, 6, 7)
    return patches.reshape(grid_h * grid_w, -1), (1, grid_h, grid_w)


def mrope_positions(token_ids: list[int], image_token_id: int,
                    grids: list[tuple[int, int, int]]) -> tuple[np.ndarray, int]:
    """Qwen3-VL (t, h, w) positions for a prompt and the offset of later text positions.

    ``grids`` are merged token grids in image order. Text continues after the
    largest position used by the preceding image.
    """
    positions = np.zeros((3, len(token_ids)), dtype=np.int64)
    index = next_position = image = 0
    while index < len(token_ids):
        if token_ids[index] == image_token_id:
            if image >= len(grids):
                raise ValueError("More image tokens than images")
            t, h, w = grids[image]
            count = t * h * w
            if token_ids[index:index + count] != [image_token_id] * count:
                raise ValueError("Image token count does not match the image grid")
            grid = np.stack([np.repeat(np.arange(t), h * w), np.tile(np.repeat(np.arange(h), w), t),
                             np.tile(np.arange(w), t * h)])
            positions[:, index:index + count] = grid + next_position
            next_position = int(positions[:, index:index + count].max()) + 1
            index += count
            image += 1
        else:
            positions[:, index] = next_position
            next_position += 1
            index += 1
    if image != len(grids):
        raise ValueError("Fewer image tokens than images")
    return positions, next_position - len(token_ids)


if nn is not None:
    def _rotate_half(x: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)

    class _PatchProjection(nn.Module):
        """The checkpoint's Conv3d patch embedding, applied as one matrix product."""

        def __init__(self, settings: VisionSettings) -> None:
            super().__init__()
            self.weight = mx.zeros((settings.hidden_size, settings.temporal_patch_size,
                                    settings.patch_size, settings.patch_size, settings.in_channels))
            self.bias = mx.zeros((settings.hidden_size,))

        def __call__(self, patches: mx.array) -> mx.array:
            weight = self.weight.reshape(self.weight.shape[0], -1)
            return patches.astype(weight.dtype) @ weight.T + self.bias

    class _PatchEmbed(nn.Module):
        def __init__(self, settings: VisionSettings) -> None:
            super().__init__()
            self.proj = _PatchProjection(settings)

        def __call__(self, patches: mx.array) -> mx.array:
            return self.proj(patches)

    class _Attention(nn.Module):
        def __init__(self, dim: int, num_heads: int) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.scale = (dim // num_heads) ** -0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
            self.proj = nn.Linear(dim, dim)

        def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
            length = x.shape[0]
            qkv = self.qkv(x).reshape(length, 3, self.num_heads, -1).transpose(1, 2, 0, 3)
            queries, keys, values = qkv[0], qkv[1], qkv[2]  # (heads, length, head_dim)
            queries = (queries * cos + _rotate_half(queries) * sin).astype(values.dtype)
            keys = (keys * cos + _rotate_half(keys) * sin).astype(values.dtype)
            output = mx.fast.scaled_dot_product_attention(
                queries[None], keys[None], values[None], scale=self.scale)[0]
            return self.proj(output.transpose(1, 0, 2).reshape(length, -1))

    class _MLP(nn.Module):
        def __init__(self, dim: int, hidden: int) -> None:
            super().__init__()
            self.linear_fc1 = nn.Linear(dim, hidden, bias=True)
            self.linear_fc2 = nn.Linear(hidden, dim, bias=True)

        def __call__(self, x: mx.array) -> mx.array:
            return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))

    class _Block(nn.Module):
        def __init__(self, settings: VisionSettings) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(settings.hidden_size, eps=1e-6)
            self.norm2 = nn.LayerNorm(settings.hidden_size, eps=1e-6)
            self.attn = _Attention(settings.hidden_size, settings.num_heads)
            self.mlp = _MLP(settings.hidden_size, settings.intermediate_size)

        def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
            x = x + self.attn(self.norm1(x), cos, sin)
            return x + self.mlp(self.norm2(x))

    class _Merger(nn.Module):
        def __init__(self, settings: VisionSettings) -> None:
            super().__init__()
            self.hidden = settings.hidden_size * settings.merge_size ** 2
            self.norm = nn.LayerNorm(settings.hidden_size, eps=1e-6)
            self.linear_fc1 = nn.Linear(self.hidden, self.hidden)
            self.linear_fc2 = nn.Linear(self.hidden, settings.out_hidden_size)

        def __call__(self, x: mx.array) -> mx.array:
            x = self.norm(x).reshape(-1, self.hidden)
            return self.linear_fc2(nn.gelu(self.linear_fc1(x)))

    class VisionTower(nn.Module):
        """Qwen3.5 vision encoder; parameter names follow the checkpoint's vision_tower.*."""

        def __init__(self, settings: VisionSettings) -> None:
            super().__init__()
            self.settings = settings
            self.patch_embed = _PatchEmbed(settings)
            self.pos_embed = nn.Embedding(settings.num_position_embeddings, settings.hidden_size)
            self.blocks = [_Block(settings) for _ in range(settings.depth)]
            self.merger = _Merger(settings)

        def _position_embeddings(self, grid_h: int, grid_w: int) -> mx.array:
            """Bilinear interpolation of the learned square position grid, in merge-block order."""
            side = int(self.settings.num_position_embeddings ** 0.5)
            merge = self.settings.merge_size
            rows = np.linspace(0, side - 1, grid_h, dtype=np.float32)
            cols = np.linspace(0, side - 1, grid_w, dtype=np.float32)
            row0, col0 = rows.astype(np.int32), cols.astype(np.int32)
            row1, col1 = np.minimum(row0 + 1, side - 1), np.minimum(col0 + 1, side - 1)
            dr, dc = rows - row0, cols - col0
            indices = np.stack([
                (row0[:, None] * side + col0[None, :]), (row0[:, None] * side + col1[None, :]),
                (row1[:, None] * side + col0[None, :]), (row1[:, None] * side + col1[None, :]),
            ]).reshape(4, -1)
            weights = np.stack([
                (1 - dr)[:, None] * (1 - dc)[None, :], (1 - dr)[:, None] * dc[None, :],
                dr[:, None] * (1 - dc)[None, :], dr[:, None] * dc[None, :],
            ]).reshape(4, -1)
            embeddings = self.pos_embed(mx.array(indices))
            embeddings = (embeddings * mx.array(weights, dtype=embeddings.dtype)[:, :, None]).sum(axis=0)
            dim = embeddings.shape[-1]
            return (embeddings.reshape(grid_h // merge, merge, grid_w // merge, merge, dim)
                    .transpose(0, 2, 1, 3, 4).reshape(-1, dim))

        def _rotary(self, grid_h: int, grid_w: int) -> tuple[mx.array, mx.array]:
            merge = self.settings.merge_size
            head_dim = self.settings.hidden_size // self.settings.num_heads
            dim = head_dim // 2
            inv_freq = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
            rows = (np.arange(grid_h // merge)[:, None, None, None] * merge
                    + np.arange(merge)[None, None, :, None])
            cols = (np.arange(grid_w // merge)[None, :, None, None] * merge
                    + np.arange(merge)[None, None, None, :])
            shape = (grid_h // merge, grid_w // merge, merge, merge)
            rows, cols = np.broadcast_to(rows, shape).reshape(-1), np.broadcast_to(cols, shape).reshape(-1)
            angles = np.concatenate([np.outer(rows, inv_freq), np.outer(cols, inv_freq)], axis=-1)
            angles = np.concatenate([angles, angles], axis=-1)  # (patches, head_dim)
            return mx.array(np.cos(angles))[None], mx.array(np.sin(angles))[None]

        def __call__(self, patches: mx.array, grid: tuple[int, int, int]) -> mx.array:
            t, grid_h, grid_w = grid
            if t != 1:
                raise ValueError("Only still images are supported")
            x = self.patch_embed(patches)
            x = x + self._position_embeddings(grid_h, grid_w).astype(x.dtype)
            cos, sin = self._rotary(grid_h, grid_w)
            for block in self.blocks:
                x = block(x, cos, sin)
            return self.merger(x)

    class MultimodalRoPE(nn.Module):
        """Interleaved (t, h, w) rotary positions for one prompt that contains images.

        Prompt tokens take their precomputed positions; later tokens continue
        at offset + delta on all three axes, which is ordinary RoPE.
        """

        def __init__(self, base: Any, positions: np.ndarray, delta: int,
                     section: tuple[int, int, int]) -> None:
            super().__init__()
            self.dims = int(base.dims)
            self.base = float(base.base)
            self.scale = float(getattr(base, "scale", 1.0))
            if getattr(base, "traditional", False):
                raise ValueError("Multimodal RoPE expects non-traditional rotary layout")
            half = self.dims // 2
            axis = np.zeros(half, dtype=np.int64)
            axis[1:min(3 * section[1], half):3] = 1
            axis[2:min(3 * section[2], half):3] = 2
            self._axis = axis
            self._inv_freq = 1.0 / (self.base ** (np.arange(0, self.dims, 2, dtype=np.float64) / self.dims))
            self._positions = positions
            self.delta = int(delta)

        def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
            offset = int(offset)
            length = x.shape[-2]
            prompt = self._positions.shape[1]
            if offset >= prompt:
                return mx.fast.rope(x, self.dims, traditional=False, base=self.base,
                                    scale=self.scale, offset=offset + self.delta)
            index = np.arange(offset, offset + length)
            positions = np.where(index[None] < prompt,
                                 self._positions[:, np.minimum(index, prompt - 1)],
                                 (index + self.delta)[None])
            angles = positions[self._axis].T * self.scale * self._inv_freq[None]  # (length, dims/2)
            cos = mx.array(np.cos(angles), dtype=mx.float32)
            sin = mx.array(np.sin(angles), dtype=mx.float32)
            half = self.dims // 2
            first = x[..., :half].astype(mx.float32)
            second = x[..., half:self.dims].astype(mx.float32)
            rotated = mx.concatenate([first * cos - second * sin, second * cos + first * sin], axis=-1)
            return mx.concatenate([rotated.astype(x.dtype), x[..., self.dims:]], axis=-1)


@dataclass
class PreparedPrompt:
    token_ids: list[int]
    embeddings: Any
    positions: np.ndarray
    delta: int
    image_tokens: int


class VisionRuntime:
    """The vision tower of one MLX checkpoint and recently encoded images."""

    def __init__(self, model_dir: str | Path, max_pixels: int) -> None:
        if nn is None:
            raise RuntimeError("MLX image input needs mlx installed")
        self.model_dir = Path(model_dir)
        self.settings = read_settings(self.model_dir)
        self.max_pixels = max_pixels
        self.tower: Any = None
        self._features: OrderedDict[str, tuple[Any, tuple[int, int, int]]] = OrderedDict()

    def load(self) -> None:
        if self.tower is not None:
            return
        index = json.loads((self.model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = sorted({shard for key, shard in index["weight_map"].items() if key.startswith("vision_tower.")})
        if not shards:
            raise ValueError("This checkpoint has no vision weights")
        weights: dict[str, Any] = {}
        for shard in shards:
            loaded = mx.load(str(self.model_dir / shard))
            weights.update({key.removeprefix("vision_tower."): value for key, value in loaded.items()
                            if key.startswith("vision_tower.")})
            del loaded
        tower = VisionTower(self.settings)
        tower.load_weights(list(weights.items()), strict=True)
        tower.eval()
        mx.eval(tower.parameters())
        self.tower = tower

    def encode(self, image: Any) -> tuple[Any, tuple[int, int, int]]:
        """Image embeddings in language-model space and the merged token grid."""
        cached = self._features.get(image.digest)
        if cached is not None:
            self._features.move_to_end(image.digest)
            return cached
        self.load()
        patches, grid = preprocess(image.image, self.settings, self.max_pixels)
        features = self.tower(mx.array(patches), grid)
        mx.eval(features)
        merge = self.settings.merge_size
        result = (features, (grid[0], grid[1] // merge, grid[2] // merge))
        self._features[image.digest] = result
        while len(self._features) > 8:
            self._features.popitem(last=False)
        return result

    def prepare(self, model: Any, token_ids: list[int], images: list[Any]) -> PreparedPrompt:
        """Expand each image token to its grid, embed the prompt, and compute positions."""
        image_token = self.settings.image_token_id
        if token_ids.count(image_token) != len(images):
            raise ValueError("The rendered prompt does not have one image slot per image")
        encoded = [self.encode(image) for image in images]
        expanded: list[int] = []
        image_index = 0
        for token in token_ids:
            if token == image_token:
                t, h, w = encoded[image_index][1]
                expanded.extend([image_token] * (t * h * w))
                image_index += 1
            else:
                expanded.append(token)
        embed_tokens = model.language_model.model.embed_tokens
        embeddings = embed_tokens(mx.array(expanded, dtype=mx.uint32))
        slots = np.flatnonzero(np.array(expanded) == image_token)
        features = mx.concatenate([item[0] for item in encoded], axis=0).astype(embeddings.dtype)
        embeddings[mx.array(slots, dtype=mx.uint32)] = features
        positions, delta = mrope_positions(expanded, image_token, [item[1] for item in encoded])
        mx.eval(embeddings)
        return PreparedPrompt(expanded, embeddings, positions, delta, len(slots))

    @contextlib.contextmanager
    def positions(self, model: Any, prepared: PreparedPrompt) -> Iterator[None]:
        """Use multimodal rotary positions in every full-attention layer for one generation."""
        replaced = []
        try:
            for layer in model.language_model.model.layers:
                attention = getattr(layer, "self_attn", None)
                if attention is None or not hasattr(attention, "rope"):
                    continue
                original = attention.rope
                attention.rope = MultimodalRoPE(original, prepared.positions, prepared.delta,
                                                self.settings.mrope_section)
                replaced.append((attention, original))
            yield
        finally:
            for attention, original in replaced:
                attention.rope = original
