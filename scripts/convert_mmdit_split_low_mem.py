#!/usr/bin/env python3
"""
Split SD3 MMDiT into Core ML stages for lower peak memory on iOS.

This converter follows the Draw Things-style runtime shape more closely than
the monolithic Core ML export:

  * Conditioning is split out into its own model:
      pooled_text_embeddings + timestep -> modulation_inputs
  * Adaptive LayerNorm is fused back into each MMDiT stage. Each stage receives
    modulation_inputs directly, computes only its own AdaLN tensors internally,
    and passes image/text token activations forward.

The Swift runtime precomputes timestep conditioning once per denoise step, then
runs the fused body stages serially.
"""

from __future__ import annotations

import argparse
import gc
import logging
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("convert_mmdit_split_low_mem")


def adaln_names_for_block(block_index: int, text_param_count: int) -> list[str]:
    names = [
        f"adaln_b{block_index:02d}_image_shift1",
        f"adaln_b{block_index:02d}_image_residual_scale1",
        f"adaln_b{block_index:02d}_image_post_attn_scale",
        f"adaln_b{block_index:02d}_image_shift2",
        f"adaln_b{block_index:02d}_image_residual_scale2",
        f"adaln_b{block_index:02d}_image_post_mlp_scale",
        f"adaln_b{block_index:02d}_text_shift1",
        f"adaln_b{block_index:02d}_text_residual_scale1",
    ]
    if text_param_count > 2:
        names += [
            f"adaln_b{block_index:02d}_text_post_attn_scale",
            f"adaln_b{block_index:02d}_text_shift2",
            f"adaln_b{block_index:02d}_text_residual_scale2",
            f"adaln_b{block_index:02d}_text_post_mlp_scale",
        ]
    return names


def all_adaln_names(model) -> list[str]:
    names: list[str] = []
    for idx, block in enumerate(model.multimodal_transformer_blocks):
        names += adaln_names_for_block(
            idx, block.text_transformer_block.num_modulation_params
        )
    names += ["adaln_final_shift", "adaln_final_residual_scale"]
    return names


def adaln_names_for_range(model, start_block: int, end_block: int) -> list[str]:
    names: list[str] = []
    for block_idx in range(start_block, end_block):
        block = model.multimodal_transformer_blocks[block_idx]
        names += adaln_names_for_block(
            block_idx, block.text_transformer_block.num_modulation_params
        )
    if end_block == model.config.depth:
        names += ["adaln_final_shift", "adaln_final_residual_scale"]
    return names


def affine_transform(x, shift, residual_scale):
    return x * (1.0 + residual_scale) + shift


def unpatchify(x, patch_size: int, target_height: int, target_width: int, vae_latent_dim: int):
    h, w = target_height // patch_size, target_width // patch_size
    x = x.transpose(1, 3).view(x.shape[0], h, w, patch_size, patch_size, vae_latent_dim)
    x = torch.einsum("bhwpqc->bchpwq", x)
    return x.reshape(x.shape[0], vae_latent_dim, target_height, target_width)


class MMDiTConditioning(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.y_embedder = model.y_embedder
        self.t_embedder = model.t_embedder

    def forward(self, pooled_text_embeddings, timestep):
        return (self.y_embedder(pooled_text_embeddings) + self.t_embedder(timestep),)


class MMDiTAdaptiveLayerNorm(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.blocks = model.multimodal_transformer_blocks
        self.final_layer = model.final_layer

    def forward(self, modulation_inputs):
        outputs = []
        for block in self.blocks:
            image_params = block.image_transformer_block.adaLN_modulation(
                modulation_inputs
            ).chunk(block.image_transformer_block.num_modulation_params, dim=1)
            text_params = block.text_transformer_block.adaLN_modulation(
                modulation_inputs
            ).chunk(block.text_transformer_block.num_modulation_params, dim=1)
            outputs.extend(image_params)
            outputs.extend(text_params)

        final_params = self.final_layer.adaLN_modulation(modulation_inputs).chunk(2, dim=1)
        outputs.extend(final_params)
        return tuple(outputs)


class MMDiTAdaptiveLayerNormStage(nn.Module):
    def __init__(self, model, start_block: int, end_block: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [model.multimodal_transformer_blocks[i] for i in range(start_block, end_block)]
        )
        self.final_layer = model.final_layer if end_block == model.config.depth else None

    def forward(self, modulation_inputs):
        outputs = []
        for block in self.blocks:
            image_params = block.image_transformer_block.adaLN_modulation(
                modulation_inputs
            ).chunk(block.image_transformer_block.num_modulation_params, dim=1)
            text_params = block.text_transformer_block.adaLN_modulation(
                modulation_inputs
            ).chunk(block.text_transformer_block.num_modulation_params, dim=1)
            outputs.extend(image_params)
            outputs.extend(text_params)

        if self.final_layer is not None:
            final_params = self.final_layer.adaLN_modulation(modulation_inputs).chunk(2, dim=1)
            outputs.extend(final_params)
        return tuple(outputs)


class MMDiTInputEmbeddingStage(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.config = model.config
        self.x_embedder = model.x_embedder
        self.x_pos_embedder = model.x_pos_embedder
        self.context_embedder = model.context_embedder

    def forward(self, latent_image_embeddings, token_level_text_embeddings, modulation_inputs):
        batch = latent_image_embeddings.shape[0]
        latent_image_embeddings = self.x_embedder(
            latent_image_embeddings
        ) + self.x_pos_embedder(latent_image_embeddings)
        latent_image_embeddings = latent_image_embeddings.view(
            batch, self.config.hidden_size, 1, -1
        )
        token_level_text_embeddings = self.context_embedder(token_level_text_embeddings)
        return latent_image_embeddings, token_level_text_embeddings


class MMDiTStage(nn.Module):
    def __init__(
        self,
        model,
        start_block: int,
        end_block: int,
        latent_h: int,
        latent_w: int,
        include_input_embedding: bool = True,
        sdpa_mode: str = "cat",
        splitkv_chunks: int = 10,
    ):
        super().__init__()
        self.config = model.config
        self.start_block = start_block
        self.end_block = end_block
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.is_first = include_input_embedding and start_block == 0
        self.is_final = end_block == model.config.depth
        self.sdpa_mode = sdpa_mode
        self.splitkv_chunks = splitkv_chunks

        if self.is_first:
            self.x_embedder = model.x_embedder
            self.x_pos_embedder = model.x_pos_embedder
            self.context_embedder = model.context_embedder

        self.blocks = nn.ModuleList(
            [model.multimodal_transformer_blocks[i] for i in range(start_block, end_block)]
        )

        if self.is_final:
            self.final_layer = model.final_layer

    @staticmethod
    def _splitkv_sdpa(query, key, value, n_heads: int, num_chunks: int):
        batch = query.shape[0]
        embed_dim = query.shape[1]
        per_head_dim = embed_dim // n_heads
        key_seq_len = key.shape[3]
        if key_seq_len % num_chunks != 0:
            raise ValueError(
                f"SplitKV requires key sequence {key_seq_len} to divide by {num_chunks}"
            )

        chunk_size = key_seq_len // num_chunks
        head_shape = (batch, n_heads, per_head_dim, -1)
        mh_q = query.view(*head_shape) * (per_head_dim ** -0.5)
        mh_k_chunks = key.view(*head_shape).split(chunk_size, dim=3)
        mh_v_chunks = value.view(*head_shape).split(chunk_size, dim=3)

        local_maxes = []
        local_sums = []
        local_weighted_values = []
        for k_chunk, v_chunk in zip(mh_k_chunks, mh_v_chunks):
            weights = torch.einsum("bhcq,bhck->bhqk", mh_q, k_chunk)
            weights_max = weights.max(dim=3, keepdim=True)[0]
            weights_exp = (weights - weights_max).exp()
            local_maxes.append(weights_max)
            local_sums.append(weights_exp.sum(dim=3, keepdim=True))
            local_weighted_values.append(
                torch.einsum("bhqk,bhck->bhcq", weights_exp, v_chunk)
            )

        global_max = torch.cat(local_maxes, dim=3).max(dim=3, keepdim=True)[0]
        value_accumulator = None
        sum_accumulator = None
        for weights_max, weights_sum, weighted_value in zip(
            local_maxes, local_sums, local_weighted_values
        ):
            correction = (weights_max - global_max).exp()
            corrected_value = weighted_value * correction.transpose(2, 3)
            corrected_sum = weights_sum * correction
            value_accumulator = (
                corrected_value
                if value_accumulator is None
                else value_accumulator + corrected_value
            )
            sum_accumulator = (
                corrected_sum if sum_accumulator is None else sum_accumulator + corrected_sum
            )

        attn = value_accumulator / sum_accumulator.transpose(2, 3)
        return attn.reshape(batch, embed_dim, 1, -1)

    @staticmethod
    def _block_forward(
        block,
        latent_image_embeddings,
        token_level_text_embeddings,
        image_params,
        text_params,
        sdpa_mode: str,
        splitkv_chunks: int,
        n_heads: int,
    ):
        image_shift1, image_residual_scale1 = image_params[0], image_params[1]
        image_pre_attn = affine_transform(
            block.image_transformer_block.norm1(latent_image_embeddings),
            shift=image_shift1,
            residual_scale=image_residual_scale1,
        )
        image_q = block.image_transformer_block.attn.q_proj(image_pre_attn)
        image_k = block.image_transformer_block.attn.k_proj(image_pre_attn)
        image_v = block.image_transformer_block.attn.v_proj(image_pre_attn)

        text_shift1, text_residual_scale1 = text_params[0], text_params[1]
        text_pre_attn = affine_transform(
            block.text_transformer_block.norm1(token_level_text_embeddings),
            shift=text_shift1,
            residual_scale=text_residual_scale1,
        )
        text_q = block.text_transformer_block.attn.q_proj(text_pre_attn)
        text_k = block.text_transformer_block.attn.k_proj(text_pre_attn)
        text_v = block.text_transformer_block.attn.v_proj(text_pre_attn)

        sdpa_query = torch.cat([image_q, text_q], dim=-1)
        sdpa_key = torch.cat([image_k, text_k], dim=-1)
        sdpa_value = torch.cat([image_v, text_v], dim=-1)
        if sdpa_mode == "splitkv":
            sdpa_outputs = MMDiTStage._splitkv_sdpa(
                sdpa_query,
                sdpa_key,
                sdpa_value,
                n_heads=n_heads,
                num_chunks=splitkv_chunks,
            )
        else:
            sdpa_outputs = block.sdpa(
                query=sdpa_query,
                key=sdpa_key,
                value=sdpa_value,
            )

        img_seq_len = latent_image_embeddings.shape[-1]
        txt_seq_len = token_level_text_embeddings.shape[-1]
        image_sdpa_output = sdpa_outputs[:, :, :, :img_seq_len]
        text_sdpa_output = sdpa_outputs[:, :, :, -txt_seq_len:]

        latent_image_embeddings = block.image_transformer_block.post_sdpa(
            residual=latent_image_embeddings,
            sdpa_output=image_sdpa_output,
            post_attn_scale=image_params[2],
            post_norm2_shift=image_params[3],
            post_norm2_residual_scale=image_params[4],
            post_mlp_scale=image_params[5],
        )

        if block.text_transformer_block.skip_post_sdpa:
            token_level_text_embeddings = token_level_text_embeddings
        else:
            token_level_text_embeddings = block.text_transformer_block.post_sdpa(
                residual=token_level_text_embeddings,
                sdpa_output=text_sdpa_output,
                post_attn_scale=text_params[2],
                post_norm2_shift=text_params[3],
                post_norm2_residual_scale=text_params[4],
                post_mlp_scale=text_params[5],
            )

        return latent_image_embeddings, token_level_text_embeddings

    def forward(self, latent_image_embeddings, token_level_text_embeddings, modulation_inputs):
        batch = latent_image_embeddings.shape[0]
        if self.is_first:
            latent_image_embeddings = self.x_embedder(
                latent_image_embeddings
            ) + self.x_pos_embedder(latent_image_embeddings)
            latent_image_embeddings = latent_image_embeddings.view(
                batch, self.config.hidden_size, 1, -1
            )
            token_level_text_embeddings = self.context_embedder(token_level_text_embeddings)

        for block in self.blocks:
            image_params = block.image_transformer_block.adaLN_modulation(
                modulation_inputs
            ).chunk(block.image_transformer_block.num_modulation_params, dim=1)
            text_params = block.text_transformer_block.adaLN_modulation(
                modulation_inputs
            ).chunk(block.text_transformer_block.num_modulation_params, dim=1)
            latent_image_embeddings, token_level_text_embeddings = self._block_forward(
                block,
                latent_image_embeddings,
                token_level_text_embeddings,
                image_params,
                text_params,
                self.sdpa_mode,
                self.splitkv_chunks,
                self.config.depth,
            )

        if not self.is_final:
            return latent_image_embeddings, token_level_text_embeddings

        final_shift, final_residual_scale = self.final_layer.adaLN_modulation(
            modulation_inputs
        ).chunk(2, dim=1)
        latent_image_embeddings = affine_transform(
            self.final_layer.norm_final(latent_image_embeddings),
            shift=final_shift,
            residual_scale=final_residual_scale,
        )
        latent_image_embeddings = self.final_layer.linear(latent_image_embeddings)
        latent_image_embeddings = unpatchify(
            latent_image_embeddings,
            patch_size=self.config.patch_size,
            target_height=self.latent_h,
            target_width=self.latent_w,
            vae_latent_dim=self.config.vae_latent_dim,
        )
        return (latent_image_embeddings,)


class MMDiTBlockPreStage(nn.Module):
    def __init__(self, model, block_index: int):
        super().__init__()
        self.block = model.multimodal_transformer_blocks[block_index]

    def forward(self, latent_image_embeddings, token_level_text_embeddings, modulation_inputs):
        block = self.block
        image_params = block.image_transformer_block.adaLN_modulation(
            modulation_inputs
        ).chunk(block.image_transformer_block.num_modulation_params, dim=1)
        text_params = block.text_transformer_block.adaLN_modulation(
            modulation_inputs
        ).chunk(block.text_transformer_block.num_modulation_params, dim=1)

        image_pre_attn = affine_transform(
            block.image_transformer_block.norm1(latent_image_embeddings),
            shift=image_params[0],
            residual_scale=image_params[1],
        )
        text_pre_attn = affine_transform(
            block.text_transformer_block.norm1(token_level_text_embeddings),
            shift=text_params[0],
            residual_scale=text_params[1],
        )

        outputs = [
            block.image_transformer_block.attn.q_proj(image_pre_attn),
            block.image_transformer_block.attn.k_proj(image_pre_attn),
            block.image_transformer_block.attn.v_proj(image_pre_attn),
            block.text_transformer_block.attn.q_proj(text_pre_attn),
            block.text_transformer_block.attn.k_proj(text_pre_attn),
            block.text_transformer_block.attn.v_proj(text_pre_attn),
            image_params[2],
            image_params[3],
            image_params[4],
            image_params[5],
        ]

        if not block.text_transformer_block.skip_post_sdpa:
            outputs.extend([text_params[2], text_params[3], text_params[4], text_params[5]])

        return tuple(outputs)


class MMDiTBlockSDPAStage(nn.Module):
    def __init__(self, model, sdpa_mode: str = "cat", splitkv_chunks: int = 10):
        super().__init__()
        self.config = model.config
        self.sdpa_mode = sdpa_mode
        self.splitkv_chunks = splitkv_chunks
        self.sdpa = model.multimodal_transformer_blocks[0].sdpa

    def forward(self, image_q, image_k, image_v, text_q, text_k, text_v):
        sdpa_query = torch.cat([image_q, text_q], dim=-1)
        sdpa_key = torch.cat([image_k, text_k], dim=-1)
        sdpa_value = torch.cat([image_v, text_v], dim=-1)
        if self.sdpa_mode == "splitkv":
            sdpa_outputs = MMDiTStage._splitkv_sdpa(
                sdpa_query,
                sdpa_key,
                sdpa_value,
                n_heads=self.config.depth,
                num_chunks=self.splitkv_chunks,
            )
        else:
            sdpa_outputs = self.sdpa(
                query=sdpa_query,
                key=sdpa_key,
                value=sdpa_value,
            )

        img_seq_len = image_q.shape[-1]
        txt_seq_len = text_q.shape[-1]
        image_sdpa_output = sdpa_outputs[:, :, :, :img_seq_len]
        text_sdpa_output = sdpa_outputs[:, :, :, -txt_seq_len:]
        return image_sdpa_output, text_sdpa_output


class MMDiTBlockSDPAImageQueryChunkStage(nn.Module):
    def __init__(
        self,
        model,
        chunk_index: int,
        num_chunks: int,
        image_seq_len: int,
        sdpa_mode: str = "cat",
        splitkv_chunks: int = 10,
    ):
        super().__init__()
        if image_seq_len % num_chunks != 0:
            raise ValueError(
                f"image sequence {image_seq_len} must divide by {num_chunks}"
            )
        self.config = model.config
        self.chunk_index = chunk_index
        self.chunk_size = image_seq_len // num_chunks
        self.sdpa_mode = sdpa_mode
        self.splitkv_chunks = splitkv_chunks
        self.sdpa = model.multimodal_transformer_blocks[0].sdpa

    def forward(self, image_q, image_k, image_v, text_k, text_v):
        start = self.chunk_index * self.chunk_size
        end = start + self.chunk_size
        sdpa_query = image_q[:, :, :, start:end]
        sdpa_key = torch.cat([image_k, text_k], dim=-1)
        sdpa_value = torch.cat([image_v, text_v], dim=-1)
        if self.sdpa_mode == "splitkv":
            sdpa_outputs = MMDiTStage._splitkv_sdpa(
                sdpa_query,
                sdpa_key,
                sdpa_value,
                n_heads=self.config.depth,
                num_chunks=self.splitkv_chunks,
            )
        else:
            sdpa_outputs = self.sdpa(
                query=sdpa_query,
                key=sdpa_key,
                value=sdpa_value,
            )
        return (sdpa_outputs,)


class MMDiTBlockSDPATextQueryStage(nn.Module):
    def __init__(self, model, sdpa_mode: str = "cat", splitkv_chunks: int = 10):
        super().__init__()
        self.config = model.config
        self.sdpa_mode = sdpa_mode
        self.splitkv_chunks = splitkv_chunks
        self.sdpa = model.multimodal_transformer_blocks[0].sdpa

    def forward(self, image_k, image_v, text_q, text_k, text_v):
        sdpa_key = torch.cat([image_k, text_k], dim=-1)
        sdpa_value = torch.cat([image_v, text_v], dim=-1)
        if self.sdpa_mode == "splitkv":
            sdpa_outputs = MMDiTStage._splitkv_sdpa(
                text_q,
                sdpa_key,
                sdpa_value,
                n_heads=self.config.depth,
                num_chunks=self.splitkv_chunks,
            )
        else:
            sdpa_outputs = self.sdpa(
                query=text_q,
                key=sdpa_key,
                value=sdpa_value,
            )
        return (sdpa_outputs,)


class MMDiTBlockSDPACombineStage(nn.Module):
    def __init__(self, num_image_chunks: int):
        super().__init__()
        self.num_image_chunks = num_image_chunks

    def forward(self, *inputs):
        image_chunks = inputs[: self.num_image_chunks]
        text_sdpa_output = inputs[self.num_image_chunks]
        image_sdpa_output = torch.cat(image_chunks, dim=-1)
        return image_sdpa_output, text_sdpa_output


class MMDiTBlockPostStage(nn.Module):
    def __init__(self, model, block_index: int, latent_h: int, latent_w: int):
        super().__init__()
        self.config = model.config
        self.block = model.multimodal_transformer_blocks[block_index]
        self.final_layer = model.final_layer if block_index == model.config.depth - 1 else None
        self.latent_h = latent_h
        self.latent_w = latent_w

    def forward(
        self,
        latent_image_embeddings,
        token_level_text_embeddings,
        image_sdpa_output,
        text_sdpa_output,
        image_post_attn_scale,
        image_post_norm2_shift,
        image_post_norm2_residual_scale,
        image_post_mlp_scale,
        text_post_attn_scale=None,
        text_post_norm2_shift=None,
        text_post_norm2_residual_scale=None,
        text_post_mlp_scale=None,
    ):
        block = self.block
        latent_image_embeddings = block.image_transformer_block.post_sdpa(
            residual=latent_image_embeddings,
            sdpa_output=image_sdpa_output,
            post_attn_scale=image_post_attn_scale,
            post_norm2_shift=image_post_norm2_shift,
            post_norm2_residual_scale=image_post_norm2_residual_scale,
            post_mlp_scale=image_post_mlp_scale,
        )

        if not block.text_transformer_block.skip_post_sdpa:
            token_level_text_embeddings = block.text_transformer_block.post_sdpa(
                residual=token_level_text_embeddings,
                sdpa_output=text_sdpa_output,
                post_attn_scale=text_post_attn_scale,
                post_norm2_shift=text_post_norm2_shift,
                post_norm2_residual_scale=text_post_norm2_residual_scale,
                post_mlp_scale=text_post_mlp_scale,
            )

        if self.final_layer is None:
            return latent_image_embeddings, token_level_text_embeddings

        final_shift, final_residual_scale = self.final_layer.adaLN_modulation(
            torch.zeros_like(image_post_attn_scale)
        ).chunk(2, dim=1)
        raise RuntimeError("final layer requires modulation_inputs")


class MMDiTBlockFinalPostStage(nn.Module):
    def __init__(self, model, block_index: int, latent_h: int, latent_w: int):
        super().__init__()
        self.config = model.config
        self.block = model.multimodal_transformer_blocks[block_index]
        self.final_layer = model.final_layer
        self.latent_h = latent_h
        self.latent_w = latent_w

    def forward(
        self,
        latent_image_embeddings,
        token_level_text_embeddings,
        modulation_inputs,
        image_sdpa_output,
        text_sdpa_output,
        image_post_attn_scale,
        image_post_norm2_shift,
        image_post_norm2_residual_scale,
        image_post_mlp_scale,
    ):
        block = self.block
        latent_image_embeddings = block.image_transformer_block.post_sdpa(
            residual=latent_image_embeddings,
            sdpa_output=image_sdpa_output,
            post_attn_scale=image_post_attn_scale,
            post_norm2_shift=image_post_norm2_shift,
            post_norm2_residual_scale=image_post_norm2_residual_scale,
            post_mlp_scale=image_post_mlp_scale,
        )

        final_shift, final_residual_scale = self.final_layer.adaLN_modulation(
            modulation_inputs
        ).chunk(2, dim=1)
        latent_image_embeddings = affine_transform(
            self.final_layer.norm_final(latent_image_embeddings),
            shift=final_shift,
            residual_scale=final_residual_scale,
        )
        latent_image_embeddings = self.final_layer.linear(latent_image_embeddings)
        latent_image_embeddings = unpatchify(
            latent_image_embeddings,
            patch_size=self.config.patch_size,
            target_height=self.latent_h,
            target_width=self.latent_w,
            vae_latent_dim=self.config.vae_latent_dim,
        )
        return (latent_image_embeddings,)


def parse_stage_sizes(raw: str, depth: int) -> list[tuple[int, int]]:
    sizes = [int(part) for part in raw.split(",") if part.strip()]
    if len(sizes) < 2:
        raise ValueError("--stage-sizes must contain at least two stages")
    if len(sizes) > depth:
        raise ValueError(f"--stage-sizes supports at most {depth} stages")
    if sum(sizes) != depth:
        raise ValueError(f"--stage-sizes sum to {sum(sizes)}, expected {depth}")
    ranges = []
    start = 0
    for size in sizes:
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def save_mlpackage(mlmodel, out_path: Path):
    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))


def cleanup_stale_split_packages(output_dir: Path):
    for pattern in [
        "MultiModalDiffusionTransformerAdaLN*.mlpackage",
        "MultiModalDiffusionTransformerStage*.mlpackage",
    ]:
        for candidate in output_dir.glob(pattern):
            logger.info("Removing stale split package: %s", candidate.name)
            shutil.rmtree(candidate)


def convert_module(module, inputs: dict[str, torch.Tensor], output_names: list[str], out_path: Path, target):
    import coremltools as ct

    logger.info("Tracing %s", out_path.name)
    traced = torch.jit.trace(module.eval(), example_inputs=tuple(inputs.values()), strict=False)
    coreml_inputs = [
        ct.TensorType(name=name, shape=tensor.shape, dtype=tensor.numpy().dtype)
        for name, tensor in inputs.items()
    ]
    outputs = [ct.TensorType(name=name) for name in output_names]
    logger.info("Converting %s", out_path.name)
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=coreml_inputs,
        outputs=outputs,
        minimum_deployment_target=target,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
        convert_to="mlprogram",
    )
    logger.info("Converted %s in %.1fs", out_path.name, time.time() - t0)
    save_mlpackage(mlmodel, out_path)
    del traced, mlmodel
    gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", default=None, help="Local sd3_medium*.safetensors path")
    parser.add_argument("--sd3-repo", default="stabilityai/stable-diffusion-3-medium")
    parser.add_argument("--ckpt-file", default="sd3_medium.safetensors")
    parser.add_argument("--latent-h", type=int, default=64)
    parser.add_argument("--latent-w", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1, choices=(1, 2))
    parser.add_argument(
        "--stage-sizes",
        default="4,4,4,4,4,4",
        help="Comma-separated block counts, sum must be 24. Default is 6 fused stages.",
    )
    parser.add_argument("--ios-target", choices=("iOS17", "iOS18"), default="iOS18")
    parser.add_argument(
        "--split-input-embedding",
        action="store_true",
        help=(
            "Emit Stage0 as only x/context embedding, then start block stages at "
            "Stage1. This lowers the ANE compile pressure of the first 1024 stage."
        ),
    )
    parser.add_argument(
        "--sdpa-mode",
        choices=("cat", "splitkv"),
        default="cat",
        help=(
            "Attention implementation inside each block. splitkv keeps full attention "
            "semantics while lowering ANE compile/live tensor pressure."
        ),
    )
    parser.add_argument(
        "--splitkv-chunks",
        type=int,
        default=10,
        help="Number of key/value chunks for --sdpa-mode splitkv.",
    )
    parser.add_argument(
        "--block-micro-stages",
        action="store_true",
        help=(
            "Emit each transformer block as pre/QKV, SDPA, and post/MLP "
            "micro-stages. This lowers the ANE compile pressure for 1024."
        ),
    )
    parser.add_argument(
        "--sdpa-query-chunks",
        type=int,
        default=1,
        help=(
            "When using --block-micro-stages, split image attention query tokens "
            "into this many SDPA stages before concatenating them back together."
        ),
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("sd3_build_split"))
    args = parser.parse_args()

    import coremltools as ct
    from huggingface_hub import hf_hub_download
    from diffusionkit.torch import mmdit
    from diffusionkit.torch.model_io import _load_mmdit_weights

    torch.set_grad_enabled(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_split_packages(args.output_dir)

    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        ckpt_path = hf_hub_download(args.sd3_repo, args.ckpt_file)
    elif not Path(ckpt_path).exists():
        raise FileNotFoundError(ckpt_path)

    cfg = mmdit.SD3_2b
    if args.latent_h > cfg.max_latent_resolution or args.latent_w > cfg.max_latent_resolution:
        raise ValueError("latent size exceeds model positional embedding limit")
    stage_ranges = parse_stage_sizes(args.stage_sizes, cfg.depth)
    deployment_target = ct.target.iOS18 if args.ios_target == "iOS18" else ct.target.iOS17

    logger.info("Building MMDiT and loading weights")
    model = mmdit.MMDiT(cfg).to("cpu").to(torch.float32).eval()
    _load_mmdit_weights(model, ckpt_path)

    batch = args.batch_size
    img_seq_len = (args.latent_h // cfg.patch_size) * (args.latent_w // cfg.patch_size)

    conditioning_inputs = {
        "pooled_text_embeddings": torch.randn(batch, cfg.pooled_text_embed_dim, 1, 1),
        "timestep": torch.randn(batch),
    }
    convert_module(
        MMDiTConditioning(model),
        conditioning_inputs,
        ["modulation_inputs"],
        args.output_dir / "MultiModalDiffusionTransformerConditioning.mlpackage",
        deployment_target,
    )

    stage_index_offset = 0
    if args.split_input_embedding:
        stage_inputs = {
            "latent_image_embeddings": torch.randn(
                batch, cfg.vae_latent_dim, args.latent_h, args.latent_w
            ),
            "token_level_text_embeddings": torch.randn(
                batch, cfg.token_level_text_embed_dim, 1, cfg.text_seq_len
            ),
            "modulation_inputs": torch.randn(batch, cfg.hidden_size, 1, 1),
        }
        convert_module(
            MMDiTInputEmbeddingStage(model),
            stage_inputs,
            ["latent_image_embeddings_out", "token_level_text_embeddings_out"],
            args.output_dir / "MultiModalDiffusionTransformerStage0.mlpackage",
            deployment_target,
        )
        stage_index_offset = 1

    if args.block_micro_stages:
        if not args.split_input_embedding:
            raise ValueError("--block-micro-stages requires --split-input-embedding")
        if args.sdpa_query_chunks < 1:
            raise ValueError("--sdpa-query-chunks must be >= 1")
        if img_seq_len % args.sdpa_query_chunks != 0:
            raise ValueError(
                f"image sequence {img_seq_len} must divide by --sdpa-query-chunks "
                f"{args.sdpa_query_chunks}"
            )

        stage_index = stage_index_offset
        hidden_image = torch.randn(batch, cfg.hidden_size, 1, img_seq_len)
        hidden_text = torch.randn(batch, cfg.hidden_size, 1, cfg.text_seq_len)
        modulation = torch.randn(batch, cfg.hidden_size, 1, 1)
        image_sdpa = torch.randn(batch, cfg.hidden_size, 1, img_seq_len)
        text_sdpa = torch.randn(batch, cfg.hidden_size, 1, cfg.text_seq_len)
        param = torch.randn(batch, cfg.hidden_size, 1, 1)

        pre_output_names = [
            "image_q",
            "image_k",
            "image_v",
            "text_q",
            "text_k",
            "text_v",
            "image_post_attn_scale",
            "image_post_norm2_shift",
            "image_post_norm2_residual_scale",
            "image_post_mlp_scale",
        ]
        text_post_names = [
            "text_post_attn_scale",
            "text_post_norm2_shift",
            "text_post_norm2_residual_scale",
            "text_post_mlp_scale",
        ]

        for block_index in range(cfg.depth):
            block = model.multimodal_transformer_blocks[block_index]
            output_names = list(pre_output_names)
            if not block.text_transformer_block.skip_post_sdpa:
                output_names += text_post_names
            convert_module(
                MMDiTBlockPreStage(model, block_index),
                {
                    "latent_image_embeddings": hidden_image,
                    "token_level_text_embeddings": hidden_text,
                    "modulation_inputs": modulation,
                },
                output_names,
                args.output_dir / f"MultiModalDiffusionTransformerStage{stage_index}.mlpackage",
                deployment_target,
            )
            stage_index += 1

            if args.sdpa_query_chunks == 1:
                convert_module(
                    MMDiTBlockSDPAStage(
                        model,
                        sdpa_mode=args.sdpa_mode,
                        splitkv_chunks=args.splitkv_chunks,
                    ),
                    {
                        "image_q": hidden_image,
                        "image_k": hidden_image,
                        "image_v": hidden_image,
                        "text_q": hidden_text,
                        "text_k": hidden_text,
                        "text_v": hidden_text,
                    },
                    ["image_sdpa_output", "text_sdpa_output"],
                    args.output_dir / f"MultiModalDiffusionTransformerStage{stage_index}.mlpackage",
                    deployment_target,
                )
                stage_index += 1
            else:
                image_chunk_len = img_seq_len // args.sdpa_query_chunks
                image_chunk_outputs = []
                for chunk_index in range(args.sdpa_query_chunks):
                    output_name = f"image_sdpa_output_{chunk_index:02d}"
                    image_chunk_outputs.append(output_name)
                    convert_module(
                        MMDiTBlockSDPAImageQueryChunkStage(
                            model,
                            chunk_index=chunk_index,
                            num_chunks=args.sdpa_query_chunks,
                            image_seq_len=img_seq_len,
                            sdpa_mode=args.sdpa_mode,
                            splitkv_chunks=args.splitkv_chunks,
                        ),
                        {
                            "image_q": hidden_image,
                            "image_k": hidden_image,
                            "image_v": hidden_image,
                            "text_k": hidden_text,
                            "text_v": hidden_text,
                        },
                        [output_name],
                        args.output_dir / f"MultiModalDiffusionTransformerStage{stage_index}.mlpackage",
                        deployment_target,
                    )
                    stage_index += 1

                convert_module(
                    MMDiTBlockSDPATextQueryStage(
                        model,
                        sdpa_mode=args.sdpa_mode,
                        splitkv_chunks=args.splitkv_chunks,
                    ),
                    {
                        "image_k": hidden_image,
                        "image_v": hidden_image,
                        "text_q": hidden_text,
                        "text_k": hidden_text,
                        "text_v": hidden_text,
                    },
                    ["text_sdpa_output"],
                    args.output_dir / f"MultiModalDiffusionTransformerStage{stage_index}.mlpackage",
                    deployment_target,
                )
                stage_index += 1

                combine_inputs = {
                    name: torch.randn(batch, cfg.hidden_size, 1, image_chunk_len)
                    for name in image_chunk_outputs
                }
                combine_inputs["text_sdpa_output"] = text_sdpa
                convert_module(
                    MMDiTBlockSDPACombineStage(args.sdpa_query_chunks),
                    combine_inputs,
                    ["image_sdpa_output", "text_sdpa_output"],
                    args.output_dir / f"MultiModalDiffusionTransformerStage{stage_index}.mlpackage",
                    deployment_target,
                )
                stage_index += 1

            if block_index == cfg.depth - 1:
                convert_module(
                    MMDiTBlockFinalPostStage(
                        model,
                        block_index,
                        args.latent_h,
                        args.latent_w,
                    ),
                    {
                        "latent_image_embeddings": hidden_image,
                        "token_level_text_embeddings": hidden_text,
                        "modulation_inputs": modulation,
                        "image_sdpa_output": image_sdpa,
                        "text_sdpa_output": text_sdpa,
                        "image_post_attn_scale": param,
                        "image_post_norm2_shift": param,
                        "image_post_norm2_residual_scale": param,
                        "image_post_mlp_scale": param,
                    },
                    ["denoiser_output"],
                    args.output_dir / f"MultiModalDiffusionTransformerStage{stage_index}.mlpackage",
                    deployment_target,
                )
            else:
                convert_module(
                    MMDiTBlockPostStage(
                        model,
                        block_index,
                        args.latent_h,
                        args.latent_w,
                    ),
                    {
                        "latent_image_embeddings": hidden_image,
                        "token_level_text_embeddings": hidden_text,
                        "image_sdpa_output": image_sdpa,
                        "text_sdpa_output": text_sdpa,
                        "image_post_attn_scale": param,
                        "image_post_norm2_shift": param,
                        "image_post_norm2_residual_scale": param,
                        "image_post_mlp_scale": param,
                        "text_post_attn_scale": param,
                        "text_post_norm2_shift": param,
                        "text_post_norm2_residual_scale": param,
                        "text_post_mlp_scale": param,
                    },
                    ["latent_image_embeddings_out", "token_level_text_embeddings_out"],
                    args.output_dir / f"MultiModalDiffusionTransformerStage{stage_index}.mlpackage",
                    deployment_target,
                )
            stage_index += 1

        logger.info("Split conversion complete: %s", args.output_dir)
        return 0

    for stage_index, (start, end) in enumerate(stage_ranges):
        stage = MMDiTStage(
            model,
            start,
            end,
            args.latent_h,
            args.latent_w,
            include_input_embedding=not args.split_input_embedding,
            sdpa_mode=args.sdpa_mode,
            splitkv_chunks=args.splitkv_chunks,
        )
        stage_inputs: dict[str, torch.Tensor] = {}
        if start == 0 and not args.split_input_embedding:
            stage_inputs["latent_image_embeddings"] = torch.randn(
                batch, cfg.vae_latent_dim, args.latent_h, args.latent_w
            )
            stage_inputs["token_level_text_embeddings"] = torch.randn(
                batch, cfg.token_level_text_embed_dim, 1, cfg.text_seq_len
            )
        else:
            stage_inputs["latent_image_embeddings"] = torch.randn(
                batch, cfg.hidden_size, 1, img_seq_len
            )
            stage_inputs["token_level_text_embeddings"] = torch.randn(
                batch, cfg.hidden_size, 1, cfg.text_seq_len
            )

        stage_inputs["modulation_inputs"] = torch.randn(batch, cfg.hidden_size, 1, 1)

        output_names = (
            ["denoiser_output"]
            if end == cfg.depth
            else ["latent_image_embeddings_out", "token_level_text_embeddings_out"]
        )
        convert_module(
            stage,
            stage_inputs,
            output_names,
            args.output_dir / f"MultiModalDiffusionTransformerStage{stage_index + stage_index_offset}.mlpackage",
            deployment_target,
        )

    logger.info("Split conversion complete: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
