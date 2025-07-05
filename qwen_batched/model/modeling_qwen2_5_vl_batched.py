# 
# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    BaseModelOutputWithPast,
    logging, 
    Qwen2_5_VLConfig,
    rotate_half,
    Qwen2_5_VLPreTrainedModel,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
    Qwen2_5_VLAttention,
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLVisionSdpaAttention,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration
)


logger = logging.get_logger(__name__)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(0).unsqueeze(-3).float()
    sin = sin.unsqueeze(0).unsqueeze(-3).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen2_5_VLVisionSdpaAttentionBatched(Qwen2_5_VLVisionSdpaAttention):

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=None
    ):
        batch_size, seq_length, _ = hidden_states.shape

        """previously, we are not using cu_seqlens, which is bad."""
        # Modified to support Batch processing
        q, k, v = self.qkv(hidden_states).reshape(batch_size, seq_length, 3, self.num_heads, -1).permute(0, 2, 3, 1, 4).unbind(1)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Adjusted attention mask
        attention_mask = torch.full(
            [batch_size, 1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        # import pdb; pdb.set_trace()
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_VLVisionBlockBatched(Qwen2_5_VLVisionBlock):

    def __init__(self, config, attn_implementation="sdpa"):
        super().__init__(config, attn_implementation)
        self.attn = Qwen2_5_VLVisionSdpaAttentionBatched(
            config.hidden_size, num_heads=config.num_heads
        )


class Qwen2_5_VisionTransformerPretrainedModelBatched(Qwen2_5_VisionTransformerPretrainedModel):
    _no_split_modules = ["Qwen2_5_VLVisionBlockBatched"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.blocks = nn.ModuleList(
            [Qwen2_5_VLVisionBlockBatched(config, config._attn_implementation) for _ in range(config.depth)]
        )

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        
        
        grid_thw_batch_one = grid_thw[0].unsqueeze(0)  # For batch processing, we assume all images have the same grid size
        rotary_pos_emb = self.rot_pos_emb(grid_thw_batch_one)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw_batch_one)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        # import pdb; pdb.set_trace()
        
        # window index:tensor([ 0,  1,  2,  3,  8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27,  4,  5,
        #  6,  7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 32, 33, 34, 35,
        # 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59, 36, 37, 38, 39, 44, 45,
        # # 46, 47, 52, 53, 54, 55, 60, 61, 62, 63])
        
        
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        # import pdb; pdb.set_trace()
        
        # pretend that we are processing a single image, so that we can use the original subwindow attention
        # This is a hack to make the code work for batch processing, we assume all images have the same grid size
        single_image_seq_length = int(grid_thw[0, 1] * grid_thw[0, 2])
        seq_len, _ = hidden_states.shape
        batch_size = seq_len // single_image_seq_length
        hidden_states = hidden_states.reshape(batch_size, single_image_seq_length, -1)  # B, h*w, embed_dim
        hidden_states = hidden_states.reshape(batch_size, single_image_seq_length // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[:, window_index, :, :] # nonsense
        hidden_states = hidden_states.reshape(seq_len, -1)
        
        
        rotary_pos_emb = rotary_pos_emb.reshape(single_image_seq_length // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(single_image_seq_length, -1) # this is another nonsense
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        
        
        # manage to get the positional embeddings like the original implementation

        cu_seqlens = torch.repeat_interleave(grid_thw_batch_one[:, 1] * grid_thw_batch_one[:, 2], grid_thw_batch_one[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw_batch_one.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # import pdb; pdb.set_trace()
        # reshape hidden_states to support batch processing
        seq_length_sum, _ = hidden_states.shape
        batch_size = seq_length_sum // single_image_seq_length
        hidden_states = hidden_states.reshape(batch_size, single_image_seq_length, -1)  # B, h*w, embed_dim
        
        # self.fullatt_block_indexes
        # print("self.fullatt_block_indexes:", self.fullatt_block_indexes)
        # import pdb; pdb.set_trace()
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

        hidden_states = hidden_states.reshape(batch_size * single_image_seq_length, -1)
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states.reshape(batch_size, len(window_index), -1)
        hidden_states = hidden_states[:,reverse_indices, :]
        hidden_states = hidden_states.reshape(batch_size * len(window_index), -1)

        return hidden_states
    
    
    """The github author version. not considering the window seq attn."""
    # def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    #     hidden_states = self.patch_embed(hidden_states)
    #     rotary_pos_emb = self.rot_pos_emb(grid_thw)

    #     cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
    #         dim=0, dtype=torch.int32
    #     )
    #     cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        
    #     # Modified: Seq===>Batch
    #     single_image_seq_length = int(grid_thw[0, 1] * grid_thw[0, 2])
    #     seq_length_sum, _ = hidden_states.shape
        
    #     """Here, we can only handle the case where all images have the same spatial size. (which is often the case in practice)"""
        
    #     batch_size = seq_length_sum // single_image_seq_length
        
    #     # change the shape of hidden_states and rotary_pos_emb to support batch processing
    #     hidden_states = hidden_states.reshape(batch_size, single_image_seq_length, -1)  # B, h*w, embed_dim
    #     rotary_pos_emb = rotary_pos_emb.reshape(batch_size, single_image_seq_length, -1)  # B, h*w, head_dim

    #     for blk in self.blocks:
    #         hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        
    #     # Modified: Batch===>Seq
    #     hidden_states = hidden_states.reshape(batch_size * single_image_seq_length, -1)
    #     return self.merger(hidden_states)


class Qwen2_5_VLRotaryEmbeddingBatched(Qwen2_5_VLRotaryEmbedding):
    def __init__(self, config, device=None):
        super().__init__(config, device)

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2_5_VLSdpaAttention(Qwen2_5_VLAttention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings_list: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Any,  # Additional arguments for BC
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2_5_VLModel is using Qwen2_5_VLSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings_list,
            )
            
        # print("using sdpa attn")

        # Handle batched input dimension squeezing (same as FlashAttention)
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)  # Remove the redundant batch dimension if it is 1 
            attention_mask = attention_mask.squeeze(1)  # Remove the redundant batch dimension if it is 1
            # import pdb; pdb.set_trace()
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        # Apply rotary position embeddings per batch item (same as FlashAttention)
        query_states_list = []
        key_states_list = []
        for i in range(len(position_embeddings_list)):
            cos, sin = position_embeddings_list[i]
            this_query_states, this_key_states = query_states[i].unsqueeze(0), key_states[i].unsqueeze(0)
            this_query_states, this_key_states = apply_multimodal_rotary_pos_emb(
                this_query_states, this_key_states, cos, sin, self.rope_scaling["mrope_section"]
            )
            query_states_list.append(this_query_states)
            key_states_list.append(this_key_states)
        query_states = torch.cat(query_states_list, dim=0)
        key_states = torch.cat(key_states_list, dim=0)

        # if past_key_value is not None:
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = ~attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        # import pdb; pdb.set_trace()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output).unsqueeze(1)  # go back to original shape [bsz, 1, q_len, hidden_size]

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2_5_VLDecoderLayerBatched(Qwen2_5_VLDecoderLayer):

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2_5_VLSdpaAttention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        text_length=0,
        use_text=True,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length_points, k_points)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states) # B, N+L, dim

        # Self Attention
        # print("inside decoder layer")
        
        """This function is already modified"""
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings_list=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Qwen2_5_VLModelBatched(Qwen2_5_VLModel):

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayerBatched(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = Qwen2_5_VLRotaryEmbeddingBatched(config=config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        
        # assert not (inputs_embeds[0]==inputs_embeds[2]).all() # avoid all same inputs, only for debug

        
        # if not self._attn_implementation == "flash_attention_2": # use 4d mask
        causal_mask_list = []
        for i in range(position_ids.shape[0]):
            causal_mask = self._update_causal_mask(
                attention_mask[i], inputs_embeds[i], cache_position, past_key_values, output_attentions
            )
            causal_mask_list.append(causal_mask)
            # import pdb; pdb.set_trace()
        causal_mask = torch.stack(causal_mask_list, dim=0)
        # else:
        #     causal_mask = attention_mask

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings_list = []
        for i in range(position_ids.shape[0]):
            position_embeddings = self.rotary_emb(hidden_states[i], position_ids[i])
            position_embeddings_list.append(position_embeddings)
        

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings_list,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings_list,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2_5_VL. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        
        """Yifan: Abrupt changes"""
        # if (
        #     self.config._attn_implementation == "sdpa"
        #     and not (using_static_cache or using_sliding_window_cache)
        #     and not output_attentions
        # ):
        #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
        #         attention_mask,
        #         inputs_embeds=input_tensor,
        #         past_key_values_length=past_seen_tokens,
        #         sliding_window=self.config.sliding_window,
        #         is_training=self.training,
        #     ):
        #         return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )
        # import pdb; pdb.set_trace()
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            pass
            # causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        # import pdb; pdb.set_trace()
        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2_5_VLConfig,
        past_key_values: Cache,
    ):
        # --- SPECIAL CASE: no static/sliding cache => just do standard causal ---
        if not isinstance(past_key_values, (StaticCache, SlidingWindowCache)):
            # mask[j, k] = True if k > j
            mask2d = torch.triu(torch.ones(sequence_length, target_length, device=device, dtype=torch.bool), 1)
            causal = mask2d[None, None, :, :].expand(batch_size, 1, sequence_length, target_length)
            if attention_mask is not None:
                pad = (attention_mask == 0)[..., :target_length]
                pad4 = pad[:, None, None, :].expand_as(causal)
                causal = causal | pad4
            return causal

        # --- otherwise fall back to your cache_position logic ---
        # if user already passed a 4D boolean mask:
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask.to(torch.bool)

        # … the rest of your old code using cache_position …
        min_dtype = torch.finfo(dtype).min
        # create the (seq_len × tgt_len) boolean “future+window” map as before
        diagonal = torch.arange(target_length, device=device).unsqueeze(0) > cache_position.reshape(-1, 1)
        if config.sliding_window is not None:
            if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                slide = torch.arange(target_length, device=device).unsqueeze(0) <= (
                    cache_position.reshape(-1, 1) - config.sliding_window
                )
                diagonal |= slide

        causal = diagonal[None, None, :sequence_length, :target_length].expand(batch_size, 1, sequence_length, target_length)
        if attention_mask is not None:
            pad = (attention_mask == 0)[..., :target_length]
            pad4 = pad[:, None, None, :].expand_as(causal)
            causal = causal | pad4

        return causal


class Qwen2_5_VLForConditionalGenerationBatched(Qwen2_5_VLForConditionalGeneration):
    _no_split_modules = ["Qwen2_5_VLDecoderLayerBatched", "Qwen2_5_VLVisionBlockBatched"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModelBatched._from_config(config.vision_config)
        self.model = Qwen2_5_VLModelBatched(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def batched_forward(
        self,
        input_ids_list: torch.LongTensor = None,
        attention_mask_list: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values_list: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw_list: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if isinstance(pixel_values_list, torch.Tensor):
            batched_pixel_values = pixel_values_list.reshape(-1, *pixel_values_list.shape[2:])
            batched_image_grid_thw = image_grid_thw_list.reshape(-1, 3)
        else: # list
            batched_pixel_values = torch.cat(pixel_values_list, dim=0)
            batched_image_grid_thw = torch.cat(image_grid_thw_list, dim=0)
        batched_image_embeds = self.visual(batched_pixel_values, grid_thw=batched_image_grid_thw)
        batched_length, channels = batched_image_embeds.shape
        bs = len(input_ids_list)
        single_length = batched_length // bs
        batched_image_embeds = batched_image_embeds.reshape(bs, single_length, channels)

        position_ids_tensor = []
        attention_mask_tensor = []
        inputs_embeds_tensor = []
        # this is easy job, no computation at all, don't want to batchify
        # Process each input independently
        for input_ids, attention_mask, visual_out, image_grid_thw in zip(
            input_ids_list,
            attention_mask_list,
            batched_image_embeds,
            image_grid_thw_list,
        ):
            # Create fresh inputs_embeds for each input (remove the if condition)
            current_inputs_embeds = self.model.embed_tokens(input_ids)
            
            if visual_out is not None:
                image_embeds = visual_out
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(current_inputs_embeds)
                image_mask = mask_expanded.to(current_inputs_embeds.device)

                image_embeds = image_embeds.to(current_inputs_embeds.device, current_inputs_embeds.dtype)
                current_inputs_embeds = current_inputs_embeds.masked_scatter(image_mask, image_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(current_inputs_embeds.device)

            # Calculate position_ids for each input independently
            current_position_ids, current_rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            
            position_ids_tensor.append(current_position_ids)
            attention_mask_tensor.append(attention_mask)
            inputs_embeds_tensor.append(current_inputs_embeds)
        
        position_ids = torch.stack(position_ids_tensor, dim=0) # torch.Size([4, 3, 1, 140])
        attention_mask = torch.stack(attention_mask_tensor, dim=0) # torch.Size([4, 1, 140])
        inputs_embeds = torch.stack(inputs_embeds_tensor, dim=0) # torch.Size([4, 1, 140, 2048])

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,  # we can use any layer
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        return hidden_states


__all__ = ["Qwen2_5_VLForConditionalGenerationBatched", "Qwen2_5_VLModelBatched", "Qwen2_5_VLPreTrainedModel"]
