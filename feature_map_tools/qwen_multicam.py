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
"""PyTorch Qwen2.5-VL model."""
from typing import Optional, List, Tuple, Union

import torch
import ipdb
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.utils import is_torchdynamo_compiling
from transformers.cache_utils import StaticCache
from torch.nn import CrossEntropyLoss
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLModel,
    Qwen2_5_VisionTransformerPretrainedModel,
)

st = ipdb.set_trace


class Qwen2_5_Projected3D(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel(config)
        
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        #              (we can't check exception 3 while compiling)
        # Exception 4: If input_embeds are passed then slice it through `cache_position`, to keep only the unprocessed tokens and
        # generate the first token for each sequence. Later use the generated Input ids for continuation.
        if past_key_values is not None:
            if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            elif (
                inputs_embeds is not None  # Exception 1
                or (is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1])  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
                "second_per_grid_ts": second_per_grid_ts,
            }
        )
        
        # add kwargs to model_inputs
        for k, v in kwargs.items():
            if k not in model_inputs:
                model_inputs[k] = v
        
        return model_inputs


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        points_cam: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        all_image_embeds: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids =can= self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # find the valid features from the feature cloud
        # assumption I am making here: the format is start <text> <image> <text> end
        if all_image_embeds is not None and pixel_values is not None:
            
            print("inside Ayush's code")
            
            # (Pdb) p all_image_embeds.shape
            # torch.Size([3367, 2048])
            # (Pdb) p points_cam.shape
            # torch.Size([1, 1, 3367, 2])
            # (Pdb) p valid_mask.shape
            # torch.Size([1, 1, 3367]) 
            

            # all_image_embeds  [3367, 2048]
            
            valid_features = all_image_embeds[valid_mask.squeeze(0, 1)]
            non_image_features = inputs_embeds[~(image_mask.sum(-1).bool())]
            new_input_embeds = torch.zeros(
                (inputs_embeds.shape[0], valid_features.shape[0] + non_image_features.shape[0], inputs_embeds.shape[2]),
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            new_image_mask = torch.zeros((image_mask.shape[0], new_input_embeds.shape[1]), device=inputs_embeds.device, dtype=torch.bool)
            nonzero_mask = image_mask.sum(-1).bool().nonzero()
            new_image_mask[:, :nonzero_mask[:, 1].min().item()] = True
            new_image_mask[:, -(image_mask.shape[1] - nonzero_mask[:, 1].max().item() - 1):] = True
            new_image_mask = ~new_image_mask
            
            # fill in the non image tokens
            new_input_embeds[~new_image_mask] = non_image_features
            
            
            
            # fill in the positional embeddings
            # hold both image and text pe
            new_position_ids = torch.zeros(
                (position_ids.shape[0], position_ids.shape[1], new_input_embeds.shape[1]),
                device=inputs_embeds.device, dtype=position_ids.dtype
            )
            
            # fill in non image positional embeddings
            non_image_position_ids = position_ids[..., ~(image_mask.sum(-1).bool())]
            new_position_ids[:, ~new_image_mask] = non_image_position_ids
            
            
            valid_points_cam = points_cam[0, 0, valid_mask.squeeze(0, 1)]
            valid_points_cam[..., 0] *= (image_grid_thw[0][2] / 2) # (0, Width)
            valid_points_cam[..., 1] *= (image_grid_thw[0][1] / 2) # (0, Height)
            # valid_points_cam = valid_points_cam.floor().long()
            valid_points_cam = valid_points_cam.round().long()
            
            
            # fill in image positional embedding
            image_position_ids = position_ids[..., image_mask.sum(-1).bool()]
            
            # Calculate indices
            indices = valid_points_cam[:, 1] * (image_grid_thw[0][2] // 2) + valid_points_cam[:, 0]
            
            # Clip indices to valid range [0, length-1]
            max_idx = image_position_ids.shape[-1] - 1
            indices = torch.clamp(indices, 0, max_idx)
            
            valid_position_ids = image_position_ids[:, indices]
            # valid_features, valid_position_ids
            
            
            
            # here I have valid_position_ids of shape (3, 359) note that valid_position_ids[1] is integer, the row index; and valid_position_ids[2] is integer, the column index. 
            
            # points_viz = valid_position_ids[1:,...].permute(1, 0) # y,x
            # visualize_grid_points(points_viz)
            
            # import ipdb
            # ipdb.set_trace()
            
            valid_features = valid_features.to(inputs_embeds)
            
            
            priority = indices
            
            sorted_indices = torch.argsort(priority, descending=False)  # Sort in ascending order of priority
            valid_position_ids = valid_position_ids[..., sorted_indices]
            valid_features = valid_features[sorted_indices]
            
            # Update new_position_ids and new_input_embeds with the sorted values
            new_position_ids[:, new_image_mask] = valid_position_ids
            new_input_embeds[new_image_mask] = valid_features.to(inputs_embeds)
            
            # shuffle_indices = torch.randperm(valid_position_ids.shape[-1], device=valid_position_ids.device)
            # valid_position_ids = valid_position_ids[..., shuffle_indices]
            # valid_features = valid_features[shuffle_indices]
            
            # points_viz = valid_position_ids[1:,...].permute(1, 0) # y,x
            # visualize_grid_points(points_viz)
            # import ipdb
            # ipdb.set_trace()
            
            new_position_ids[:, new_image_mask] = valid_position_ids
            # fill in the image tokens
            new_input_embeds[new_image_mask] = valid_features
            
            
            # reset all the inputs to the model
            position_ids = new_position_ids
            attention_mask = torch.ones_like(new_image_mask).to(attention_mask)
            inputs_embeds = new_input_embeds
            cache_position = None # just to be safe
            
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


import matplotlib.pyplot as plt
import numpy as np

def visualize_grid_points(valid_points_cam, grid_height=17, grid_width=23, width_divisor=2):
    """
    Visualize points on a grid with their linear indices.
    
    Args:
        valid_points_cam: Tensor of shape (N, 2) containing (x, y) coordinates
        grid_height: Height of the grid (default: 17)
        grid_width: Width of the grid (default: 23)
        width_divisor: Divisor for width when calculating linear indices (default: 2)
    """
    # Create figure and axis
    plt.figure(figsize=(15, 10))
    ax = plt.gca()
    
    # Draw grid
    for i in range(grid_height + 1):
        ax.axhline(i, color='gray', linestyle='-', alpha=0.3)
    for j in range(grid_width + 1):
        ax.axvline(j, color='gray', linestyle='-', alpha=0.3)
    
    # Convert tensor to numpy for plotting if needed
    if isinstance(valid_points_cam, torch.Tensor):
        points = valid_points_cam.cpu().numpy()
    else:
        points = valid_points_cam
    
    y_min = points[:, 0].min()
    x_min = points[:, 1].min()
    # Plot each point with its linear index
    for i in range(len(points)):
        y, x = points[i]
        y = y - y_min
        x = x - x_min
        linear_idx = i
        
        # Plot point
        ax.plot(x + 0.5, y + 0.5, 'ro', markersize=5)
        
        # Add text label with linear index
        ax.text(x + 0.6, y + 0.5, str(linear_idx), fontsize=8)
    
    # Set axis limits and labels
    ax.set_xlim(0, grid_width)
    ax.set_ylim(grid_height, 0)  # Invert y-axis to match image coordinates
    ax.set_xticks(np.arange(0, grid_width + 1))
    ax.set_yticks(np.arange(0, grid_height + 1))
    ax.set_title(f'Grid of {grid_height}x{grid_width} with {len(points)} points')
    ax.set_xlabel('Column Index (X)')
    ax.set_ylabel('Row Index (Y)')
    
    # Save and show the figure
    plt.tight_layout()
    plt.savefig('grid_visualization.png', dpi=300)
    plt.show()