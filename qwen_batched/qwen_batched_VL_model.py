import torch
from torch import nn
from typing import List, Union
from qwen_batched.model.tensor_processor import Qwen2_5_VLProcessorBatched
from qwen_batched.model.modeling_qwen2_5_vl_batched import (
    Qwen2_5_VLForConditionalGenerationBatched
)

class QwenBatchedVLModel(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = None
    ):
        super().__init__()
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # tensor processor (prep images+text → tensors)
        self.processor = Qwen2_5_VLProcessorBatched.from_pretrained(model_name)
        # batched ViT + decoder
        self.model = Qwen2_5_VLForConditionalGenerationBatched.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).to(self.device)

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,       # (B, ncam, 3, H, W)
        texts: List[str],           # len(texts)==B
        max_length: int = 50
    ) -> torch.Tensor:
        B, ncam, C, H, W = images.shape
        
        B, ncam, C, H, W = images.shape
        # --- Inject vision markers automatically per camera ---
        vision_token = "<|vision_start|><|image_pad|><|vision_end|>"
        texts = [
            text + " " + " ".join([vision_token] * ncam)
            for text in texts
        ]
        
        # flatten cams
        imgs = images.flatten(0, 1).to(self.device)  # (B*ncam, C, H, W)

        # run batched processor (pads text & images)
        inputs = self.processor(
            images=imgs,
            text=list(texts),
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        outputs = self.model.batched_forward(
            input_ids_list=inputs['input_ids'][:, None],
            attention_mask_list=inputs['attention_mask'][:, None],
            pixel_values_list=inputs['pixel_values'].unflatten(0, (len(inputs['input_ids']), -1)),
            image_grid_thw_list=inputs['image_grid_thw'].unflatten(0, (len(inputs['input_ids']), -1))
        )
        
        return outputs