from modeling_qwen2_5_vl_batched import Qwen2_5_VLForConditionalGeneration as Qwen2_5_VLForConditionalGenerationBatched
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as Qwen2_5_VLForConditionalGenerationOriginal
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from tensor_processor import QwenProc
from transformers import AutoTokenizer
import torch
from PIL import Image
import numpy as np
import os
import time

def tensor_to_pil_images(tensor_images):
    """
    Convert tensor images (B, C, H, W) in [0, 1] to list of PIL Images
    """
    # Clamp to [0, 1] and convert to uint8
    tensor_images = torch.clamp(tensor_images, 0, 1)
    # Convert to numpy and scale to [0, 255]
    numpy_images = (tensor_images * 255).byte().cpu().numpy()
    
    pil_images = []
    for i in range(numpy_images.shape[0]):
        # Convert from (C, H, W) to (H, W, C)
        img_array = numpy_images[i].transpose(1, 2, 0)
        pil_img = Image.fromarray(img_array, mode='RGB')
        pil_images.append(pil_img)
    
    return pil_images

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'
model_original = Qwen2_5_VLForConditionalGenerationOriginal.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation = "flash_attention_2").to(device)
model_batched = Qwen2_5_VLForConditionalGenerationBatched.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
processor_tensor = QwenProc.from_pretrained(model_name)
processor_original = Qwen2_5_VLProcessor.from_pretrained(model_name)


input_images = torch.rand(1, 2, 3, 224, 224).to(device)  # 4 batches, each with 16 images of shape (3, 336, 336)
# import pdb; pdb.set_trace()
images_per_batch = input_images.shape[1]  # 16 images per batch

pil_images_all = []
for i in range(input_images.shape[0]):
    pil_images_batch = tensor_to_pil_images(input_images[i])  # Convert 16 images for batch i
    pil_images_all.append(pil_images_batch)
    
    

text_list = ["Was kann Ich wissen?", "Was soll Ich tun?", "Was darf Ich hoffen?", "Was ist der Mensch?"]

# add image 
text_list = [text +" ".join(["<|vision_start|><|image_pad|><|vision_end|>"] * images_per_batch) for text in text_list]

full_input_list = []
# here using a "for" in processor is not too bad. Can easily be converted to batch processing for images
for i in range(len(input_images)):
    text = [text_list[i]]
    
    # Tensor representation
    images_tensor = input_images[i]  # (16, 3, 336, 336)
    
    # PIL representation  
    images_pil = pil_images_all[i]   # List of 16 PIL Images
    
    # Process with tensor processor
    # inputs_tensor = processor_tensor(
    #     images=images_tensor,
    #     text=text,
    #     return_tensors='pt',
    # )
    
    # Process with original processor
    inputs_original = processor_original(
        images=images_pil,
        text=text,
        return_tensors='pt',
    )
    
    full_input_list.append(inputs_original)
    # print(f"Tensor inputs: {inputs_tensor.keys()}")
    # print(f"Original inputs: {inputs_original.keys()}")
    # Tensor inputs: dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])
    # Original inputs: dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])
    
    
    """Test the QwenProc"""
    if 0: 
        # Check if the tensor processor outputs match the original processor
        assert (inputs_tensor['input_ids']==inputs_original['input_ids']).all()
        assert (inputs_tensor['attention_mask']==inputs_original['attention_mask']).all()
        print(inputs_tensor['image_grid_thw'])
        print(inputs_original['image_grid_thw'])
        #     tensor([[ 8, 24, 24]])
        # tensor([[ 1, 24, 24],
        #         [ 1, 24, 24],
        #         [ 1, 24, 24],
        #         [ 1, 24, 24],
        #         [ 1, 24, 24],
        #         [ 1, 24, 24],
        #         [ 1, 24, 24],
        #         [ 1, 24, 24],
        #         [ 1, 24, 24],
        #         [ 1, 24, 24],
        #         [ 1, 24, 24],
        #         [ 1, 24, 24],
        #         [ 1, 24, 24],
        #         [ 1, 24, 24],
        #         [ 1, 24, 24],
        #         [ 1, 24, 24]])
        # Traceback (most recent call last):
        #   File "/home/yifanliu/qwen_batched/testbench.py", line 81, in <module>
        #     assert (inputs_tensor['image_grid_thw'][0]==inputs_original['image_grid_thw'][0]).all()
        # AssertionError

        """a bug: QwenProc is giving [8,24,24] out of 16 images, where it should be [1,24,24] * 16 """
        
        
        assert (inputs_tensor['image_grid_thw'][0]==inputs_original['image_grid_thw'][0]).all()
        assert torch.allclose(inputs_tensor['pixel_values'], inputs_original['pixel_values'], atol=1e-6)
        assert (inputs_tensor['image_grid_thw'][0]==inputs_original['image_grid_thw'][0]).all()
        import pdb; pdb.set_trace()
    

"""Test the whole model """
# here we must use batched processing, putting all the images from all batches together


# first pass through the original model one by one
outputs_original = []
for inputs in full_input_list:
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward through the original model
    with torch.no_grad():
        output = model_original(**inputs, output_hidden_states=True)
    
    outputs_original.append(output)


# forward in a batched manner

# Forward through our batched model

from modeling_qwen2_5_vl_batched import just_pad
inputs_padded = just_pad(full_input_list, device=device)
outputs_batched = model_batched.batched_forward(**inputs_padded)
# torch.Size([4, 1, 140, 2048])
ori_out= outputs_original[0]["hidden_states"][36]
real_length = ori_out.shape[1]
batched_out= outputs_batched[0][:, :real_length, :]
# Check if the outputs match
# ori_out has shape [1, seq, dim], squeeze to [seq, dim]
ori = ori_out.squeeze(0)
# batched_out has shape [batch, seq, dim], take the same index = 1
bat = batched_out[0]
import pdb; pdb.set_trace()
abs_error = torch.abs(bat - ori)
rel_error = abs_error / (torch.abs(ori) + 1)

print(f"Max abs error:  {abs_error.max().item():.6f}")
print(f"Mean abs error: {abs_error.mean().item():.6f}")
print(f"Max rel error:  {rel_error.max().item():.6f}")
print(f"Mean rel error: {rel_error.mean().item():.6f}")

# mean abs error: Mean abs error: 0.030487
# Mean rel error: 0.012009

# Optionally assert tolerance
assert abs_error.max() < 1e-2, "Absolute error exceeds 1e-2"
assert rel_error.max() < 1e-2, "Relative error exceeds 1e-2"

import pdb; pdb.set_trace()
# compare the hiddens states