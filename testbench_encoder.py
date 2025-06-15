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
model_original = Qwen2_5_VLForConditionalGenerationOriginal.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
model_batched = Qwen2_5_VLForConditionalGenerationBatched.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
processor_tensor = QwenProc.from_pretrained(model_name)
processor_original = Qwen2_5_VLProcessor.from_pretrained(model_name)


input_images = torch.rand(4, 8, 3, 224, 224).to(device)  # 4 batches, each with 16 images of shape (3, 336, 336)
# import pdb; pdb.set_trace()

pil_images_all = []
for i in range(input_images.shape[0]):
    pil_images_batch = tensor_to_pil_images(input_images[i])  # Convert 16 images for batch i
    pil_images_all.append(pil_images_batch)
    
    

text_list = ["Was kann Ich wissen?", "Was soll Ich tun?", "Was darf Ich hoffen?", "Was ist der Mensch?"]




full_input_list = []
# here using a "for" in processor is not too bad. Can easily be converted to batch processing for images
for i in range(len(input_images)):
    text = [text_list[i]]
    
    # Tensor representation
    images_tensor = input_images[i]  # (16, 3, 336, 336)
    
    # PIL representation  
    images_pil = pil_images_all[i]   # List of 16 PIL Images
    
    # Process with tensor processor
    inputs_tensor = processor_tensor(
        images=images_tensor,
        text=text,
        return_tensors='pt',
    )
    
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
    

"""Test image_encoder"""
# here we must use batched processing, putting all the images from all batches together


# import pdb; pdb.set_trace()
with torch.no_grad():
    visual_original = model_original.visual.to(device)
    visual_batched = model_batched.visual.to(device)


    # Process all images together
    processed_images = torch.cat([inputs['pixel_values'] for inputs in full_input_list], dim=0).to(device)
    processor_grid_thw = torch.cat([inputs['image_grid_thw'] for inputs in full_input_list], dim=0).to(device)
    # processed_images = full_input_list[0]['pixel_values'].to(device)
    # processor_grid_thw = full_input_list[0]['image_grid_thw'].to(device)
    

    start_time = time.time()
    vision_outputs_original = visual_original(
                    processed_images, 
                    grid_thw=processor_grid_thw
                )    

    mid_time = time.time()
    print(f"Original vision processing time: {mid_time - start_time:.4f} seconds")

    vision_outputs_batched = visual_batched(
                    processed_images, 
                    grid_thw=processor_grid_thw
                )
    end_time = time.time()
    print(f"Batched vision processing time: {end_time - mid_time:.4f} seconds")

    # test 1
    # Original vision processing time: 1.1970 seconds
    # Batched vision processing time: 0.0395 seconds
    
    # test 2
    # Original vision processing time: 1.0908 seconds
    # Batched vision processing time: 0.0397 seconds 
    # 25x faster for 32 images
    
    # import pdb; pdb.set_trace()
    
    # Check if the outputs match
    # Calculate relative error
    diff = vision_outputs_batched - vision_outputs_original
    print(f"Max absolute error: {torch.max(torch.abs(diff)):.12f}")
    print(f"Mean absolute error: {torch.mean(torch.abs(diff)):.12f}")
    
    
    relative_error = torch.abs(diff) / (torch.abs(vision_outputs_original) + 1e-8)
    max_relative_error = torch.max(relative_error)
    mean_relative_error = torch.mean(relative_error)
    
    print(f"Max relative error: {max_relative_error:.12f}")
    print(f"Mean relative error: {mean_relative_error:.12f}")
    
    assert torch.allclose(vision_outputs_original, vision_outputs_batched, atol=1e-10), "Vision outputs do not match!"
    
    print("Vision outputs match successfully!")
    
    # the next stage would be building the decoder layers in a batched manner, batched attention.
    
    

    