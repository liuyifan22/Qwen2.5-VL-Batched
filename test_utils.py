import torch
from PIL import Image


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


def inputs_to_list(inputs):
    b = len(inputs['input_ids'])
    return [
        {
            'input_ids': inputs['input_ids'][i][None],
            'attention_mask': inputs['attention_mask'][i][None],
            'pixel_values': inputs['pixel_values'].unflatten(0, (b, -1))[i],
            'image_grid_thw': inputs['image_grid_thw'].unflatten(0, (b, -1))[i]
        }
        for i in range(b)
    ]


def just_pad(input_list, device):
    """
    Pad input_ids and attention_mask to same length and return as dict with lists
    """
    # Extract all components
    input_ids_list = [inp['input_ids'].squeeze(0) for inp in input_list]
    attention_mask_list = [inp['attention_mask'].squeeze(0) for inp in input_list]
    
    # Find max sequence length
    max_seq_len = max(ids.shape[0] for ids in input_ids_list)
    
    # Initialize result lists
    padded_input_ids = []
    padded_attention_masks = []
    pixel_values_list = []
    image_grid_thw_list = []
    
    for i, inputs in enumerate(input_list):
        ids = input_ids_list[i]
        mask = attention_mask_list[i]
        
        pad_len = max_seq_len - ids.shape[0]
        
        if pad_len > 0:
            # Pad with zeros (or tokenizer.pad_token_id)
            padded_ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype, device=ids.device)])
            padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype, device=mask.device)])
        else:
            padded_ids = ids
            padded_mask = mask
        
        # Add to lists (with batch dimension)
        padded_input_ids.append(padded_ids.unsqueeze(0).to(device))
        padded_attention_masks.append(padded_mask.unsqueeze(0).to(device))
        pixel_values_list.append(inputs['pixel_values'].to(device))
        image_grid_thw_list.append(inputs['image_grid_thw'].to(device))
    
    # Return as dict with lists
    return {
        'input_ids_list': padded_input_ids,
        'attention_mask_list': padded_attention_masks,
        'pixel_values_list': pixel_values_list,
        'image_grid_thw_list': image_grid_thw_list
    }
