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
