from qwen_batched.model.modeling_qwen2_5_vl_batched import Qwen2_5_VLForConditionalGenerationBatched
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as Qwen2_5_VLForConditionalGenerationOriginal
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from qwen_batched.model.tensor_processor import Qwen2_5_VLProcessorBatched
import torch

from qwen_batched.model.utils import tensor_to_pil_images, just_pad


def process_original(input_images, text_list):
    # Convert to PIL images (for standard processor)
    pil_images_all = []
    for i in range(input_images.shape[0]):
        pil_images_all.append(tensor_to_pil_images(input_images[i]))

    full_input_list = []
    for i in range(len(input_images)):
        # Process with original processor
        inputs_original = processor_original(
            images=pil_images_all[i],
            text=[text_list[i]],
            return_tensors='pt'
        ).to(input_images.device)
        full_input_list.append(inputs_original)
    return full_input_list


def process_batch(input_images, text_list):
    """
    Returns: {
        input_ids: (B, max_len)
        attention_mask: (B, max_len)
        pixel_values: (B*ncam*d1, d2), with d1/d2 results of patching
        image_grid_thw: (B*ncam, 3)
    }
    """
    return processor_batched(
        images=input_images.flatten(0, 1),
        text=list(text_list),
        return_tensors='pt',
        padding=True
    ).to(device)


@torch.no_grad()
def forward_original(full_input_list):
    outputs_original = []
    for inputs in full_input_list:
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward through the original model
        output = model_original(**inputs, output_hidden_states=True)

        outputs_original.append(output)
    return outputs_original


@torch.no_grad()
def forward_batched(full_input_list):
    inputs_padded = just_pad(full_input_list, device=device)
    outputs_batched = model_batched.batched_forward(**inputs_padded)
    return outputs_batched


@torch.no_grad()
def forward_batched2(inputs):
    return model_batched.batched_forward(
        input_ids_list=inputs['input_ids'][:, None],
        attention_mask_list=inputs['attention_mask'][:, None],
        pixel_values_list=inputs['pixel_values'].unflatten(0, (len(inputs['input_ids']), -1)),
        image_grid_thw_list=inputs['image_grid_thw'].unflatten(0, (len(inputs['input_ids']), -1))
    )


@torch.no_grad()
def forward_batched2_(inputs):
    return model_batched.batched_forward(
        input_ids_list=[inp for inp in inputs['input_ids'][:, None]],
        attention_mask_list=[inp for inp in inputs['attention_mask'][:, None]],
        pixel_values_list=[inp for inp in inputs['pixel_values'].unflatten(0, (len(inputs['input_ids']), -1))],
        image_grid_thw_list=[inp for inp in inputs['image_grid_thw'].unflatten(0, (len(inputs['input_ids']), -1))]
    )


@torch.no_grad()
def forward_batched3(inputs):
    return model_batched.batched_forward(**inputs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model initialization
    model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'
    model_original = Qwen2_5_VLForConditionalGenerationOriginal.from_pretrained(
        model_name, torch_dtype=torch.float32,
        attn_implementation="sdpa"
    ).to(device)
    processor_original = Qwen2_5_VLProcessor.from_pretrained(model_name)
    # Batched version
    model_batched = Qwen2_5_VLForConditionalGenerationBatched.from_pretrained(
        model_name, torch_dtype=torch.float32,
        attn_implementation="sdpa"
    ).to(device)
    processor_batched = Qwen2_5_VLProcessorBatched.from_pretrained(model_name)

    # Initialize random inputs: (B, ncam, 3, 224, 224)
    torch.manual_seed(0)
    input_images = torch.rand(5, 2, 3, 224, 224).to(device)
    # Trick to avoid numerical errors later
    input_images = (input_images * 255).byte().float() / 255
    images_per_batch = input_images.shape[1]  # 16 images per batch

    # Text
    text_list = [
        "Was kann Ich wissen?",
        "Was soll Ich tun?",
        "Was darf Ich hoffen?",
        "Was ist der Mensch?",
        "Why is everything in German?"
    ]
    # Add image slots
    text_list = [
        text + " ".join(["<|vision_start|><|image_pad|><|vision_end|>"] * images_per_batch)
        for text in text_list
    ]

    # Original forward pass
    full_input_list = process_original(input_images, text_list)
    outputs_original = forward_original(full_input_list)

    # Batched processing
    inputs_batched = process_batch(input_images, text_list)
    # Calculate preprocessing error
    print('Processing error')
    ori, bat = [], []
    for i in range(len(full_input_list)):
        _tens = full_input_list[i]['input_ids'][0]
        assert torch.all(_tens == inputs_batched['input_ids'][i][:len(_tens)])
        assert (
            full_input_list[i]['attention_mask'].sum()
            == inputs_batched['attention_mask'][i].sum()
        )
        _tens = full_input_list[i]['pixel_values']
        ori.append(_tens)
        _l = len(_tens)
        bat.append(inputs_batched['pixel_values'][i*_l:(i + 1)*_l])
        _l = len(full_input_list[i]['image_grid_thw'])
        assert torch.all(
            full_input_list[i]['image_grid_thw']
            == inputs_batched['image_grid_thw'][i*_l:(i + 1)*_l]
        )
    ori = torch.cat(ori)
    bat = torch.cat(bat)

    abs_error = torch.abs(bat - ori)
    rel_error = abs_error / (torch.abs(ori) + 1)

    print(f"Max abs error:  {abs_error.max().item():.6f}")
    print(f"Mean abs error: {abs_error.mean().item():.6f}")
    print(f"Max rel error:  {rel_error.max().item():.6f}")
    print(f"Mean rel error: {rel_error.mean().item():.6f}")

    # Insanity: compare just_pad outputs to inputs_batched
    inputs_padded = just_pad(full_input_list, device=device)
    list_inputs_batched = {
        'input_ids_list': inputs_batched['input_ids'][:, None],
        'attention_mask_list': inputs_batched['attention_mask'][:, None],
        'pixel_values_list': inputs_batched['pixel_values'].unflatten(0, (len(inputs_batched['input_ids']), -1)),
        'image_grid_thw_list': inputs_batched['image_grid_thw'].unflatten(0, (len(inputs_batched['input_ids']), -1))
    }
    assert torch.allclose(
        list_inputs_batched['pixel_values_list'],
        torch.stack(inputs_padded['pixel_values_list']),
        atol=1e-6
    )
    assert torch.allclose(
        list_inputs_batched['image_grid_thw_list'],
        torch.stack(inputs_padded['image_grid_thw_list']),
        atol=1e-6
    )
    assert torch.all(
        torch.stack(inputs_padded['input_ids_list'])
        == list_inputs_batched['input_ids_list']*list_inputs_batched['attention_mask_list']
    )
    assert torch.all(
        torch.stack(inputs_padded['attention_mask_list'])
        == list_inputs_batched['attention_mask_list']
    )
    list_inputs_batched['pixel_values_list'] = torch.stack(inputs_padded['pixel_values_list'])
    # import ipdb; ipdb.set_trace()

    # Batched forward pass
    for exp in range(3):
        if exp == 0:
            print("Using original processor")
            outputs_batched = forward_batched(full_input_list)  # B 1 ntok 2048
        elif exp == 1:
            print("Using batched processor")
            outputs_batched = forward_batched2(inputs_batched)  # B 1 ntok 2048
        else:
            print("Using batched processor but non-batched pixel values")
            outputs_batched = forward_batched3(list_inputs_batched)

        # Collect results and compare
        ori, bat = [], []
        for i in range(len(outputs_batched)):
            _tens = outputs_original[i]["hidden_states"][-1][0]
            ori.append(_tens)
            bat.append(outputs_batched[i, 0, :len(_tens)])
        ori = torch.cat(ori)
        bat = torch.cat(bat)

        abs_error = torch.abs(bat - ori)
        rel_error = abs_error / (torch.abs(ori) + 1)

        print(f"Max abs error:  {abs_error.max().item():.6f}")
        print(f"Mean abs error: {abs_error.mean().item():.6f}")
        print(f"Max rel error:  {rel_error.max().item():.6f}")
        print(f"Mean rel error: {rel_error.mean().item():.6f}")
