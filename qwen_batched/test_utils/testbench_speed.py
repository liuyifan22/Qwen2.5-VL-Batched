import torch
import gc
from time import time
import numpy as np
from qwen_batched.model.modeling_qwen2_5_vl_batched import Qwen2_5_VLForConditionalGenerationBatched
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as Qwen2_5_VLForConditionalGenerationOriginal
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from qwen_batched.model.tensor_processor import Qwen2_5_VLProcessorBatched
from qwen_batched.model.utils import tensor_to_pil_images, just_pad


def clear_memory():
    """Clear GPU memory between tests"""
    gc.collect()
    torch.cuda.empty_cache()


def test_processor_speed(num_runs=50):
    """Test 1: Processor comparison - Original one-by-one vs Batched"""
    print("=" * 60)
    print("TEST 1: PROCESSOR SPEED COMPARISON")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'
    
    # Initialize processors
    processor_original = Qwen2_5_VLProcessor.from_pretrained(model_name)
    processor_batched = Qwen2_5_VLProcessorBatched.from_pretrained(model_name)
    
    # Test data: batch of 8 samples with 4 images each
    torch.manual_seed(42)
    batch_size = 8
    images_per_sample = 4
    input_images = torch.rand(batch_size, images_per_sample, 3, 224, 224).to(device)
    input_images = (input_images * 255).byte().float() / 255
    
    text_list = [
        f"Describe these {images_per_sample} images in detail." + 
        " ".join(["<|vision_start|><|image_pad|><|vision_end|>"] * images_per_sample)
        for _ in range(batch_size)
    ]
    
    # Test original processor (one by one)
    def process_original_sequential():
        pil_images_all = []
        for i in range(input_images.shape[0]):
            pil_images_all.append(tensor_to_pil_images(input_images[i]))
        
        results = []
        for i in range(len(input_images)):
            inputs = processor_original(
                images=pil_images_all[i],
                text=[text_list[i]],
                return_tensors='pt'
            ).to(device)
            results.append(inputs)
        return results
    
    # Test batched processor
    def process_batched_all():
        return processor_batched(
            images=input_images.flatten(0, 1),
            text=list(text_list),
            return_tensors='pt',
            padding=True
        ).to(device)
    
    # Warmup
    for _ in range(5):
        process_original_sequential()
        process_batched_all()
    
    # Time original processor
    torch.cuda.synchronize()
    start_time = time()
    for _ in range(num_runs):
        process_original_sequential()
    torch.cuda.synchronize()
    original_time = time() - start_time
    
    # Time batched processor
    torch.cuda.synchronize()
    start_time = time()
    for _ in range(num_runs):
        process_batched_all()
    torch.cuda.synchronize()
    batched_time = time() - start_time
    
    print(f"Original processor (sequential): {original_time:.4f}s for {num_runs} runs")
    print(f"Batched processor: {batched_time:.4f}s for {num_runs} runs")
    print(f"Speedup: {original_time / batched_time:.2f}x")
    print(f"Per-sample original: {original_time * 1000 / (num_runs * batch_size):.2f}ms")
    print(f"Per-sample batched: {batched_time * 1000 / (num_runs * batch_size):.2f}ms")
    
    clear_memory()


def test_visual_encoder_speed(num_runs=20):
    """Test 2: Visual encoder comparison with 32 image sequence"""
    print("\n" + "=" * 60)
    print("TEST 2: VISUAL ENCODER SPEED COMPARISON (64 images)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'
    
    # Test data: 
    torch.manual_seed(42)
    num_images = 64
    pixel_values = torch.rand(65536, 1176).to(device)  
    image_grid_thw = torch.tensor([[1, 32,32]] * 64).to(device)
    
    # Test original model visual encoder
    print("Loading original model...")
    model_original = Qwen2_5_VLForConditionalGenerationOriginal.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    
    @torch.no_grad()
    def test_original_visual():
        # Process images one by one using the processed inputs
        return model_original.visual(
            pixel_values, image_grid_thw
        )
    
    # Warmup
    for _ in range(3):
        test_original_visual()
    
    # Time original visual encoder
    torch.cuda.synchronize()
    start_time = time()
    for _ in range(num_runs):
        test_original_visual()
    torch.cuda.synchronize()
    original_time = time() - start_time
    
    # Clear original model
    del model_original
    clear_memory()
    
    # Test batched model visual encoder
    print("Loading batched model...")
    model_batched = Qwen2_5_VLForConditionalGenerationBatched.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    
    @torch.no_grad()
    def test_batched_visual():
        return model_batched.visual(
            pixel_values, image_grid_thw
        )
    
    # Warmup
    for _ in range(3):
        test_batched_visual()
    
    # Time batched visual encoder
    torch.cuda.synchronize()
    start_time = time()
    for _ in range(num_runs):
        test_batched_visual()
    torch.cuda.synchronize()
    batched_time = time() - start_time
    
    print(f"Original visual encoder (sequential): {original_time:.4f}s for {num_runs} runs")
    print(f"Batched visual encoder: {batched_time:.4f}s for {num_runs} runs")
    print(f"Speedup: {original_time / batched_time:.2f}x")
    print(f"Per-image original: {original_time * 1000 / (num_runs * num_images):.2f}ms")
    print(f"Per-image batched: {batched_time * 1000 / (num_runs * num_images):.2f}ms")
    
    del model_batched
    clear_memory()


def test_full_model_speed(num_runs=10):
    """Test 3: Full model comparison - batch of 4 with 4 images each"""
    print("\n" + "=" * 60)
    print("TEST 3: FULL MODEL SPEED COMPARISON (4 samples x 4 images)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'
    
    # Test data
    torch.manual_seed(42)
    batch_size = 4
    images_per_sample = 8
    input_images = torch.rand(batch_size, images_per_sample, 3, 224, 224).to(device)
    input_images = (input_images * 255).byte().float() / 255
    
    text_list = [
        f"Sample {i}: Analyze these images." + 
        " ".join(["<|vision_start|><|image_pad|><|vision_end|>"] * images_per_sample)
        for i in range(batch_size)
    ]
    
    # Test original model
    print("Loading original model...")
    model_original = Qwen2_5_VLForConditionalGenerationOriginal.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    processor_original = Qwen2_5_VLProcessor.from_pretrained(model_name)
    
    def prepare_original_inputs():
        pil_images_all = []
        for i in range(input_images.shape[0]):
            pil_images_all.append(tensor_to_pil_images(input_images[i]))
        
        inputs_list = []
        for i in range(len(input_images)):
            inputs = processor_original(
                images=pil_images_all[i],
                text=[text_list[i]],
                return_tensors='pt'
            ).to(device)
            inputs_list.append(inputs)
        return inputs_list
    
    @torch.no_grad()
    def test_original_full():
        inputs_list = prepare_original_inputs()
        results = []
        for inputs in inputs_list:
            output = model_original(**inputs, output_hidden_states=True)
            results.append(output)
        return results
    
    # Warmup
    for _ in range(2):
        test_original_full()
    
    # Time original model
    torch.cuda.synchronize()
    start_time = time()
    for _ in range(num_runs):
        test_original_full()
    torch.cuda.synchronize()
    original_time = time() - start_time
    
    # Clear original model
    del model_original, processor_original
    clear_memory()
    
    # Test batched model
    print("Loading batched model...")
    model_batched = Qwen2_5_VLForConditionalGenerationBatched.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    processor_batched = Qwen2_5_VLProcessorBatched.from_pretrained(model_name)
    
    def prepare_batched_inputs():
        return processor_batched(
            images=input_images.flatten(0, 1),
            text=list(text_list),
            return_tensors='pt',
            padding=True
        ).to(device)
    
    @torch.no_grad()
    def test_batched_full():
        inputs = prepare_batched_inputs()
        return model_batched.batched_forward(
            input_ids_list=inputs['input_ids'][:, None],
            attention_mask_list=inputs['attention_mask'][:, None],
            pixel_values_list=inputs['pixel_values'].unflatten(0, (len(inputs['input_ids']), -1)),
            image_grid_thw_list=inputs['image_grid_thw'].unflatten(0, (len(inputs['input_ids']), -1))
        )
    
    # Warmup
    for _ in range(2):
        test_batched_full()
    
    # Time batched model
    torch.cuda.synchronize()
    start_time = time()
    for _ in range(num_runs):
        test_batched_full()
    torch.cuda.synchronize()
    batched_time = time() - start_time
    
    print(f"Original full model (sequential): {original_time:.4f}s for {num_runs} runs")
    print(f"Batched full model: {batched_time:.4f}s for {num_runs} runs")
    print(f"Speedup: {original_time / batched_time:.2f}x")
    print(f"Per-sample original: {original_time * 1000 / (num_runs * batch_size):.2f}ms")
    print(f"Per-sample batched: {batched_time * 1000 / (num_runs * batch_size):.2f}ms")
    
    del model_batched, processor_batched
    clear_memory()


def main():
    """Run all speed tests"""
    print("QWEN BATCHED MODEL SPEED BENCHMARK")
    print("=" * 60)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Run tests sequentially to avoid memory issues
    test_processor_speed(num_runs=50)
    test_visual_encoder_speed(num_runs=10)
    test_full_model_speed(num_runs=10)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()