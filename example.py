import torch
from qwen_batched.qwen_batched_VL_model import QwenBatchedVLModel

device = torch.device("cuda")
model = QwenBatchedVLModel().to(device)

# dummy inputs
images = torch.rand(2, 8, 3, 224, 224, device=device)  # B=2, ncam=8
texts = [
    "What do you see?",
    "Describe this."
]

# get logits / generated token IDs
outputs = model(images, texts, max_length=100)
print(outputs.shape)  # e.g. (2, seq_len, vocab_size) or token IDs