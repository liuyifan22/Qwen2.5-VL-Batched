# Qwen 2.5 VL Batched Implementation

Having a headache for Qwen2.5-VL's non-batched implementation?

We provide a high-performance batched implementation of Qwen 2.5 Vision-Language model for efficient multi-image, multi-sample inference.

## Overview

This project provides an optimized batched version of the Qwen 2.5 VL model that enables efficient processing of multiple images and text prompts simultaneously. The implementation focuses on maximizing GPU utilization and reducing inference latency for computer vision applications requiring multi-camera inputs. A typical use case includes extracting feature from large number of images with the Qwen2.5VL visual encoder.

## Features

- 🚀 **Batched Processing**: Process multiple samples and cameras in a single forward pass
- 🖥️ **Flow Optimized**: Avoid frequent transfer between CPU and GPU
- 🔧 **Modular Implementation**: Either of the tree modules can be utilized freely
- 📊 **Numerical Fidelity**: Experimented and guaranteed close numerical output with origianl model


## Core Components

- **Tensor Processor** (`qwen_batched/model/tensor_processor.py`): Batched preprocessing of images and text INSIDE the GPU memory, avoiding CPU-GPU transfer overhead.
- **Batched Vision Transformer** (`qwen_batched/model/modeling_qwen2_5_vl_batched.py`): Optimized ViT implementation which processes batched images in parallel. Reducing computation from O(N^2) to O(N).
- **Batched Decoder** (`qwen_batched/model/modeling_qwen2_5_vl_batched.py`): Efficient language model decoder which accepts batched inputs and generates features in parallel.
- **Unified Model** (`qwen_batched/qwen_batched_VL_model.py`): High-level API wrapper, easy to use.


## Tested Environment

We do experiments on the following configurations:
- Python 3.10
- CUDA 12.4
- pytorch==2.5.1
- transformers==4.51.1

## Quick Start

### Input and Output
The batched implementation accepts inputs in the following format:
- **Images**: A tensor of shape `(B, n_cameras, 3, H, W)` where `B` is the batch size, `n_cameras` is the number of cameras, and `(H, W)` is the image resolution.
- **Texts**: A list of strings, where each string corresponds to a text prompt for each sample in the batch.

The output is a list of hidden states from the model, which can be used for downstream tasks like feature extraction or text generation. You may easily modify this to return the logits or generated text for your application.

### Run Example Class
To get a quick usage of the batched implementation, you can run the example script provided in this repository.

```bash
git clone https://github.com/liuyifan22/qwen_batched.git
cd qwen_batched
python example.py
```

You can also go to `./qwen_batched/qwen_batched_VL_model.py` to see the model architecture.

### Test Deviation
We tested the numerical deviation of the batched implementation against the original Qwen 2.5 VL model. The results show that the batched version produces outputs within a small margin of error (Mean rel error: 0.000439), ensuring high fidelity in feature extraction and text generation tasks. We did experiments and found that the error comes from:
- The processor produces small noise due to operation order difference.
- The noise is amplified by the large number of layers of ViT and LLM decoder.

You can run the following command to see the numerical deviation of the batched implementation against the original Qwen 2.5 VL model.

```bash
bash test.sh
```




## Contributing

We welcome contributions! Please contact us via GitHub issues or pull requests.



## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you find our project useful, please cite:

```bibtex
@misc{qwen_batched_2025,
  title={Qwen 2.5 VL Batched Implementation},
  author={Yifan Liu and Nikolaos Gkanatsios},
  year={2025},
  url={https://github.com/liuyifan22/qwen_batched}
}
```

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen2-VL) for the original implementation
- [HuggingFace Transformers](https://github.com/huggingface/transformers) for the model framework

---

**Note**: This is an unofficial implementation. For production use, please validate thoroughly against your specific requirements.