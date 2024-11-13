# AMD GPU ML Guide

A community-driven collection of notebooks and guides for running Machine Learning workloads on AMD GPUs with ROCm.

## 📚 Overview

This repository provides practical examples and guides for ML development on AMD GPUs. Whether you're setting up your first environment or optimizing advanced workloads, you'll find helpful resources here.

## 🔧 Getting Started

### Basic Setup
1. [ROCm Installation Guide](docs/setup/rocm-install.md)
   - System requirements
   - Installation steps
   - Common issues and solutions

2. [PyTorch Setup](docs/setup/pytorch-setup.md)
   - Installing PyTorch with ROCm support
   - Testing your installation
   - Environment setup

### Quick Start
```bash
# Install ROCm (Ubuntu 22.04)
wget https://repo.radeon.com/amdgpu-install/5.7/ubuntu/focal/amdgpu-install_5.7.50700-1_all.deb
sudo apt install ./amdgpu-install_5.7.50700-1_all.deb
sudo amdgpu-install --usecase=hiplibsdk,rocm,pytorch

# Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Verify Installation
python3 -c "import torch; print(torch.cuda.is_available())"
```

## 📓 Notebooks

### Language Models
| Notebook | Description |
|----------|-------------|
| [Basic-LLM-Inference](notebooks/llm/basic_inference.ipynb) | Running basic inference with Llama 2 and Mistral |
| [Simple-Fine-Tuning](notebooks/llm/simple_finetune.ipynb) | Basic fine-tuning with LoRA and QLoRA |
| [Model-Quantization](notebooks/llm/quantization.ipynb) | Quantizing models for efficient inference |

### Image Generation
| Notebook | Description |
|----------|-------------|
| [SD-Basic-Pipeline](notebooks/diffusion/basic_pipeline.ipynb) | Running Stable Diffusion with ROCm |
| [ControlNet-Guide](notebooks/diffusion/controlnet.ipynb) | Setting up ControlNet models |
| [SDXL-Pipeline](notebooks/diffusion/sdxl.ipynb) | Working with SDXL |

### General ML
| Notebook | Description |
|----------|-------------|
| [Memory-Management](notebooks/general/memory.ipynb) | Understanding and optimizing GPU memory usage |
| [Multi-GPU-Training](notebooks/general/multi_gpu.ipynb) | Distributed training setup |
| [Debugging-Guide](notebooks/general/debugging.ipynb) | Common issues and how to solve them |

## 📊 AMD GPU Benchmarks

These benchmarks are performed on a Radeon 7900XTX (24GB VRAM) to help developers understand performance characteristics.

> **Software**: ROCm 6.0.2, PyTorch 2.2.1

### 🤖 LLM Inference

| Model | Batch Size | Context Length | Tokens/sec | Memory Usage | Notes |
|-------|------------|----------------|------------|--------------|-------|
| Llama-2-7B | 1 | 2048 | TBD | TBD | Regular fp16 inference |
| Llama-2-7B | 1 | 2048 | TBD | TBD | With 4-bit quantization |
| Mistral-7B | 1 | 2048 | TBD | TBD | Regular fp16 inference |
| Mixtral-8x7B | 1 | 2048 | TBD | TBD | MoE model performance |

### 🎨 Stable Diffusion

| Model | Resolution | Steps | Images/sec | Memory Usage | Notes |
|-------|------------|--------|------------|--------------|-------|
| SD 1.5 | 512x512 | 20 | TBD | TBD | Standard pipeline |
| SDXL | 1024x1024 | 20 | TBD | TBD | Base + Refiner |
| SD + ControlNet | 512x512 | 20 | TBD | TBD | With canny conditioning |

### 🚀 Training Benchmarks

| Task | Batch Size | Steps/sec | Memory Usage | Notes |
|------|------------|-----------|--------------|-------|
| LoRA (Llama-2-7B) | 1 | TBD | TBD | 8-bit base model |
| QLoRA (Llama-2-7B) | 2 | TBD | TBD | 4-bit base model |
| SD LoRA | 1 | TBD | TBD | 512x512 images |

### 💡 Performance Tips

1. **Memory Management**
   - Use `torch.cuda.empty_cache()` frequently
   - Monitor with `rocm-smi` for memory leaks
   - Enable compute preemption: `export HSA_ENABLE_SDMA=0`

2. **Optimization Tips**
   - Start with smaller batch sizes and increase gradually
   - Use gradient accumulation for larger effective batches
   - Monitor GPU utilization with `rocm-smi`

## 🔍 Troubleshooting

Common issues and solutions:

### ROCm Installation
```bash
# Check ROCm installation
rocminfo
rocm-smi

# Common fix for permission issues
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER
```

### PyTorch Issues
```python
# Verify GPU is detected
import torch
print(torch.version.hip)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a Pull Request

For more details, see [CONTRIBUTING.md](CONTRIBUTING.md)

## 📖 Resources

### Official Documentation
- [ROCm Documentation](https://rocmdocs.amd.com/en/latest/)
- [PyTorch ROCm Guide](https://pytorch.org/docs/stable/notes/hip.html)

### Community Links
- [ROCm GitHub](https://github.com/RadeonOpenCompute/ROCm)
- [PyTorch Forums](https://discuss.pytorch.org/)

## ⚖️ License

MIT License - See [LICENSE](LICENSE) for details.

---

🌟 **Remember**: This is a community resource - if you find something unclear or have improvements to suggest, please open an issue or PR!