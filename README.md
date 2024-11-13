# AMD GPU ML Guide

A community-driven collection of notebooks and guides for running Machine Learning workloads on AMD GPUs with ROCm.

## 📚 Overview

This repository aims to provide practical examples and guides for ML development on AMD GPUs. Whether you're setting up your first environment or optimizing advanced workloads, you'll find helpful resources here.

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

### Framework Guides
- [HuggingFace on ROCm](docs/frameworks/huggingface.md)
- [PyTorch Lightning](docs/frameworks/lightning.md)
- [Diffusers Library](docs/frameworks/diffusers.md)

## 📓 Notebooks

### Language Models
| Notebook | Description |
|----------|-------------|
| Basic-LLM-Inference | Running basic inference with Llama 2 and Mistral |
| Simple-Fine-Tuning | Basic fine-tuning with LoRA and QLoRA |
| Model-Quantization | Quantizing models for efficient inference |

### Image Generation
| Notebook | Description |
|----------|-------------|
| SD-Basic-Pipeline | Running Stable Diffusion with ROCm |
| ControlNet-Guide | Setting up ControlNet models |
| SDXL-Pipeline | Working with SDXL |

### General ML
| Notebook | Description |
|----------|-------------|
| Memory-Management | Understanding and optimizing GPU memory usage |
| Multi-GPU-Training | Distributed training setup |
| Debugging-Guide | Common issues and how to solve them |

## 🔍 Troubleshooting

Common issues and solutions for:
- ROCm installation
- Framework compatibility
- Memory management
- Model loading
- Training issues

## 🤝 Contributing

We welcome contributions of all kinds! Whether you want to:
- Add new notebooks
- Improve documentation
- Share troubleshooting tips
- Fix bugs

Check our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📖 Resources

### Official Documentation
- [ROCm Documentation](https://rocmdocs.amd.com/en/latest/)
- [PyTorch ROCm Guide](https://pytorch.org/docs/stable/notes/hip.html)
- [AMD ML Resources](https://www.amd.com/en/graphics/servers-solutions-rocm)

### Community Links
- [ROCm GitHub](https://github.com/RadeonOpenCompute/ROCm)
- [PyTorch Discussion Forums](https://discuss.pytorch.org/)

## ⚖️ License

MIT License - See [LICENSE](LICENSE) for details.

---

🌟 **Remember**: This is a community resource - if you find something unclear or have improvements to suggest, please open an issue or PR!