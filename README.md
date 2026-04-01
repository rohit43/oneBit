# Bonsai 1-Bit LLM on Apple Silicon

Running [PrismML's Bonsai 8B](https://huggingface.co/prism-ml/Bonsai-8B-mlx-1bit) — a 1-bit language model — locally on Mac using MLX.

## Setup

Requires macOS with Apple Silicon and full Xcode (not just Command Line Tools).

```bash
# Install MLX with 1-bit kernel support (PrismML fork)
pip install mlx-lm
pip install "mlx @ git+https://github.com/PrismML-Eng/mlx.git@prism"
```

If you hit Metal shader errors, make sure Xcode is selected and Metal Toolchain is installed:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcodebuild -downloadComponent MetalToolchain
```

## Scripts

| Script | What it does |
|--------|-------------|
| `01_download.py` | Downloads the model from HuggingFace |
| `02_architecture.py` | Prints full model architecture |
| `03_inspect.py` | Shows model stats — params, layers, quantization |
| `04_generate.py` | Basic text generation (no streaming) |
| `05_stream.py` | Streaming generation with tok/s |

## Model Details

- **Model**: Bonsai-8B-mlx-1bit (Qwen3-8B architecture)
- **Quantization**: 1-bit g128 (1.25 bits per weight)
- **Size on disk**: ~1.3 GB (down from 16 GB at FP16)
- **Parameters**: 8.19B original, stored in 1-bit
- **Context**: 65,536 tokens

## Links

- [HuggingFace Collection](https://huggingface.co/collections/prism-ml/bonsai)
- [PrismML](https://prismml.com)
- [Whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf)
- [MLX Fork (1-bit kernels)](https://github.com/PrismML-Eng/mlx)
