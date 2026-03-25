# RSR-core

**RSR (Redundant Segment Reduction)** algorithm.

Reference: [UIC-InDeXLab/RSR](https://github.com/UIC-InDeXLab/RSR)

## Installation

**Prerequisites:** Python >= 3.10, a C compiler (for CPU kernels), and optionally CUDA for GPU support.

```bash
git clone https://github.com/UIC-InDeXLab/RSR-Core.git
cd RSR-Core
pip install -e .
```

## Structure

```
RSR-core/
├── multiplier/             # Python wrappers for kernels
│   ├── bit_1/              # 1-bit (binary) multipliers (CPU/CUDA)
│   └── bit_1_58/           # 1.58-bit (ternary) multipliers (CPU/CUDA)
├── kernels/                # Low-level C/CUDA kernel source
│   ├── bit_1/
│   │   ├── cpu/            #   C kernels
│   │   └── cuda/           #   CUDA kernels (.cu)
│   └── bit_1_58/
│       ├── cpu/            #   C kernels
│       └── cuda/           #   CUDA kernels (.cu)
├── integrations/           # Model integrations
│   └── hf/                 #   HuggingFace integration
├── benchmarking/           # Benchmarking scripts & results
└── tests/                  # Unit and integration tests
```


## Demo

<!-- <p align="center">
  <a href="assets/rsr_baseline_compare.mp4">
    <img src="assets/rsr_baseline_compare.webp" alt="Comparison of the Hugging Face baseline and RSR inference on 1.58-bit LLM inference. Click to open the MP4 version." width="900" />
  </a>
</p> -->

[![RSR vs Baseline](assets/rsr_baseline_compare.webp)](https://drive.google.com/file/d/1ub-MITJUepmfBLkyUZFb50hbJsuhgwCH/view?usp=sharing)

## Benchmark Results

### Matrix-Vector Multiplication

#### CPU 🖥️

| 1-bit | 1.58-bit |
|:---:|:---:|
| ![1-bit CPU](assets/cpu_bit_1.png) | ![1.58-bit CPU](assets/cpu_bit_1_58.png) |

#### CUDA ⚡

| 1-bit | 1.58-bit |
|:---:|:---:|
| ![1-bit CUDA](assets/cuda_bit_1.png) | ![1.58-bit CUDA](assets/cuda_bit_1_58.png) |

### Ternary (1.58bit) LLMs

Speedup is computed against the HuggingFace `bfloat16` baseline for the same model.

#### CPU 🖥️

| Model | HF Tok/s | RSR Tok/s | Speedup |
| :--- | ---: | ---: | ---: |
| Falcon3-10B-Instruct-1.58bit | 0.2 | **11.3** | **62.0x** |
| Llama3-8B-1.58-100B-tokens | 0.2 | **13.4** | **53.8x** |
| bitnet-b1.58-2B-4T-bf16 | 2.1 | **28.8** | **13.9x** |
| bitnet-b1.58-2B-4T | 14.2 | **29.3** | **2.1x** |

#### CUDA ⚡

| Model | HF Tok/s | RSR Tok/s | Speedup |
| :--- | ---: | ---: | ---: |
| Falcon3-10B-Instruct-1.58bit | 25.2 | **47.4** | **1.9x** |
| Llama3-8B-1.58-100B-tokens | 31.9 | **59.3** | **1.9x** |
| bitnet-b1.58-2B-4T-bf16 | 33.1 | **57.4** | **1.7x** |
| bitnet-b1.58-2B-4T | 41.6 | **57.1** | **1.4x** |
