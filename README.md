# RSR-core

**RSR (Redundant Segment Reduction)** algorithm.

Reference: [UIC-InDeXLab/RSR](https://github.com/UIC-InDeXLab/RSR)

## Structure

```
RSR-core/
├── poc/            # Python proof-of-concept implementation
├── kernels/
│   ├── cpu/        # CPU kernels (C/C++)
│   └── cuda/       # CUDA GPU kernels
├── tests/          # Unit and integration tests
└── benchmarks/     # Performance benchmarks
```

## Benchmark Results

### Matrix-Vector Multiplication

#### CPU:

#### CUDA:

### Ternary (1.58bit) LLMs

Speedup is computed from `Avg Time` against the `HF bfloat16` baseline for the same model.

#### CPU 🖥️
| Model | HF Time | RSR (ours) Time | HF Tok/s | RSR (ours) Tok/s | Speedup vs HF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Falcon3-10B-Instruct-1.58bit | 351.215s | **5.663s** | 0.2 | **11.3** | **62.0x** |
| Llama3-8B-1.58-100B-tokens | 261.557s | **4.862s** | 0.2 | **13.4** | **53.8x** |
| bitnet-b1.58-2B-4T-bf16 | 31.446s | **2.258s** | 2.1 | **28.8** | **13.9x** |
| bitnet-b1.58-2B-4T | 4.582s | **2.221s** | 14.2 | **29.3** | **2.1x** |

#### CUDA ⚡
| Model | HF Time | RSR (ours) Time | HF Tok/s | RSR (ours) Tok/s | Speedup vs HF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Falcon3-10B-Instruct-1.58bit | 2.536s | **1.351s** | 25.2 | **47.4** | **1.9x** |
| Llama3-8B-1.58-100B-tokens | 2.035s | **1.097s** | 31.9 | **59.3** | **1.9x** |
| bitnet-b1.58-2B-4T-bf16 | 1.966s | **1.133s** | 33.1 | **57.4** | **1.7x** |
| bitnet-b1.58-2B-4T | 1.563s | **1.139s** | 41.6 | **57.1** | **1.4x** |
