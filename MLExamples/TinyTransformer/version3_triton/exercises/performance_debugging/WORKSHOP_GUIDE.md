# V3 Performance Debugging Workshop Guide

## Quick Start

```bash
cd version3_triton/exercises/performance_debugging

# Read the comprehensive guide
cat README.md

# Run all stages with automatic profiling
./run_all_stages.sh
```

## What This Exercise Teaches

### 1. Correctness Before Performance (Stage 1)
Missing weight init → Loss 942 → 7.0 after one-line fix

### 2. Memory Layout Matters (Stage 2→3)
Non-contiguous tensors → 20x speedup with `.contiguous()`

### 3. Measure Accurately (Stage 3→4)
GPU ops are async → `torch.cuda.synchronize()` required

### 4. Know When NOT to Use Custom Kernels (Stage 4→5)
Triton SwiGLU 2M+ threads → PyTorch rocBLAS 8x faster

### 5. Hybrid Optimization Wins
Final: 2065 samples/sec (5.5x faster than V1!)

## Key Takeaways

| Stage | Loss | Speed | vs Baseline | Key Issue |
|-------|------|-------|-------------|-----------|
| 1 | 942 | N/A | N/A | No weight init |
| 2 | 7.0 | 15 samp/s | 0.04x | Non-contig tensors |
| 3 | 7.0 | 311 samp/s | 0.83x | Wrong timing |
| 4 | 7.0 | 306 samp/s | 0.82x | Slow SwiGLU |
| 5 | 7.0 | **2065 samp/s** | **5.5x** | **OPTIMAL** |

**Baseline (V1):** 372.9 samples/sec, 522.3 MB

## Profiling Commands

```bash
rocprof --stats python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 20
rocprofv2 --kernel-trace -o trace.json python tiny_llama_v3.py ...
```

## Resources

- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [ROCm Profiling](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
