# Performance Debugging Exercise: V3 Triton Optimization Journey

## Overview

This exercise demonstrates systematic debugging and optimization for V3 Triton kernels:

1. **Diagnose incorrect model behavior** (wrong loss values)
2. **Fix correctness issues** (weight initialization)
3. **Profile and identify performance bottlenecks**
4. **Systematically optimize for performance**

## Exercise Progression

### Stage 1: Broken Loss (`v3_stage1_broken_loss.py`)
**Problem:** Loss = 942 instead of ~7
**Root Cause:** Missing weight initialization

```bash
python v3_stage1_broken_loss.py --batch-size 8 --seq-len 128 --num-steps 20
```

### Stage 2: Fixed Loss, Terrible Performance (`v3_stage2_slow_performance.py`)
**Problem:** Loss fixed (7.0) but only 15.2 samples/sec (vs V1's 97 samples/sec)
**Root Cause:** Non-contiguous tensors after `repeat_interleave` for GQA

```bash
python v3_stage2_slow_performance.py --batch-size 8 --seq-len 128 --num-steps 20
```

### Stage 3: Better Performance, Wrong Timing (`v3_stage3_fake_timing.py`)
**Problem:** Improved to 310 samples/sec but timing breakdown is wrong
**Root Cause:** Missing CUDA synchronization for timing

```bash
python v3_stage3_fake_timing.py --batch-size 8 --seq-len 128 --num-steps 20
```

### Stage 4: Accurate Timing, Slow Kernels (`v3_stage4_slow_kernels.py`)
**Problem:** Forward pass is 25.5ms (2.4x slower than V1's 10.8ms)
**Root Cause:** Inefficient Triton SwiGLU kernel doing manual matrix multiplication

```bash
python v3_stage4_slow_kernels.py --batch-size 8 --seq-len 128 --num-steps 20
```

### Stage 5: Final Optimized (`../tiny_llama_v3.py`)
**Solution:** Use PyTorch for matrix multiplies, Triton only for element-wise fusion
**Result:** 2065 samples/sec (5.5x faster than V1!)

```bash
cd .. && python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 20
```

## Summary Table

| Stage | Loss | Speed (samples/sec) | Issue | Fix |
|-------|------|---------------------|-------|-----|
| 1 | 942 | N/A | Missing weight init | Add `_init_weights()` |
| 2 | 7.0 | 15.2 | Non-contiguous tensors | Add `.contiguous()` |
| 3 | 7.0 | 310.8 | Wrong timing | Add CUDA sync |
| 4 | 7.0 | 305.9 | Slow Triton SwiGLU | Use PyTorch matmul |
| 5 | 7.0 | 2065.0 | **OPTIMIZED!** | Hybrid approach |

**Baseline (V1):** 372.9 samples/sec | **Final Speedup:** 5.5x faster, 46% less memory

## Key Learnings

1. **Correctness First**: Validate loss/accuracy before optimizing
2. **Tensor Contiguity**: Always `.contiguous()` before Triton kernels
3. **Accurate Timing**: Use `torch.cuda.synchronize()` for GPU timing
4. **Hybrid Approach**: Triton for memory-bound ops, PyTorch BLAS for matrix ops

## Profiling Commands

```bash
# Basic profiling
rocprof --stats python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 20

# Detailed kernel trace
rocprofv2 --kernel-trace -o trace.json python tiny_llama_v3.py ...
# View at https://ui.perfetto.dev
```

## Resources

- [Triton Documentation](https://triton-lang.org/)
- [ROCm Profiling Guide](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
