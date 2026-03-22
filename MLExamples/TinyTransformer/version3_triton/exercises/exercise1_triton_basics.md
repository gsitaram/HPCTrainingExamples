## Exercise 1: Understanding Triton Kernel Basics

**Objective**: Learn Triton GPU programming fundamentals and analyze basic kernel performance.

**Time**: 45 minutes | **Prerequisites**: Completed Version 1 and Version 2 exercises

### Part A: Kernel Structure Analysis

Examine the `rmsnorm_kernel` in `tiny_llama_v3.py`:

```python
@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
```

**Questions:**
1. How does Triton handle memory pointers compared to CUDA?
2. What is the role of `BLOCK_SIZE`?
3. Why are `eps` and `BLOCK_SIZE` marked as `tl.constexpr`?

### Part B: Performance Profiling

Run the Triton profiling script:

```bash
cd version3_triton/
python3 run_triton_profiling.py
```

Run ROCProfiler analysis:

```bash
./run_rocprof_triton.sh
cat rocprof_results/triton_analysis_summary.md
```

### Part C: Block Size Optimization

Experiment with block sizes (64, 128, 256, 512, 1024) and measure:
- Execution time
- Memory transactions
- GPU occupancy

### Results Template

| Metric | Triton RMSNorm | PyTorch RMSNorm | Speedup |
|--------|----------------|------------------|---------|
| Execution Time (ms) | | | |
| Memory Usage (MB) | | | |

### Common Issues

- **Compilation Errors**: Check tensor shapes and constexpr values
- **Performance Regression**: Verify block size tuning and proper warmup
- **Numerical Differences**: Small FP precision differences are normal

### Resources

- [Triton Documentation](https://triton-lang.org/main/index.html)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
