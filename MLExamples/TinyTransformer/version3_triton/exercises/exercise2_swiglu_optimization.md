## Exercise 2: SwiGLU Kernel Optimization

**Objective**: Master advanced Triton kernel development through SwiGLU optimization.

**Time**: 60 minutes | **Prerequisites**: Completed Exercise 1

### Background

SwiGLU (Swish-Gated Linear Unit) combines gate projection with SiLU activation, up projection, element-wise multiplication, and down projection. Our Triton kernel fuses the gate and up projections with activation.

### Part A: SwiGLU Kernel Analysis

Examine `swiglu_kernel` in `tiny_llama_v3.py`:

```python
@triton.jit
def swiglu_kernel(
    x_ptr, gate_weight_ptr, up_weight_ptr, output_ptr,
    batch_size, seq_len, d_model, d_ff,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
```

**Questions:**
1. Why does this kernel use three different block sizes?
2. How are tensors laid out in memory?
3. What is the arithmetic intensity?

### Part B: Performance Optimization

Test different block size combinations:
- (1, 1, 32), (1, 1, 64), (1, 1, 128)
- (1, 2, 64), (2, 1, 64), (1, 1, 256)

### Part C: Arithmetic Intensity Analysis

Calculate for batch_size=4, seq_len=512, d_model=2048:
- Total FLOPs (gate + up projections + activation)
- Total memory traffic
- Arithmetic intensity (FLOPs/byte)

Determine if kernel is compute-bound or memory-bound using roofline analysis.

### Results Template

| Configuration | Time (ms) | Speedup vs PyTorch | Bandwidth (GB/s) |
|---------------|-----------|-------------------|------------------|
| Block Size (1,1,64) | | | |
| Block Size (1,1,128) | | | |

### Key Findings

1. **Optimal Block Size**: _____
2. **Memory Layout Impact**: _____
3. **Performance Bottleneck**: _____ (compute/memory)

### Resources

- Arithmetic intensity and roofline model concepts
- Memory coalescing patterns for multi-dimensional data
