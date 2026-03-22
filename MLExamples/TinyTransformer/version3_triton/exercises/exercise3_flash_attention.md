## Exercise 3: Flash Attention Implementation

**Objective**: Master memory-efficient attention patterns and Flash Attention in Triton.

**Time**: 75 minutes | **Prerequisites**: Completed Exercises 1 and 2

### Background

Flash Attention reduces memory complexity from O(N²) to O(N) using tiling and online statistics, enabling significant speedups for long sequences.

### Part A: Algorithm Understanding

Examine `flash_attention_kernel` in `tiny_llama_v3.py`:

```python
@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    batch_size, num_heads, seq_len, head_dim,
    scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
```

**Key Questions:**
1. How does tiling achieve O(N) memory complexity?
2. Why subtract the maximum before exponentiation?
3. How are running statistics updated correctly?

### Part B: Performance Analysis

Benchmark configurations:
- (1, 8, 128, 64), (2, 16, 256, 64), (4, 32, 512, 64)
- (2, 16, 1024, 64), (1, 8, 2048, 64)

Compare Flash Attention vs standard PyTorch attention:
- Execution time
- Memory usage
- Speedup and memory reduction

### Part C: Block Size Optimization

Test block sizes: (32,32), (64,64), (128,128), (64,32), (32,64), (128,64)

### Results Template

| Sequence Length | Flash (ms) | Standard (ms) | Speedup | Memory Reduction |
|----------------|------------|---------------|---------|------------------|
| 128 | | | | |
| 512 | | | | |
| 1024 | | | | |

### Troubleshooting

- **Kernel Compilation**: Check dimension compatibility and block size limits
- **Performance Regression**: Verify block sizes are optimal for sequence length
- **Numerical Instability**: Monitor overflow in softmax, check running statistics

### Resources

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- Online softmax algorithm
