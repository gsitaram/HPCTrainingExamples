# Exercise 3: Bottleneck Identification and Optimization Planning

exercise_3_bottleneck_identification.md from `HPCTrainingExamples/MLExamples/TinyTransformer/version1_pytorch_baseline/exercises` in the Training Examples repository.

## Objective

Systematically identify performance bottlenecks in the baseline model and create an optimization roadmap.

## Step 1: Comprehensive Profiling

Run the complete profiling suite:

```
python tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 30 \
    --enable-pytorch-profiler \
    --profile-dir ./bottleneck_analysis
```

## Step 2: Operator-Level Analysis

Examine the profiling results to identify computational bottlenecks. Look for the top time-consuming operations in the profiler output.

Expected top operations by GPU time:
- Matrix multiplications (aten::mm, aten::addmm, aten::bmm)
- Softmax operations
- Element-wise operations

## Step 3: Efficiency Analysis

Key efficiency metrics to examine:
- Model FLOPS Utilization (MFU)
- Memory bandwidth utilization
- Kernel launch overhead

Typical baseline efficiency:
- MFU: 20-35% (memory-bound workload)
- Bandwidth utilization: 30-60%

## Typical Bottleneck Hierarchy

1. **Attention Operations (35-45% of time)**
   - QKV projections (3 separate kernel launches)
   - Attention score computation
   - Softmax operations

2. **Feed-Forward Network (25-35% of time)**
   - Gate and up projections
   - SwiLU activation
   - Down projection

3. **Kernel Launch Overhead (10-20% of time)**
   - Multiple small operations
   - Memory transfers between kernels

## Optimization Roadmap

**Priority 1: Kernel Fusion (Expected 1.4-1.8x speedup)**
- QKV Fusion: Combine Q, K, V projections into single GEMM
- SwiGLU Fusion: Combine gate and up projections

**Priority 2: Attention Optimization (Expected 1.3-2.0x speedup)**
- Flash Attention: Memory-efficient attention computation
- Reduces memory from O(S^2) to O(S)

**Priority 3: Additional Optimizations (Expected 1.1-1.3x speedup)**
- torch.compile for automatic kernel fusion
- Mixed precision (FP16/BF16)

## Troubleshooting

Missing analysis files:

```
python tiny_llama_v1.py --batch-size 8 --profile-dir ./bottleneck_retry
```

Check GPU status:

```
rocm-smi
```
