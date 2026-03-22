# Exercise 2: Memory Analysis and Optimization

exercise_2_memory_analysis.md from `HPCTrainingExamples/MLExamples/TinyTransformer/version1_pytorch_baseline/exercises` in the Training Examples repository.

## Objective

Understand memory usage patterns, identify memory bottlenecks, and analyze memory bandwidth utilization.

## Step 1: Memory Profiling with Different Batch Sizes

```
python tiny_llama_v1.py \
    --batch-size 4 \
    --seq-len 128 \
    --num-steps 15 \
    --enable-pytorch-profiler \
    --profile-dir ./memory_analysis_bs4

python tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 15 \
    --enable-pytorch-profiler \
    --profile-dir ./memory_analysis_bs8

python tiny_llama_v1.py \
    --batch-size 16 \
    --seq-len 128 \
    --num-steps 15 \
    --enable-pytorch-profiler \
    --profile-dir ./memory_analysis_bs16
```

## Step 2: Memory Timeline Analysis

Launch TensorBoard for memory analysis:

```
tensorboard --logdir ./memory_analysis_bs8 --port 6007
```

In TensorBoard, navigate to the PROFILE tab and select Memory Timeline view.

## Step 3: Sequence Length Scaling

Test memory scaling with sequence length:

```
python tiny_llama_v1.py --batch-size 8 --seq-len 64 --num-steps 10
python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 10
python tiny_llama_v1.py --batch-size 8 --seq-len 256 --num-steps 10
python tiny_llama_v1.py --batch-size 4 --seq-len 512 --num-steps 5
```

## Expected Observations

**Memory Scaling:**
- Memory scales approximately linearly with batch size
- Memory scales quadratically with sequence length (due to attention matrices)

**Memory Hotspots:**
- Attention QKV matrices
- Attention score computation `[B, H, S, S]`
- FFN intermediate tensors

**Bandwidth Classification:**
- Arithmetic Intensity < 10 FLOPS/byte: Memory-bound
- Arithmetic Intensity 10-100 FLOPS/byte: Mixed workload
- Arithmetic Intensity > 100 FLOPS/byte: Compute-bound

## Optimization Targets

1. **Flash Attention**: Reduce attention memory from O(S^2) to O(S)
2. **Gradient Checkpointing**: Trade compute for memory
3. **Mixed Precision (FP16/BF16)**: 2x memory reduction
4. **Kernel Fusion**: Reduce intermediate tensor allocations

## Troubleshooting

Out of memory errors:

```
python tiny_llama_v1.py --batch-size 2 --seq-len 64
```

Memory fragmentation:

```
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```
