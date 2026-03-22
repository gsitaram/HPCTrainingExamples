# Exercise 1: Baseline Performance Analysis

exercise_1_baseline_analysis.md from `HPCTrainingExamples/MLExamples/TinyTransformer/version1_pytorch_baseline/exercises` in the Training Examples repository.

## Objective

Establish baseline performance metrics for Tiny LLaMA V1 and understand profiling methodology.

## Step 1: Run Baseline Training

```
cd version1_pytorch_baseline
python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 20
```

Expected output:
- Model configuration summary
- Training progress with loss values
- Performance metrics (samples/sec, memory usage)

## Step 2: Enable PyTorch Profiler

```
mkdir exercise1_profiles
python tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 20 \
    --enable-pytorch-profiler \
    --profile-dir ./exercise1_profiles
```

Profile files will be generated in `./exercise1_profiles/`.

## Step 3: Analyze Results

Launch TensorBoard to visualize profiling results:

```
tensorboard --logdir ./exercise1_profiles --port 6006
```

Alternatively, examine JSON traces directly:

```
ls -la ./exercise1_profiles/
```

## Key Observations

Typical baseline performance characteristics:
- Training speed: 50-100 samples/sec (varies by hardware)
- GPU utilization: 60-75%
- Memory usage: 2-4 GB depending on batch size
- Kernel count: 40-50 different kernel launches per step

## Optimization Opportunities

Based on profiling analysis:
- Attention operations consume ~40% of total compute time
- Matrix multiplications (GEMM) are the dominant kernels
- Multiple small operations create kernel launch overhead
- Memory allocation patterns show optimization opportunities

## Troubleshooting

CUDA/ROCm memory errors:

```
python tiny_llama_v1.py --batch-size 4 --seq-len 64 --num-steps 10
```

Check GPU utilization:

```
rocm-smi
```
