# PyTorch Micro-Benchmark Notes

This file collects a few technical notes that are useful when varying the default benchmark case described in `README.md`.

## Mixed precision and compilation

Mixed precision can be enabled with:

```bash
python micro_benchmarking_pytorch.py --network densenet121 --batch-size 64 --iterations 10 --fp16 1
```

Compilation can be enabled with:

```bash
python micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 10 --compile
```

For short runs, the one-time compile cost may dominate the reported timing, so a compiled case may appear slower than the eager baseline even when the steady-state behavior is better. When the goal is steady-state comparison, use a larger iteration count.

Additional compile options may be passed through `--compileContext`, for example:

```bash
python micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 64 \
    --iterations 20 \
    --compile \
    --compileContext "{'mode': 'max-autotune', 'fullgraph': 'True'}"
```

## MIOpen tuning

On systems that use MIOpen, it can be useful to allow the library to tune and cache its convolution choices before comparing results:

```bash
export MIOPEN_FIND_ENFORCE=3
python micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 10
```

The first run may spend additional time building the performance database. Subsequent runs are then more meaningful for comparison.

## PyTorch profiler options

The script also supports framework-level profiling:

```bash
python micro_benchmarking_pytorch.py \
    --network densenet121 \
    --batch-size 2048 \
    --compile \
    --fp16 1 \
    --kineto \
    --iterations 10
```

This path is useful when the goal is to correlate Python-level and operator-level behavior before moving to ROCm tools.

For ROCTX correlation with ROCm profilers, use:

```bash
python micro_benchmarking_pytorch.py \
    --network densenet121 \
    --batch-size 2048 \
    --compile \
    --fp16 1 \
    --autograd_profiler
```

## DeepSpeed FLOPS profiling

If DeepSpeed is available, the benchmark can also be run with FLOPS profiling:

```bash
python micro_benchmarking_pytorch.py \
    --network densenet121 \
    --batch-size 2048 \
    --fp16 1 \
    --flops-prof-step 10 \
    --iterations 20
```

This mode is useful when the question is about model-level efficiency rather than kernel-level execution.

## Multi-GPU runs

For distributed cases, `--batch-size` is the global batch size across all ranks. For example:

```bash
torchrun --nproc-per-node <ngpu> micro_benchmarking_pytorch.py --network densenet121 --batch-size 2048 --compile --fp16 1
```

Each rank processes `batch-size / <ngpu>` samples. When comparing distributed results, it is important to keep that interpretation in mind.
