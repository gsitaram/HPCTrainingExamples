# ML Example: TinyTransformer Baseline with ROCm Profiling

In this version we consider a baseline PyTorch implementation of a small decoder-only transformer. This is the reference point for the optimized versions in the directory. The model is intentionally modest in size so that full training runs and profiler traces can be collected without introducing unnecessary complexity.

## Features of this version

- plain PyTorch implementation of the model and training loop
- configurable batch size, sequence length, hidden dimension, and layer count
- optional PyTorch profiler and DeepSpeed FLOPS profiler hooks in the Python driver
- ROCm profiling scripts for runtime traces, kernel traces, hardware metrics, and system traces

## Overview of the model

The main command-line arguments are:

- `--batch-size <N>`: batch size for training
- `--seq-len <N>`: sequence length
- `--num-steps <N>`: number of training steps
- `--hidden-dim <N>`: hidden dimension
- `--num-layers <N>`: number of transformer layers
- `--num-heads <N>`: number of attention heads
- `--learning-rate <float>`: learning rate
- `--use-amp`: enable automatic mixed precision
- `--enable-pytorch-profiler`: enable the PyTorch profiler
- `--enable-deepspeed-flops`: enable DeepSpeed FLOPS profiling

This version is the one to profile first because it establishes the kernel mix and memory behavior before any fusion or custom kernels are introduced.

## Running the baseline

Load the required modules:

```bash
module load pytorch rocm
```

Run a short baseline case:

```bash
python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 10
```

The main quantities to record are the average time per step, the throughput, and the reported memory use. These are the reference numbers to compare with the later versions.

## Runtime trace with `get_trace.sh`

Run the script:

```bash
./get_trace.sh
```

The script writes a timestamped directory under `traces/trace_*`. Open the generated `.pftrace` file in Perfetto:

```text
https://ui.perfetto.dev/
```

At this stage it is useful to identify the basic structure of one training step:

- host-side launch activity
- forward kernels
- backward kernels
- synchronization points

If a ROCm 7.x database is produced instead of a Perfetto trace, convert it with:

```bash
rocpd2pftrace -i <db_file> -o trace.pftrace
```

## Kernel trace with `get_counters.sh`

Run the script:

```bash
./get_counters.sh
```

The script writes to `counters/counter_*`. On ROCm 7.x the output is typically a SQLite database. Two useful follow-up commands are:

```bash
rocpd2csv -i <db_file> -o kernel_stats.csv
rocpd summary -i <db_file> --region-categories KERNEL
```

For the baseline version, the first quantities to inspect are:

- total GPU time
- number of kernel dispatches
- number of unique kernels
- the kernels that dominate the forward and backward passes

Those quantities become more informative once the later versions are compared against them.

## Hardware metrics with `get_rocprof_compute.sh`

Run the script:

```bash
./get_rocprof_compute.sh
```

The script writes to `rocprof_compute/profile_*`. The report generation step has the form:

```bash
rocprof-compute analyze \
    -p rocprof_compute/profile_<timestamp>/workloads/<workload_name>/rocprof \
    --dispatch <N> \
    -n tiny_llama_dispatch
```

This step is most useful after the kernel trace has identified a dispatch worth studying in more detail.

## System trace with `get_rocprof_sys.sh`

Run the script:

```bash
./get_rocprof_sys.sh
```

The script writes to `rocprof_sys/profile_*`. Open the resulting `.proto` file in Perfetto:

```text
https://ui.perfetto.dev/
```

This view is helpful when the interaction between Python, libraries, and GPU execution matters more than kernel timing alone.

## Optional framework-level profiling

The Python driver also exposes framework-level instrumentation. For example:

```bash
python tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 20 \
    --enable-pytorch-profiler \
    --profile-dir ./pytorch_profiles \
    --profile-steps 5
```

The resulting trace can be viewed with TensorBoard:

```bash
tensorboard --logdir ./pytorch_profiles --port 6006
```

A short exercise sequence for this directory is given in [`PYTORCH_BASELINE_WORKSHOP_WALKTHROUGH.md`](PYTORCH_BASELINE_WORKSHOP_WALKTHROUGH.md).

## Additional resources

- rocprofv3: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html
- rocpd tools: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocpd-output-format.html
- Perfetto UI: https://ui.perfetto.dev/
