# ML Example: TinyTransformer Fused with ROCm Profiling

In this version we keep the baseline model structure, but introduce a first round of fusion through framework-level mechanisms. This directory is useful as an intermediate case between the plain PyTorch baseline and the later Triton-based versions. It shows what changes in the traces and hotspot lists when some operations are fused, even if the end-to-end speedup is still modest.

## Changes relative to version 1

This version is written to expose the following optimizations when supported by the software stack:

- fused Q, K, and V projection path
- fused or memory-efficient attention path
- fused SwiGLU path
- `torch.compile`-driven graph and kernel fusion

The repository comparison in [`../VERSION_COMPARISON.md`](../VERSION_COMPARISON.md) shows that version 2 changes the kernel mix more than the end-to-end timing. That is precisely what makes it useful as a teaching step.

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

## Running the fused version

Load the required modules:

```bash
module load pytorch rocm
```

Run a short case:

```bash
python tiny_llama_v2.py --batch-size 8 --seq-len 128 --num-steps 10
```

The key comparison is not the absolute time alone. It is the difference between the kernel mix seen here and the one seen in version 1.

## Runtime trace with `get_trace.sh`

Run:

```bash
./get_trace.sh
```

Open the generated `.pftrace` file in Perfetto:

```text
https://ui.perfetto.dev/
```

Compare the trace with version 1 and look for:

- fewer short-lived kernels
- reduced launch fragmentation
- any visible change in the attention region of the step

## Kernel trace with `get_counters.sh`

Run:

```bash
./get_counters.sh
```

For ROCm 7.x, summarize the resulting database with:

```bash
rocpd2csv -i <db_file> -o kernel_stats.csv
rocpd summary -i <db_file> --region-categories KERNEL
```

The important comparison against version 1 is:

- dispatch count
- number of unique kernels
- whether the dominant kernels become more concentrated

## Hardware metrics with `get_rocprof_compute.sh`

Run:

```bash
./get_rocprof_compute.sh
```

Then analyze one heavy dispatch:

```bash
rocprof-compute analyze \
    -p rocprof_compute/profile_<timestamp>/workloads/<workload_name>/rocprof \
    --dispatch <N> \
    -n tiny_llama_dispatch
```

The main question is whether the fused path has shifted the dominant cost or merely rearranged it.

## System trace with `get_rocprof_sys.sh`

Run:

```bash
./get_rocprof_sys.sh
```

Open the resulting `.proto` file in Perfetto when a broader system view is needed.

## Hotspot summary with `get_hotspots.sh`

Run:

```bash
./get_hotspots.sh
```

This is a convenient first pass when the goal is simply to see which kernels account for most of the GPU time before collecting larger traces.

## Additional resources

- comparison across versions: [`../VERSION_COMPARISON.md`](../VERSION_COMPARISON.md)
- rocprofv3: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html
- Perfetto UI: https://ui.perfetto.dev/
