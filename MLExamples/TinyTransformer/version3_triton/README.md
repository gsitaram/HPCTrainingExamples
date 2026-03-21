# ML Example: TinyTransformer Triton with ROCm Profiling

In this version we replace several frequently executed operations with custom Triton kernels. This is the first stage in the progression where the kernel mix changes substantially and the reduction in memory use becomes pronounced. For that reason, version 3 is often the most instructive comparison against the baseline.

## Changes relative to version 2

This version introduces:

- Triton RMSNorm kernels
- Triton attention kernels
- a hybrid SwiGLU path that combines framework kernels and specialized code
- implementation choices aimed at reducing launch count and intermediate memory traffic

The repository comparison in [`../VERSION_COMPARISON.md`](../VERSION_COMPARISON.md) shows that this version reduces dispatch count, total GPU time, and peak memory relative to version 1 by a substantial margin.

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

## Running the Triton version

Load the required modules:

```bash
module load pytorch rocm triton
```

Run a short case:

```bash
python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 10
```

For this version, it is useful to compare not only throughput but also kernel count and memory use against versions 1 and 2.

## Runtime trace with `get_trace.sh`

Run:

```bash
./get_trace.sh
```

Open the generated `.pftrace` file in Perfetto:

```text
https://ui.perfetto.dev/
```

Compared with the earlier versions, the main questions are:

- whether the step is composed of fewer, heavier kernels
- whether the attention region is easier to isolate in the trace
- whether host-side launch overhead has become less visible

## Kernel trace with `get_counters.sh`

Run:

```bash
./get_counters.sh
```

For ROCm 7.x, summarize the database with:

```bash
rocpd2csv -i <db_file> -o kernel_stats.csv
rocpd summary -i <db_file> --region-categories KERNEL
```

This version is a good place to compare:

- dispatch count versus version 1
- concentration of time in the top kernels
- whether Triton kernels now appear among the dominant entries

## Hardware metrics with `get_rocprof_compute.sh`

Run:

```bash
./get_rocprof_compute.sh
```

Then analyze a dispatch of interest:

```bash
rocprof-compute analyze \
    -p rocprof_compute/profile_<timestamp>/workloads/<workload_name>/rocprof \
    --dispatch <N> \
    -n tiny_llama_dispatch
```

At this stage the report is especially useful because the set of important kernels is smaller than in the baseline.

## System trace with `get_rocprof_sys.sh`

Run:

```bash
./get_rocprof_sys.sh
```

Use the system trace when the interaction between Python, Triton compilation, and GPU execution needs to be studied at a broader level.

## Hotspot summary with `get_hotspots.sh`

Run:

```bash
./get_hotspots.sh
```

This is often the quickest way to confirm that the dominant kernels have changed in the expected direction before collecting larger traces.

## Workshop note

A short companion exercise sequence is given in [`README_WORKSHOP.md`](README_WORKSHOP.md). The performance-debugging exercise under [`exercises/performance_debugging`](exercises/performance_debugging) is also useful when the goal is to understand how the final optimized path was reached.

## Additional resources

- comparison across versions: [`../VERSION_COMPARISON.md`](../VERSION_COMPARISON.md)
- Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/index.html
- Perfetto UI: https://ui.perfetto.dev/
