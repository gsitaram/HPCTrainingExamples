# ML Example: TinyTransformer PyTorch SDPA with ROCm Profiling

In this version we keep the fused structure developed in the later TinyTransformer examples, but replace the custom attention path with PyTorch scaled dot product attention. The directory is useful for comparing a framework-provided attention implementation against the more custom Triton path in version 3 while keeping the rest of the workflow largely unchanged.

## Changes relative to version 3

This version uses:

- PyTorch SDPA for the attention path
- the same general model structure and profiling workflow as the later fused versions
- the same ROCm scripts used to compare traces, kernel summaries, and hardware reports

The comparison in [`../VERSION_COMPARISON.md`](../VERSION_COMPARISON.md) shows that versions 3 and 4 are similar in throughput and memory use for the repository test case. The value of version 4 is therefore not only raw performance. It also shows how much of the optimized behavior can be retained while relying on a framework-maintained attention path.

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

## Running the SDPA version

Load the required modules:

```bash
module load pytorch rocm triton
```

Run a short case:

```bash
python tiny_llama_v4.py --batch-size 8 --seq-len 128 --num-steps 10
```

This run is best interpreted together with a version 3 run on the same system and with the same problem size.

## Runtime trace with `get_trace.sh`

Run:

```bash
./get_trace.sh
```

Open the generated `.pftrace` file in Perfetto:

```text
https://ui.perfetto.dev/
```

The main comparison against version 3 is whether the attention region looks materially different even when the overall step time remains similar.

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

The most useful comparison points are:

- the kernels that dominate the attention portion of the step
- total dispatch count versus version 3
- whether the dominant time shifts toward framework kernels or remains in a small number of heavy kernels

## Hardware metrics with `get_rocprof_compute.sh`

Run:

```bash
./get_rocprof_compute.sh
```

Then analyze one of the dominant dispatches:

```bash
rocprof-compute analyze \
    -p rocprof_compute/profile_<timestamp>/workloads/<workload_name>/rocprof \
    --dispatch <N> \
    -n tiny_llama_dispatch
```

This report is most useful when the question is whether the SDPA-based path changes the limiting factor of the dominant kernels.

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

This is a convenient first pass when the goal is to compare the dominant kernels in versions 3 and 4 before collecting larger traces.

## Additional resources

- comparison across versions: [`../VERSION_COMPARISON.md`](../VERSION_COMPARISON.md)
- PyTorch SDPA overview: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Perfetto UI: https://ui.perfetto.dev/
