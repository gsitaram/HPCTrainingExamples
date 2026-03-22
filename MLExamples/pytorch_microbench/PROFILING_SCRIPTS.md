# PyTorch Micro-Benchmark Profiling Scripts

The `README.md` file in this directory is the primary walkthrough and the only full tutorial. This note is only a short reference to the profiling scripts and their outputs.

## Default workload

Unless modified, the scripts profile the following command:

```bash
python micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 10
```

The scripts can be retargeted without editing them:

- `PYTORCH_MICROBENCH_NETWORK`: override the model name
- `PYTORCH_MICROBENCH_BATCH_SIZE`: override the batch size
- `PYTORCH_MICROBENCH_ITERATIONS`: override the iteration count
- `PYTORCH_MICROBENCH_EXTRA_ARGS`: append simple benchmark flags such as `--fp16 1` or `--compile`
- `PYTORCH_MICROBENCH_OUTPUT_ROOT`: write results under a different root directory
- `PYTORCH_MICROBENCH_PYTHON`: select a specific Python executable

Example:

```bash
PYTORCH_MICROBENCH_NETWORK=densenet121 \
PYTORCH_MICROBENCH_EXTRA_ARGS="--fp16 1" \
./get_trace.sh
```

## Script summary

| Script | Tool | Main output | Primary use |
|--------|------|-------------|-------------|
| `get_trace.sh` | `rocprofv3 --runtime-trace` | `profiling_results/trace_*` | Timeline view of host activity, kernel launches, and memory traffic |
| `get_gpu_hotspots.sh` | `rocprofv3 --kernel-trace` | `profiling_results/gpu_hotspots_*` | Kernel counts, total GPU time, and hotspot identification |
| `get_performance_metrics.sh` | `rocprof-compute profile` | `profiling_results/performance_metrics_*` | Hardware counter analysis for selected dispatches |
| `get_rocprof_sys.sh` | `rocprof-sys-run --profile --trace` | `profiling_results/rocprof_sys_*` | System-level view in Perfetto |

In a typical first pass through the example, `get_trace.sh`, `get_gpu_hotspots.sh`, and `get_rocprof_sys.sh` should produce the expected trace or summary outputs. If hardware-counter collection is unsupported on the local system, `get_performance_metrics.sh` exits early with a short explanation.

## ROCm 7.x note

For ROCm 7.x, `get_gpu_hotspots.sh` commonly produces a SQLite database rather than a CSV file. Two useful follow-up commands are:

```bash
rocpd2csv -i <db_file> -o kernel_stats.csv
rocpd summary -i <db_file> --region-categories KERNEL
```

For `get_trace.sh`, if a database is produced instead of a `.pftrace` file, convert it with:

```bash
rocpd2pftrace -i <db_file> -o trace.pftrace
```

## `rocprof-compute` note

The performance-metrics script follows the same pattern used in the `rocprof-compute` training examples in this repository:

1. collect a workload
2. list kernels and dispatches
3. analyze a selected dispatch with targeted metric blocks

The first post-processing command should therefore be:

```bash
rocprof-compute analyze -p <profile_dir> --list-stats
```

After selecting a dispatch, two useful analysis commands are:

```bash
rocprof-compute analyze -p <profile_dir> --dispatch <N> --block 2.1.15 6.2.7
rocprof-compute analyze -p <profile_dir> --dispatch <N> --block 16.1 17.1
```

Counter availability is best on supported Instinct class GPUs. Other systems may expose only a subset of the metrics, or no supported counter collection path at all.

When counter collection is unsupported on the local system, the script reports that condition explicitly and exits without attempting collection.

The script accepts an optional mode argument:

- `no-roof`: default tutorial mode; collect counters only and skip the roofline stage
- `full`: collect the default counters and roofline data
- `roof-only`: collect roofline data only

## Recommended order

For a first pass through the example, we suggest:

1. `get_trace.sh`
2. `get_gpu_hotspots.sh`
3. `get_performance_metrics.sh`
4. `get_rocprof_sys.sh`
