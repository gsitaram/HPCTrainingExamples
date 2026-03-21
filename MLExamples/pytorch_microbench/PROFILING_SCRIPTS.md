# PyTorch Micro-Benchmark Profiling Scripts

The `README.md` file in this directory is the primary tutorial. This note is only a short reference to the profiling scripts and their outputs.

## Default workload

Unless modified, the scripts profile the following command:

```bash
python micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 10
```

## Script summary

| Script | Tool | Main output | Primary use |
|--------|------|-------------|-------------|
| `get_trace.sh` | `rocprofv3 --runtime-trace` | `profiling_results/trace_*` | Timeline view of host activity, kernel launches, and memory traffic |
| `get_counters.sh` | `rocprofv3 --kernel-trace` | `profiling_results/counters_*` | Kernel counts, total GPU time, and hotspot identification |
| `get_rocprof_compute.sh` | `rocprof-compute profile` | `profiling_results/rocprof_compute_*` | Hardware counter analysis for selected dispatches |
| `get_rocprof_sys.sh` | `rocprof-sys-run --profile --trace` | `profiling_results/rocprof_sys_*` | System-level view in Perfetto |

## ROCm 7.x note

For ROCm 7.x, `get_counters.sh` commonly produces a SQLite database rather than a CSV file. Two useful follow-up commands are:

```bash
rocpd2csv -i <db_file> -o kernel_stats.csv
rocpd summary -i <db_file> --region-categories KERNEL
```

For `get_trace.sh`, if a database is produced instead of a `.pftrace` file, convert it with:

```bash
rocpd2pftrace -i <db_file> -o trace.pftrace
```

## `rocprof-compute` note

The `rocprof-compute` script prints the analysis command to use at the end of the run. In general it has the form:

```bash
rocprof-compute analyze \
    -p profiling_results/rocprof_compute_<timestamp>/workloads/<workload_name>/rocprof \
    --dispatch <N> \
    -n resnet50_dispatch
```

Counter availability is best on Instinct class GPUs. Consumer GPUs may expose only a subset of the metrics.

In the validated RX 7900 XTX container environment, `rocprof-compute` did not support the detected `gfx1100` device. The script now reports that case explicitly and exits without attempting collection.

## Recommended order

For a first pass through the example, we suggest:

1. `get_trace.sh`
2. `get_counters.sh`
3. `get_rocprof_compute.sh`
4. `get_rocprof_sys.sh`
