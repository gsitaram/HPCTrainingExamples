# ML Example: PyTorch Micro-Benchmarking with ROCm Profiling

In this example we consider a compact PyTorch workload that is useful for learning the ROCm profiling tools on a model that is small enough to run quickly, but large enough to produce non-trivial GPU activity. The driver runs forward and backward passes for common CNN architectures and reports throughput in images per second. The scripts in this directory use `resnet50`, batch size `64`, and `10` iterations so that the outputs from the different profilers can be compared on the same workload.

The purpose of the directory is straightforward. We begin with one reproducible benchmark run, then examine the same execution with a timeline trace, a kernel summary, a hardware counter report, and a system trace. In that sense, the example is meant to be read and run in the same spirit as the GhostExchange materials: one workload, a small number of commands, and a clear progression from run to analysis.

This `README.md` file is the primary walkthrough for the directory. The other markdown files are short reference notes and optional training checklists, not separate full tutorials.

## Overview of the benchmark

The benchmark is controlled with the following arguments:

- `--network <name>`: network to benchmark, for example `resnet50`, `resnet101`, `densenet121`, `vgg16`, or `alexnet`
- `--batch-size <N>`: global mini-batch size
- `--iterations <N>`: number of timed iterations
- `--fp16 <0|1>`: enable mixed precision when supported
- `--compile`: enable `torch.compile`
- `--compileContext <dict>`: pass compile options as a Python dictionary string
- `--distributed_dataparallel`: run with distributed data parallel
- `--device_ids <ids>`: comma-separated GPU ids for distributed runs

## Profiling scripts in this directory

The directory contains four short profiling scripts:

- `get_trace.sh`: collect a runtime trace with `rocprofv3`
- `get_gpu_hotspots.sh`: collect a kernel trace and hotspot summary with `rocprofv3`
- `get_performance_metrics.sh`: collect hardware counter reports with `rocprof-compute`
- `get_rocprof_sys.sh`: collect a system trace with `rocprof-sys`

We recommend using them in the order listed above. The runtime trace shows the overall execution flow. The kernel trace identifies the dominant GPU kernels. The compute report is most useful once there is a narrower question about occupancy, memory traffic, or arithmetic intensity.

All four scripts use the same default workload, but they can be retargeted without editing the files. The common overrides are `PYTORCH_MICROBENCH_NETWORK`, `PYTORCH_MICROBENCH_BATCH_SIZE`, `PYTORCH_MICROBENCH_ITERATIONS`, `PYTORCH_MICROBENCH_EXTRA_ARGS`, and `PYTORCH_MICROBENCH_OUTPUT_ROOT`. For example, `PYTORCH_MICROBENCH_EXTRA_ARGS="--fp16 1" ./get_trace.sh` profiles the default trace workflow with mixed precision enabled.

## Running the benchmark

Load the required modules:

```bash
module load pytorch rocm
```

Run a baseline case:

```bash
python micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 10
```

An example output from one run is shown below. The exact timing values depend on the model, GPU, ROCm version, and whether relevant caches are already warm:

```text
INFO: running forward and backward for warmup.
INFO: running the benchmark..
OK: finished running benchmark..
--------------------SUMMARY--------------------------
Microbenchmark for network : resnet50
Num devices: 1
Dtype: FP32
Mini batch size [img] : 64
Time per mini-batch : 0.1770334005355835
Throughput [img/sec] : 361.51370197024534
```

The main quantity to record from this run is the throughput. For profiling, it is also useful to note the problem size and whether `torch.compile` or `--fp16 1` was enabled. The values should be treated as measurements for the current system, not as targets that should match across devices.

The plot below was generated from fresh container runs with `generate_example_plots.py`, using the same commands shown in this README.

![pytorch_microbench example measurements from validated container runs](images/pytorch_microbench_example_runs.png)

## Runtime trace with `get_trace.sh`

Run the script:

```bash
./get_trace.sh
```

The script writes a timestamped directory under `profiling_results/trace_*`. On ROCm 6.x and 7.x it requests Perfetto output directly, so the main file to look for is a `.pftrace` file. Open it in Perfetto:

```text
https://ui.perfetto.dev/
```

A successful run prints the generated trace path explicitly, for example:

```text
Perfetto trace file: profiling_results/trace_20260321_231808/<hostname>/19455_results.pftrace
Open it in Perfetto UI: https://ui.perfetto.dev/
```

When reading the trace, the first questions to ask are:

- where the host spends time between launches
- whether GPU kernels run back-to-back or with visible gaps
- how much explicit memory traffic appears relative to compute work
- whether synchronization points serialize the execution

On systems that expose more than one agent, `rocprofv3` may print warnings about one of the agents before the trace starts. In many cases the trace is still produced successfully, so the first check should be whether the expected output file was generated.

If a ROCm 7.x database is generated instead of a Perfetto trace, convert it with:

```bash
rocpd2pftrace -i <db_file> -o trace.pftrace
```

## GPU hotspots with `get_gpu_hotspots.sh`

Run the script:

```bash
./get_gpu_hotspots.sh
```

The script writes to `profiling_results/gpu_hotspots_*`. On ROCm 6.x the main output is usually a CSV file. On ROCm 7.x the output is typically a SQLite database. For ROCm 7.x, the two most useful follow-up commands are:

```bash
rocpd2csv -i <db_file> -o kernel_stats.csv
rocpd summary -i <db_file> --region-categories KERNEL
```

For this benchmark, the quantities that usually matter first are:

- total GPU time
- number of kernel dispatches
- number of unique kernels
- the few kernels that dominate the total time

For `resnet50`, the dominant entries are often convolution, batch normalization, and elementwise kernels from MIOpen and PyTorch. The exact names vary across hardware and ROCm versions, but the methodology does not.

One example kernel summary from this workflow produced the following dominant entries:

- `miopenSp3AsmConv_v30_3_1_gfx11_fp32_f2x3_stride1`: `763.126806 ms`
- `MIOpenBatchNormBwdSpatial`: `167.792579 ms`
- `ATen vectorized elementwise kernel`: `120.853175 ms`

The next plot was generated from the `get_gpu_hotspots.sh` run that produced the example summary above.

![pytorch_microbench GPU hotspots from validated container run](images/pytorch_microbench_gpu_hotspots.png)

## Performance metrics with `get_performance_metrics.sh`

Run the script in its default mode:

```bash
./get_performance_metrics.sh
```

The script writes a timestamped workload directory under `profiling_results/performance_metrics_*`. The default tutorial mode uses `--no-roof`, which keeps the run focused on detailed counter collection. Use the follow-up analysis commands to inspect specific dispatches and metric blocks.

Treat this as a short workshop sequence:

```bash
rocprof-compute analyze -p <profile_dir> --list-stats
rocprof-compute analyze -p <profile_dir> --dispatch <N>
rocprof-compute analyze -p <profile_dir> --dispatch <N> --block 2.1.15 6.2.7
rocprof-compute analyze -p <profile_dir> --dispatch <N> --block 16.1 17.1
```

Use `--list-stats` to find the kernel and dispatch to study, then inspect that dispatch in the default report. The `2.1.15 6.2.7` blocks are useful for occupancy and LDS-related limits. The `16.1 17.1` blocks are useful for L1 and L2 speed-of-light metrics.

This step is most useful after `get_trace.sh` and `get_gpu_hotspots.sh` have identified a kernel worth studying.

The script also supports explicit modes:

```bash
./get_performance_metrics.sh
./get_performance_metrics.sh full
./get_performance_metrics.sh roof-only
```

The default mode is `no-roof`. Use `full` when the roofline stage is needed, and use `roof-only` when the immediate question is where the kernel falls on the roofline.

`rocprof-compute` has the best counter coverage on supported Instinct class GPUs. On other systems, some counters may be unavailable, or the collection path may be unsupported. In that case the script exits early with a short explanation, so this step should be treated as optional unless the tutorial is running on a system with supported hardware-counter collection.

One example of that skip path is:

```text
Skipping rocprof-compute profiling for pytorch_microbench...
Detected GPU architecture: gfx1100
rocprof-compute hardware-counter collection currently requires a supported Instinct GPU
Use get_trace.sh and get_gpu_hotspots.sh on this system instead.
```

## System trace with `get_rocprof_sys.sh`

Run the script:

```bash
./get_rocprof_sys.sh
```

The script writes to `profiling_results/rocprof_sys_*`. Open the resulting `.proto` file in Perfetto:

```text
https://ui.perfetto.dev/
```

A successful run prints the trace file directly, for example:

```text
Perfetto trace file: profiling_results/rocprof_sys_20260321_231923/rocprofsys-python-output/<timestamp>/perfetto-trace-19832.proto
Open it in Perfetto UI: https://ui.perfetto.dev/
```

This tool is useful when the question is broader than kernel timing alone, for example when the interaction between the Python runtime, libraries, and the GPU execution needs to be examined.

On some systems, `rocprof-sys` may print warnings related to host performance-counter permissions or device telemetry before continuing. The important check is whether the run completes and produces the expected `.proto` file. The script prints that path explicitly so that the file can be opened directly.

## Variations to try

Once the baseline case has been examined, the following variations are reasonable next steps:

- change the network, for example `--network densenet121` or `--network vgg16`
- enable mixed precision with `--fp16 1`
- enable compilation with `--compile`
- run a distributed case with `torchrun`

For example:

```bash
python micro_benchmarking_pytorch.py --network densenet121 --batch-size 64 --iterations 10 --fp16 1
python micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 10 --compile
torchrun --nproc-per-node <ngpu> micro_benchmarking_pytorch.py --network resnet50 --batch-size 128
```

For distributed runs, set `<ngpu>` to the number of visible GPUs on the system. If only one GPU is available, omit the distributed example and stay with the single-device path.

`--fp16 1` and `--compile` are useful follow-up comparisons, but the direction and magnitude of the change will depend on the system and workload. For `--compile`, use a larger iteration count if the goal is steady-state performance rather than functionality; with only `10` iterations, startup effects may still influence the result.

Example outputs from two such follow-up runs are shown below:

```text
$ python micro_benchmarking_pytorch.py --network densenet121 --batch-size 64 --iterations 10 --fp16 1
INFO: running forward and backward for warmup.
INFO: running the benchmark..
OK: finished running benchmark..
--------------------SUMMARY--------------------------
Microbenchmark for network : densenet121
Num devices: 1
Dtype: FP16
Mini batch size [img] : 64
Time per mini-batch : 0.1000108003616333
Throughput [img/sec] : 639.9308851502005

$ python micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 10 --compile
INFO: running forward and backward for warmup.
INFO: running the benchmark..
OK: finished running benchmark..
--------------------SUMMARY--------------------------
Microbenchmark for network : resnet50
Num devices: 1
Dtype: FP32
Mini batch size [img] : 64
Time per mini-batch : 0.1676210880279541
Throughput [img/sec] : 381.8135340424872
```

## Performance note

On systems that use MIOpen, it can be useful to allow the library to tune and cache convolution choices before comparing results:

```bash
export MIOPEN_FIND_ENFORCE=3
python micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 10
```

## Additional resources

- [`generate_example_plots.py`](generate_example_plots.py): regenerates the example plots from container logs
- rocprofv3: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html
- rocpd tools: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocpd-output-format.html
- Perfetto UI: https://ui.perfetto.dev/
