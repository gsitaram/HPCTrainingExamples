# ML Example: PyTorch Micro-Benchmarking with ROCm Profiling

In this example we consider a compact PyTorch workload that is useful for learning the ROCm profiling tools on a model that is small enough to run quickly, but large enough to produce non-trivial GPU activity. The driver runs forward and backward passes for common CNN architectures and reports throughput in images per second. The scripts in this directory use `resnet50`, batch size `64`, and `10` iterations so that the outputs from the different profilers can be compared on the same workload.

The purpose of the directory is straightforward. We begin with one reproducible benchmark run, then examine the same execution with a timeline trace, a kernel summary, a hardware counter report, and a system trace. In that sense, the example is meant to be read and run in the same spirit as the GhostExchange materials: one workload, a small number of commands, and a clear progression from run to analysis.

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
- `get_counters.sh`: collect a kernel trace and kernel summary data with `rocprofv3`
- `get_rocprof_compute.sh`: collect hardware counter reports with `rocprof-compute`
- `get_rocprof_sys.sh`: collect a system trace with `rocprof-sys`

We recommend using them in the order listed above. The runtime trace shows the overall execution flow. The kernel trace identifies the dominant GPU kernels. The compute report is most useful once there is a narrower question about occupancy, memory traffic, or arithmetic intensity.

## Running the benchmark

Load the required modules:

```bash
module load pytorch rocm
```

Run a baseline case:

```bash
python micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 10
```

Representative output from one run is shown below:

```text
INFO: running forward and backward for warmup.
INFO: running the benchmark..
OK: finished running benchmark..
--------------------SUMMARY--------------------------
Microbenchmark for network : resnet50
Num devices: 1
Dtype: FP32
Mini batch size [img] : 64
Time per mini-batch : 0.177
Throughput [img/sec] : 356.09
```

The main quantity to record from this run is the throughput. For profiling, it is also useful to note the problem size and whether `torch.compile` or `--fp16 1` was enabled.

## Runtime trace with `get_trace.sh`

Run the script:

```bash
./get_trace.sh
```

The script writes a timestamped directory under `profiling_results/trace_*`. On ROCm 6.x and 7.x it requests Perfetto output directly, so the main file to look for is a `.pftrace` file. Open it in Perfetto:

```text
https://ui.perfetto.dev/
```

When reading the trace, the first questions to ask are:

- where the host spends time between launches
- whether GPU kernels run back-to-back or with visible gaps
- how much explicit memory traffic appears relative to compute work
- whether synchronization points serialize the execution

On systems that expose more than one GPU agent, `rocprofv3` may print a warning about an unsupported secondary agent before the trace starts. In the container validation on an RX 7900 XTX system, that warning did not prevent generation of the `.pftrace` file.

If a ROCm 7.x database is generated instead of a Perfetto trace, convert it with:

```bash
rocpd2pftrace -i <db_file> -o trace.pftrace
```

## Kernel trace with `get_counters.sh`

Run the script:

```bash
./get_counters.sh
```

The script writes to `profiling_results/counters_*`. On ROCm 6.x the main output is usually a CSV file. On ROCm 7.x the output is typically a SQLite database. For ROCm 7.x, the two most useful follow-up commands are:

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

## Hardware metrics with `get_rocprof_compute.sh`

Run the script:

```bash
./get_rocprof_compute.sh
```

The script writes a timestamped workload directory under `profiling_results/rocprof_compute_*`. The command printed at the end of the run is the command to use for report generation. In general it has the form

```bash
rocprof-compute analyze \
    -p profiling_results/rocprof_compute_<timestamp>/workloads/<workload_name>/rocprof \
    --dispatch <N> \
    -n resnet50_dispatch
```

This step is most useful after the runtime trace and kernel summary have identified a small set of kernels worth studying. The report can then be used to decide whether the dominant kernels appear to be limited by arithmetic throughput, memory bandwidth, or occupancy.

`rocprof-compute` has the best counter coverage on Instinct class GPUs. On consumer GPUs some counters may be unavailable.

On the RX 7900 XTX container used for validation, `rocprof-compute` did not start collection and reported `Cannot find a supported arch in rocminfo`. For that reason, this step should be treated as optional unless the tutorial is being run on a supported Instinct GPU. The script in this directory now exits early with a short explanatory message when it detects an unsupported architecture.

## System trace with `get_rocprof_sys.sh`

Run the script:

```bash
./get_rocprof_sys.sh
```

The script writes to `profiling_results/rocprof_sys_*`. Open the resulting `.proto` file in Perfetto:

```text
https://ui.perfetto.dev/
```

This tool is useful when the question is broader than kernel timing alone, for example when the interaction between the Python runtime, libraries, and the GPU execution needs to be examined. If the run produces excessive memory map output or is otherwise noisy on a given system, use `get_trace.sh` first and return to `rocprof-sys` only if the higher-level system view is necessary.

In the container validation run, `rocprof-sys` printed warnings about `perf_event_paranoid=4` and an `RSMI_STATUS_UNEXPECTED_DATA` exception before continuing. The run still completed and generated a usable Perfetto trace. The script now prints the exact `.proto` path so that the file can be opened directly.

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

For distributed runs, set `<ngpu>` to the number of visible GPUs on the system. The container validation for this tutorial used a single discrete GPU, so the multi-GPU example was not exercised there.

For `--compile`, use a larger iteration count if the goal is steady-state performance rather than functionality. In the validated container run, a `10`-iteration compiled case was dominated by compile overhead and therefore ran much slower than the non-compiled baseline.

## Performance note

On systems that use MIOpen, it can be useful to allow the library to tune and cache convolution choices before comparing results:

```bash
export MIOPEN_FIND_ENFORCE=3
python micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 10
```

## Additional resources

- rocprofv3: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html
- rocpd tools: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocpd-output-format.html
- Perfetto UI: https://ui.perfetto.dev/
