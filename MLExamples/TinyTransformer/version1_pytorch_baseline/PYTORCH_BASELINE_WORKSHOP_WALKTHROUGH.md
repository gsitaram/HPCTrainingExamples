# TinyTransformer Baseline Workshop Guide

The main reference for this directory is [`README.md`](README.md). This note keeps the same material in a shorter lab order.

## Preparation

Load the required modules:

```bash
module load pytorch rocm
```

Use the default case unless there is a reason to change it:

```bash
python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 10
```

From one validated run, the baseline reference numbers were:

- `291.3 samples/sec`
- `27.5 ms` average batch time
- `434.3 MB` peak memory

## Exercise 1: Establish the baseline

Run the model once and record:

- average training speed
- average batch time
- peak memory usage

Those are the reference numbers for the later TinyTransformer versions.

## Exercise 2: Collect a quick hotspot list

Run:

```bash
./get_hotspots.sh
```

Record the top three kernels by total time. In the validated run, the top entries were GEMM-heavy kernels around `30.8 ms`, `30.1 ms`, and `26.6 ms` of total GPU time.

## Exercise 3: Collect a runtime trace

Run:

```bash
./get_trace.sh
```

Open the resulting `.pftrace` file in Perfetto:

```text
https://ui.perfetto.dev/
```

Identify:

- host launches
- forward-pass kernels
- backward-pass kernels
- visible synchronization points

## Exercise 4: Collect the full kernel trace

Run:

```bash
./get_counters.sh
```

Record:

- total GPU time
- dispatch count
- top kernels by time

If the result is a ROCm 7.x database, summarize it with:

```bash
rocpd2csv -i <db_file> -o kernel_stats.csv
rocpd summary -i <db_file> --region-categories KERNEL
```

## Exercise 5: Hardware metrics

Run:

```bash
./get_rocprof_compute.sh
```

On supported Instinct GPUs, use the printed `rocprof-compute analyze` sequence. On unsupported GPUs, the script exits cleanly and you can continue with the trace-based exercises.

Questions to answer:

- does the dominant dispatch look memory bound or compute bound
- is occupancy likely to matter
- does the report agree with the hotspot list

## Exercise 6: Optional system trace

Run:

```bash
./get_rocprof_sys.sh
```

This script uses a smaller default step count than the other profiling scripts. Open the generated `.proto` file in Perfetto and use it when the interaction between Python, libraries, and GPU execution matters more than kernel timing alone.

## Exercise 7: Compare with the next version

Move to `../version2_pytorch_fused` and repeat the same sequence. The comparison is more useful than any single run in isolation.

## Closing note

If only a short session is available, Exercises 1 through 4 are enough. That gives a complete path from baseline run to hotspot list to runtime trace to full kernel trace.
