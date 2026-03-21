# TinyTransformer Baseline Workshop Guide

The main reference for this directory is the `README.md` file. This note arranges the same material as a short lab sequence that can be run in a single session.

## Preparation

Load the required modules:

```bash
module load pytorch rocm
```

Use the default case from the profiling scripts unless there is a reason to change it:

```bash
python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 10
```

## Exercise 1: Establish the baseline

Run the model once and record:

- average time per step
- throughput
- reported memory use

These numbers are the reference point for the later TinyTransformer versions.

## Exercise 2: Use the PyTorch profiler

Collect a short framework-level profile:

```bash
python tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 20 \
    --enable-pytorch-profiler \
    --profile-dir ./pytorch_profiles \
    --profile-steps 5
```

Open the result with TensorBoard:

```bash
tensorboard --logdir ./pytorch_profiles --port 6006
```

This step is useful for understanding the operator-level view before moving to ROCm tools.

## Exercise 3: Collect a runtime trace

Run:

```bash
./get_trace.sh
```

Open the resulting `.pftrace` file in Perfetto:

```text
https://ui.perfetto.dev/
```

Identify the broad structure of one training step:

- host launches
- forward-pass kernels
- backward-pass kernels
- synchronization events

## Exercise 4: Identify hotspot kernels

Run:

```bash
./get_counters.sh
```

If the result is a ROCm 7.x database, summarize it with:

```bash
rocpd2csv -i <db_file> -o kernel_stats.csv
rocpd summary -i <db_file> --region-categories KERNEL
```

Record:

- total GPU time
- number of dispatches
- top three kernels by time

The goal here is to establish what the baseline spends time on before any fusion is introduced.

## Exercise 5: Hardware metrics

Run:

```bash
./get_rocprof_compute.sh
```

Then generate a report for one heavy dispatch:

```bash
rocprof-compute analyze \
    -p rocprof_compute/profile_<timestamp>/workloads/<workload_name>/rocprof \
    --dispatch <N> \
    -n tiny_llama_dispatch
```

Questions to answer:

- does the kernel appear memory bound or compute bound
- is occupancy a likely concern
- does the report agree with the hotspot list from Exercise 4

## Exercise 6: Compare with the next version

After the baseline has been characterized, move to `../version2_pytorch_fused` and repeat the same sequence. The comparison is more useful than any single run in isolation.

## Closing remark

If only a short session is available, Exercises 1 through 4 are sufficient. They provide a complete path from baseline run to trace to hotspot identification.
