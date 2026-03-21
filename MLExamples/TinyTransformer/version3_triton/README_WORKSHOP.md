# TinyTransformer Triton Workshop Guide

The main reference for this directory is the `README.md` file. This note keeps a short exercise sequence for a training session focused on the Triton version.

## Preparation

Load the required modules:

```bash
module load pytorch rocm triton
```

Run a short case:

```bash
python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 10
```

Record the throughput and the reported memory use.

## Exercise 1: Compare against the baseline

Before profiling version 3 in isolation, place its throughput and memory side by side with the numbers from `../version1_pytorch_baseline`. The comparison is the main point of the exercise.

## Exercise 2: Hotspot list

Collect a fast hotspot summary:

```bash
./get_hotspots.sh
```

Use this run to identify the kernels that dominate time and to confirm that the kernel set is more concentrated than in the baseline.

## Exercise 3: Runtime trace

Collect a runtime trace:

```bash
./get_trace.sh
```

Open the resulting `.pftrace` file in Perfetto:

```text
https://ui.perfetto.dev/
```

Compare the trace with version 1 and ask:

- does the step consist of fewer, heavier kernels
- is the attention region easier to recognize
- are there fewer visible gaps between launches

## Exercise 4: Kernel trace

Collect a kernel trace:

```bash
./get_counters.sh
```

If needed, summarize a ROCm 7.x database with:

```bash
rocpd2csv -i <db_file> -o kernel_stats.csv
rocpd summary -i <db_file> --region-categories KERNEL
```

Record:

- dispatch count
- number of unique kernels
- top three kernels by time

## Exercise 5: Performance debugging path

If time permits, run the staged exercise:

```bash
cd exercises/performance_debugging
./run_all_stages.sh
```

This exercise is useful because it shows that the final performance comes from a sequence of correctness and layout fixes, not from a single change.

## Closing remark

Version 3 is often the clearest point in the tutorial sequence to discuss why kernel specialization changes both performance and profiler output. For a short lab, Exercises 1 through 4 are sufficient.
