# TinyTransformer Triton Workshop Guide

The main reference for this directory is [`README.md`](README.md). This note keeps the Triton version in a short lab order.

## Preparation

Load the required modules:

```bash
module load pytorch rocm triton
```

Run:

```bash
python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 10
```

From one validated run, the reference numbers were:

- `829.9 samples/sec`
- `9.6 ms` average batch time
- `193.8 MB` peak memory

## Exercise 1: Compare against the baseline

Place the version 3 numbers next to the version 1 baseline before you start profiling. That comparison is the main point of the exercise.

## Exercise 2: Hotspot list

Run:

```bash
./get_hotspots.sh
```

Use this to check whether the dominant kernel set is smaller and more concentrated than in the baseline.

## Exercise 3: Runtime trace

Run:

```bash
./get_trace.sh
```

Open the `.pftrace` file in Perfetto:

```text
https://ui.perfetto.dev/
```

Ask:

- does the step consist of fewer, heavier kernels
- is the attention region easier to recognize
- are there fewer visible launch gaps

## Exercise 4: Full kernel trace

Run:

```bash
./get_counters.sh
```

Record:

- dispatch count
- number of unique kernels
- top kernels by total time

## Exercise 5: Performance debugging path

If time permits, run the staged exercise:

```bash
cd exercises/performance_debugging
./run_all_stages.sh
```

This is useful because it shows that the final performance comes from a sequence of implementation changes, not from a single switch.

## Closing note

Version 3 is usually the clearest point in the progression to discuss why kernel specialization changes both performance and profiler output. For a short lab, Exercises 1 through 4 are enough.
