# TinyTransformer Version 4: SDPA Path

Version 4 keeps the later fused structure but swaps the attention path to PyTorch SDPA. The main use of this directory is to compare a framework-maintained attention implementation against the custom Triton path in version 3 while keeping the workload fixed.

## What changed

Relative to version 3, this version:

- uses PyTorch SDPA for attention
- keeps the later fused model structure
- keeps the same profiling workflow as the rest of the progression

## Baseline run

Load the required modules:

```bash
module load pytorch rocm triton
```

Run:

```bash
python tiny_llama_v4.py --batch-size 8 --seq-len 128 --num-steps 10
```

Example output from one validated run:

```text
Performance Summary V4:
   Average training speed: 830.7 samples/sec
   Throughput: 106332 tokens/sec
   Average batch time: 9.6 ms
   Peak memory usage: 193.9 MB
```

On the validated container, version 4 landed very close to version 3. That is the main comparison to keep in mind when profiling this directory.

## Profiling workflow

Use the same scripts as the earlier versions:

- `./get_hotspots.sh`
- `./get_trace.sh`
- `./get_counters.sh`
- `./get_rocprof_compute.sh`
- `./get_rocprof_sys.sh`

Start with `./get_hotspots.sh` and `./get_trace.sh`. The main question is whether the attention region and dominant kernel set change materially relative to version 3, even when the overall step time remains similar.

## Comparison target

Compare this directory directly against [`../version3_triton`](../version3_triton). The interesting result is not whether version 4 is slightly faster or slower on one machine. It is whether the optimized behavior is preserved while relying on a framework path.

## References

- comparison across versions: [`../VERSION_COMPARISON.md`](../VERSION_COMPARISON.md)
- PyTorch SDPA overview: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Perfetto UI: https://ui.perfetto.dev/
