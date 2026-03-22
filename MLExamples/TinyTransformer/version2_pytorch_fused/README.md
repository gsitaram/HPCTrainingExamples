# TinyTransformer Version 2: Framework-Level Fusion

This version keeps the same workload as version 1 and asks a narrower question: what changes when the model is routed through framework-level fusion paths?

## What changed

The intended differences relative to version 1 are:

- fused QKV projection path
- fused or memory-efficient attention path
- fused SwiGLU path
- `torch.compile`-driven graph and kernel fusion when available

Whether those paths are actually active depends on the software stack. That is part of the lesson for this version.

## Baseline run

Load the same environment as version 1:

```bash
module load pytorch rocm
```

Run:

```bash
python tiny_llama_v2.py --batch-size 8 --seq-len 128 --num-steps 10
```

Example output from one validated run:

```text
Performance Summary V2:
   Average training speed: 259.0 samples/sec
   Throughput: 33152 tokens/sec
   Average batch time: 30.9 ms
   Peak memory usage: 434.3 MB

Fusion Efficiency:
   QKV Fusion Active: False
   Flash Attention Active: False
   SwiGLU Fusion Active: False
   Kernel Reduction: 0.0%
```

On this stack, the fused paths were not active. That is still useful training material because it shows that version 2 should be treated as a check, not as a guaranteed speedup.

## Profiling workflow

Use the same scripts as version 1:

- `./get_hotspots.sh`
- `./get_trace.sh`
- `./get_counters.sh`
- `./get_rocprof_compute.sh`
- `./get_rocprof_sys.sh`

The first question is whether the hotspot list and trace structure actually differ from version 1. If the fused paths are active, you should expect fewer short-lived kernels and a more concentrated dominant kernel set. If they are inactive, version 2 becomes a useful negative control.

## Comparison target

Compare this version directly against [`../version1_pytorch_baseline`](../version1_pytorch_baseline). The comparison is more important than the absolute number from any single run.

## References

- comparison across versions: [`../VERSION_COMPARISON.md`](../VERSION_COMPARISON.md)
- rocprofv3: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html
- Perfetto UI: https://ui.perfetto.dev/
