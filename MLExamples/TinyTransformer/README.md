# ML Example: TinyTransformer Profiling Progression

In this directory we consider a small transformer training problem that is used to study profiling and performance changes across several implementations. The same model is advanced through a sequence of versions so that the effect of each optimization can be examined with the same workload and the same profiling tools.

The point of the progression is not only to obtain a faster model. It is also to see how the profiler output changes as the computation is restructured. We begin with a plain PyTorch baseline, then introduce operator fusion, custom Triton kernels, and an SDPA-based attention path. Each directory contains a short README with the commands needed to run and profile that version.

## Features of the various versions

- [`version1_pytorch_baseline`](version1_pytorch_baseline): reference PyTorch implementation; this is the right place to start
- [`version2_pytorch_fused`](version2_pytorch_fused): first round of fusion using framework-level mechanisms
- [`version3_triton`](version3_triton): custom Triton kernels for selected operations
- [`version4_pytorch_sdpa`](version4_pytorch_sdpa): SDPA-based attention together with the later fused paths

## Representative comparison

Representative results collected in [`VERSION_COMPARISON.md`](VERSION_COMPARISON.md) on an RX 7900 XTX with ROCm 6.4.4 are summarized below:

| Version | Samples/sec | Peak Memory | Main change |
|---------|-------------|-------------|-------------|
| V1 baseline | 240.6 | 434.3 MB | Plain PyTorch reference |
| V2 fused | 247.4 | 434.3 MB | First round of fusion |
| V3 Triton | 1054.8 | 193.8 MB | Custom Triton kernels |
| V4 SDPA | 1054.5 | 193.9 MB | PyTorch SDPA plus fused path |

These numbers will change with hardware, ROCm version, and problem size. The more stable point is the methodology: keep the model fixed, change one implementation layer at a time, and compare the traces, hotspot lists, and memory behavior.

## Common profiling tools

The version directories use a common set of ROCm profiling scripts:

- `get_trace.sh`: runtime trace with `rocprofv3`
- `get_counters.sh`: kernel trace with `rocprofv3`
- `get_rocprof_compute.sh`: hardware counter collection with `rocprof-compute`
- `get_rocprof_sys.sh`: system trace with `rocprof-sys`

Versions 2 through 4 also include `get_hotspots.sh`, which provides a fast first look at the kernels that dominate execution time.

## Running a first case

Load the required modules:

```bash
module load pytorch rocm
```

For versions 3 and 4, load Triton as well:

```bash
module load triton
```

We recommend the following order:

1. Run and profile `version1_pytorch_baseline`.
2. Compare the result to `version2_pytorch_fused` to see what modest fusion changes.
3. Move to `version3_triton` and `version4_pytorch_sdpa` to examine the larger change in kernel mix and memory use.

## Additional material

The following files provide the broader context for the example:

- [`VERSION_COMPARISON.md`](VERSION_COMPARISON.md): side-by-side profiling comparison across versions
- [`TINY_LLAMA_ARCHITECTURE.md`](TINY_LLAMA_ARCHITECTURE.md): model structure and implementation notes
- [`TECHNICAL_APPENDICES.md`](TECHNICAL_APPENDICES.md): supplementary technical discussion
