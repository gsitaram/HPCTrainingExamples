# rocprofv3 Test Results - Version 1 Baseline

ROCPROFV3_VERSION1_RESULTS.md from `HPCTrainingExamples/MLExamples/TinyTransformer/version1_pytorch_baseline` in the Training Examples repository.

## Summary

rocprofv3 successfully captures profiling data from version1 baseline. This document shows example results from runtime trace collection.

## Test Configuration

**Command:**

```
rocprofv3 --runtime-trace --output-format pftrace -- python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 10
```

**Environment:**
- ROCm Version: 6.4.x
- PyTorch: ROCm-enabled build
- GPU: AMD Instinct or Radeon with gfx support

## Example Output

```
Training completed: 10 steps, batch size 8
Performance: 262.3 samples/sec, 33,571 tokens/sec
Memory usage: 434.3 MB peak
rocprofv3 exit code: 0 (success)
```

## Generated Files

Output directory contains Perfetto trace files:

| File | Size | Description |
|------|------|-------------|
| `<pid>_results.pftrace` | ~40-50 MB | Main trace with full profiling data |
| Additional `.pftrace` files | ~600 bytes | Minimal traces from subprocesses |

The main trace file (largest) contains the full profiling data for timeline analysis.

## Viewing the Trace

1. Visit https://ui.perfetto.dev/
2. Click "Open trace file"
3. Select the main `.pftrace` file
4. Examine:
   - GPU kernel timeline
   - Memory transfer operations
   - HIP API calls
   - Kernel duration and overlap

## Warnings

The following warning may appear and can be ignored:

```
rocprofiler_iterate_agent_supported_counters returned ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED for agent X (gfxXXXX)
```

This typically relates to integrated GPUs or unsupported architectures and does not affect profiling of the target GPU.

## Additional Resources

- rocprofv3 documentation: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html
- Perfetto UI: https://ui.perfetto.dev/
