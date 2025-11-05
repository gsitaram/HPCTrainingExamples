#!/bin/bash
# Script to profile inference_benchmark with rocprof-sys
# This captures system-level performance with call stack sampling
#
# Compatible with ROCm 6.x and 7.x
#
# NOTE: rocprof-sys may produce memory map dumps in some configurations
# This is a known issue tracked in GitHub. If profiling fails or produces
# excessive output, consider using rocprofv3 or rocprof-compute instead.

set -e

# Create output directory with timestamp
OUTPUT_DIR="profiling_results/rocprof_sys_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Starting rocprof-sys profiling for inference_benchmark..."
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "NOTE: If you see excessive memory map output, this is a known issue."
echo "Consider using rocprofv3 (get_trace.sh) or rocprof-compute (get_rocprof_compute.sh) instead."
echo ""

cd "$OUTPUT_DIR"

# Run with rocprof-sys to collect system-level profile
# Using resnet50 as the default network with standard batch size
rocprof-sys-run \
    --profile \
    --trace \
    -- python ../../micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 64 \
    --iterations 10

cd ../..

echo ""
echo "Profiling complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "To analyze results, use rocprof-sys tools:"
echo "  rocprof-sys-avail --help"
echo "  rocprof-sys-analyze --help"
