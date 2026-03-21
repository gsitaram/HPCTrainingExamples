#!/bin/bash
# Script to profile pytorch_microbench with rocprof-compute
# This captures detailed GPU hardware metrics and compute performance analysis
#
# Compatible with ROCm 6.x and 7.x

set -e

# rocprof-compute counter support is primarily available on Instinct GPUs.
# On consumer parts such as gfx1100, the tool may fail before profiling starts.
GPU_ARCH=$(rocminfo 2>/dev/null | awk '/^[[:space:]]+Name:[[:space:]]+gfx/ {print $2; exit}')
SUPPORTED_ARCH_REGEX='^(gfx908|gfx90a|gfx940|gfx941|gfx942)$'

if [ -n "$GPU_ARCH" ] && ! echo "$GPU_ARCH" | grep -Eq "$SUPPORTED_ARCH_REGEX"; then
    echo "Skipping rocprof-compute profiling for pytorch_microbench..."
    echo "Detected GPU architecture: $GPU_ARCH"
    echo "rocprof-compute hardware-counter collection currently requires a supported Instinct GPU"
    echo "(for example gfx908, gfx90a, gfx940, gfx941, or gfx942)."
    echo "Use get_trace.sh and get_counters.sh on this system instead."
    exit 0
fi

# Create output directory with timestamp
OUTPUT_DIR="profiling_results/rocprof_compute_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Generate unique workload name with timestamp
WORKLOAD_NAME="pytorch_microbench_resnet50_$(date +%Y%m%d_%H%M%S)"

echo "Starting rocprof-compute profiling for pytorch_microbench..."
echo "Workload name: $WORKLOAD_NAME"
echo "Output directory: $OUTPUT_DIR"

# Run with rocprof-compute to collect detailed GPU metrics
# Using resnet50 as the default network with standard batch size
rocprof-compute profile \
    --name "$WORKLOAD_NAME" \
    -d "$OUTPUT_DIR" \
    -- python micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 64 \
    --iterations 10

echo ""
echo "Profiling complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "To analyze results:"
echo "  rocprof-compute analyze -p $OUTPUT_DIR/workloads/${WORKLOAD_NAME}/rocprof --dispatch <N> -n inference_dispatch"
echo ""
echo "For help on analysis options:"
echo "  rocprof-compute analyze --help"
