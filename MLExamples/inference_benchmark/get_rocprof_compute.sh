#!/bin/bash
# Script to profile inference_benchmark with rocprof-compute
# This captures detailed GPU hardware metrics and compute performance analysis
#
# Compatible with ROCm 6.x and 7.x

set -e

# Create output directory with timestamp
OUTPUT_DIR="profiling_results/rocprof_compute_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Generate unique workload name with timestamp
WORKLOAD_NAME="inference_benchmark_resnet50_$(date +%Y%m%d_%H%M%S)"

echo "Starting rocprof-compute profiling for inference_benchmark..."
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
echo "To analyze results, use rocprof-compute analyze tools:"
echo "  rocprof-compute analyze --help"
echo "  rocprof-compute analyze --workload-dir $OUTPUT_DIR"
