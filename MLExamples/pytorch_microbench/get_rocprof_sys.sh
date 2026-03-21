#!/bin/bash
# Script to profile pytorch_microbench with rocprof-sys
# This captures system-level performance with call stack sampling
#
# Compatible with ROCm 6.x and 7.x
#
# NOTE: rocprof-sys may produce memory map dumps in some configurations.
# Issue reference: TBD

set -e

# Create output directory with timestamp
OUTPUT_DIR="profiling_results/rocprof_sys_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Starting rocprof-sys profiling for pytorch_microbench..."
echo "Output directory: $OUTPUT_DIR"
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
echo "To analyze results:"
PROTO_FILE=$(find "$OUTPUT_DIR" -name "*.proto" 2>/dev/null | head -1)
if [ -n "$PROTO_FILE" ]; then
    echo "  Perfetto trace file: $PROTO_FILE"
    echo "  Open it in Perfetto UI: https://ui.perfetto.dev/"
else
    echo "  Open the generated .proto file in Perfetto UI: https://ui.perfetto.dev/"
fi
