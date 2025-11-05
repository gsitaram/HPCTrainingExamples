#!/bin/bash
#
# Get system-level profiling using rocprof-sys
# Compatible with ROCm 6.x and 7.x
#

set -e

echo "=========================================="
echo "rocprof-sys Profiling - Version 3"
echo "=========================================="
echo ""

OUTPUT_DIR="./rocprof_sys/profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run with rocprof-sys to collect system-level traces
# rocprof-sys-run provides call-stack sampling and system-level profiling
echo "Running: rocprof-sys-run --profile --trace -- python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 10"
echo ""

cd "$OUTPUT_DIR"
rocprof-sys-run --profile --trace -- python ../../tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 10
ROCPROF_EXIT=$?

echo ""
if [ $ROCPROF_EXIT -eq 0 ]; then
    echo "[SUCCESS] rocprof-sys profiling completed"
else
    echo "[FAILED] rocprof-sys profiling failed with exit code $ROCPROF_EXIT"
    exit 1
fi
echo ""

echo "Generated files:"
find . -type f -ls | head -20
echo ""

echo "rocprof-sys provides system-level profiling:"
echo "  - Call stack sampling"
echo "  - System trace timeline"
echo "  - CPU and GPU activity correlation"
echo "  - Function-level performance breakdown"
echo ""

echo "To view results, check for .perfetto-trace or .proto files"
echo "Perfetto traces can be viewed at: https://ui.perfetto.dev/"
echo ""
