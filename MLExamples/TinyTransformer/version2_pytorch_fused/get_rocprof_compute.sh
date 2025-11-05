#!/bin/bash
#
# Get detailed GPU metrics using rocprof-compute
# Compatible with ROCm 6.x and 7.x
#

set -e

echo "=========================================="
echo "rocprof-compute Profiling - Version 2"
echo "=========================================="
echo ""

OUTPUT_DIR="./rocprof_compute/profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run with rocprof-compute to collect detailed GPU metrics
# rocprof-compute requires: profile mode --name <workload_name> -d <dir> -- <command>
WORKLOAD_NAME="tiny_llama_v2_$(date +%Y%m%d_%H%M%S)"
echo "Running: rocprof-compute profile --name $WORKLOAD_NAME -d $OUTPUT_DIR -- python tiny_llama_v2.py --batch-size 8 --seq-len 128 --num-steps 10"
echo ""

rocprof-compute profile --name "$WORKLOAD_NAME" -d "$OUTPUT_DIR" -- python tiny_llama_v2.py --batch-size 8 --seq-len 128 --num-steps 10
ROCPROF_EXIT=$?

echo ""
if [ $ROCPROF_EXIT -eq 0 ]; then
    echo "[SUCCESS] rocprof-compute profiling completed"
else
    echo "[FAILED] rocprof-compute profiling failed with exit code $ROCPROF_EXIT"
    exit 1
fi
echo ""

echo "Generated files:"
find "$OUTPUT_DIR" -type f -ls
echo ""

echo "rocprof-compute provides detailed GPU performance analysis:"
echo "  - Kernel execution timeline"
echo "  - Memory transfer analysis"
echo "  - Hardware counter metrics"
echo "  - Occupancy statistics"
echo ""

echo "To view results, check the output directory for CSV and report files."
echo ""
