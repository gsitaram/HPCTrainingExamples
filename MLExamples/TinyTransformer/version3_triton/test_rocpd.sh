#!/bin/bash
#
# Test rocpd (ROCm Profiling Daemon) for continuous profiling
#

set -e

echo "=========================================="
echo "rocpd Test - Version 3"
echo "=========================================="
echo ""

# Check if rocpd is available
if ! command -v rocpd &> /dev/null; then
    echo "[ERROR] rocpd not found in PATH"
    echo "rocpd may not be installed or available in this ROCm version"
    exit 1
fi

echo "rocpd location: $(which rocpd)"
echo ""

OUTPUT_DIR="./rocpd/rocpd_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Start rocpd in background
echo "Starting rocpd daemon..."
rocpd --output-dir "$OUTPUT_DIR" &
ROCPD_PID=$!
echo "rocpd running with PID: $ROCPD_PID"
echo ""

# Give rocpd time to initialize
sleep 2

# Run workload
echo "Running workload: python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 10"
python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 10
WORKLOAD_EXIT=$?
echo ""

# Stop rocpd
echo "Stopping rocpd daemon..."
kill $ROCPD_PID 2>/dev/null || true
wait $ROCPD_PID 2>/dev/null || true
echo ""

if [ $WORKLOAD_EXIT -eq 0 ]; then
    echo "[SUCCESS] Workload completed"
else
    echo "[FAILED] Workload failed with exit code $WORKLOAD_EXIT"
fi
echo ""

echo "Generated files in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"
echo ""

echo "rocpd output is a SQLite3 database file"
echo ""
echo "To view the database:"
echo "  - Use VS Code SQLite Viewer extension"
echo "  - rocprof-compute and rocprof-systems can consume it directly"
echo "  - No official CLI tool is provided for viewing"
echo ""
echo "rocpd provides continuous profiling with minimal overhead"
echo ""
