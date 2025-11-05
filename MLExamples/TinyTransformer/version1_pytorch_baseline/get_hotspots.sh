#!/bin/bash
#
# Get hotspots analysis using rocprofv3
# Compatible with ROCm 6.x and 7.x
#

set -e

echo "=========================================="
echo "rocprofv3 Hotspots Analysis - Version 1"
echo "=========================================="
echo ""

OUTPUT_DIR="./hotspots/hotspot_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Running: rocprofv3 --stats -- python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 10"
echo ""

cd "$OUTPUT_DIR"
rocprofv3 --stats -- python ../../tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 10
ROCPROF_EXIT=$?

echo ""
if [ $ROCPROF_EXIT -eq 0 ]; then
    echo "[SUCCESS] Hotspot analysis completed"
else
    echo "[FAILED] Hotspot analysis failed with exit code $ROCPROF_EXIT"
    exit 1
fi
echo ""

echo "Generated files:"
find . -type f -ls
echo ""

# Check for stats/CSV files
if ls *.csv 1> /dev/null 2>&1; then
    echo "Statistics files found:"
    for f in *.csv; do
        echo ""
        echo "File: $f"
        echo "Top 10 entries:"
        head -11 "$f"
    done
else
    echo "Looking for statistics in subdirectories:"
    find . -name "*.csv" -exec echo "Found: {}" \; -exec head -11 {} \;
fi
echo ""

echo "Hotspot analysis identifies GPU kernels with highest time consumption."
echo ""
