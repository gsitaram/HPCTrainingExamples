#!/bin/bash
#
# Test exact command from GitHub issue #1386
# Issue: "No device activity" with rocprofv3 on version2
#

set -e

echo "=========================================="
echo "GitHub Issue #1386 Reproduction Test"
echo "=========================================="
echo ""

OUTPUT_DIR="./github_issue_test/test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Reproducing exact command from GitHub issue #1386:"
echo "rocprofv3 --runtime-trace --output-format pftrace -- python tiny_llama_v2.py --batch-size 8 --seq-len 128"
echo ""
echo "Note: GitHub issue did NOT specify --num-steps, so default value will be used"
echo ""

cd "$OUTPUT_DIR"
rocprofv3 --runtime-trace --output-format pftrace -- python ../../tiny_llama_v2.py --batch-size 8 --seq-len 128
ROCPROF_EXIT=$?

echo ""
if [ $ROCPROF_EXIT -eq 0 ]; then
    echo "[SUCCESS] rocprofv3 profiling completed"
else
    echo "[FAILED] rocprofv3 profiling failed with exit code $ROCPROF_EXIT"
    exit 1
fi
echo ""

echo "Generated files:"
find . -type f -ls
echo ""

echo "Checking trace file sizes:"
if compgen -G "*/*.pftrace" > /dev/null; then
    for f in */*.pftrace; do
        SIZE=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo "unknown")
        SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc)
        echo "  $f - ${SIZE_MB} MB"
    done
    echo ""
    LARGEST=$(find . -name "*.pftrace" -exec ls -l {} \; | sort -k5 -n -r | head -1 | awk '{print $9, $5}')
    LARGEST_FILE=$(echo $LARGEST | awk '{print $1}')
    LARGEST_SIZE=$(echo $LARGEST | awk '{print $2}')
    LARGEST_MB=$(echo "scale=2; $LARGEST_SIZE / 1048576" | bc)

    echo "Largest trace: $LARGEST_FILE (${LARGEST_MB} MB)"
    echo ""

    if (( $(echo "$LARGEST_MB < 1" | bc -l) )); then
        echo "[WARNING] Trace file is very small (< 1 MB)"
        echo "This may indicate 'no device activity' issue from GitHub #1386"
    else
        echo "[OK] Trace file size looks normal"
        echo "Version2 profiling appears to be working correctly"
    fi
else
    echo "[ERROR] No .pftrace files found"
fi
echo ""

echo "Comparison with version1 baseline:"
echo "  Version1 trace size: ~44 MB"
echo "  Version2 trace size: ${LARGEST_MB} MB"
echo ""
