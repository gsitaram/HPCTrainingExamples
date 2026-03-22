#!/bin/bash
# Script to identify pytorch_microbench GPU hotspots with rocprofv3.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profile_common.sh"

require_cmd rocprofv3
require_cmd "$PYTHON_BIN"
ensure_benchmark_script
build_benchmark_cmd

ROCM_VERSION="$(detect_rocm_version)"
OUTPUT_DIR="$(make_output_dir gpu_hotspots)"

echo "Starting rocprofv3 GPU hotspot collection for pytorch_microbench..."
if [ -n "$ROCM_VERSION" ]; then
    echo "Detected ROCm version: $ROCM_VERSION"
else
    echo "Warning: Could not detect ROCm version. Proceeding with default rocprofv3 behavior."
fi
echo "Output directory: $OUTPUT_DIR"
print_workload_summary
echo ""

rocprofv3 \
    --kernel-trace \
    --output-directory "$OUTPUT_DIR" \
    -- "${BENCHMARK_CMD[@]}"

echo ""
echo "Profiling complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
print_generated_files "$OUTPUT_DIR" 3
echo ""

echo "To analyze results:"
DB_FILE="$(find "$OUTPUT_DIR" -name "*.db" 2>/dev/null | head -1)"
CSV_FILE="$(find "$OUTPUT_DIR" -name "*_kernel_trace.csv" 2>/dev/null | head -1)"
AGENT_INFO_FILE="$(find "$OUTPUT_DIR" -name "*_agent_info.csv" 2>/dev/null | head -1)"

if [ -n "$CSV_FILE" ]; then
    echo "  Kernel trace CSV: $CSV_FILE"
fi
if [ -n "$AGENT_INFO_FILE" ]; then
    echo "  Agent info CSV: $AGENT_INFO_FILE"
fi
if [ -n "$DB_FILE" ]; then
    echo "  SQLite database: $DB_FILE"
    echo ""
    echo "  Export to CSV:"
    echo "    rocpd2csv -i \"$DB_FILE\" -o kernel_stats.csv"
    echo ""
    echo "  Get kernel summary:"
    echo "    rocpd summary -i \"$DB_FILE\" --region-categories KERNEL"
fi
if [ -z "$CSV_FILE" ] && [ -z "$DB_FILE" ]; then
    echo "  WARNING: No ROCm profiler output file was detected under $OUTPUT_DIR"
fi
