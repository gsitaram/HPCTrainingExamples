#!/bin/bash
# Script to profile pytorch_microbench with rocprofv3 runtime trace.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profile_common.sh"

require_cmd rocprofv3
require_cmd "$PYTHON_BIN"
ensure_benchmark_script
build_benchmark_cmd

ROCM_VERSION="$(detect_rocm_version)"
ROCM_MAJOR="$(rocm_major_from_version "$ROCM_VERSION")"
OUTPUT_DIR="$(make_output_dir trace)"

echo "Starting rocprofv3 runtime trace profiling for pytorch_microbench..."
if [ -n "$ROCM_VERSION" ]; then
    echo "Detected ROCm version: $ROCM_VERSION"
else
    echo "Warning: Could not detect ROCm version. Proceeding without version-specific assumptions."
fi
echo "Output directory: $OUTPUT_DIR"
print_workload_summary

TRACE_CMD=(rocprofv3 --runtime-trace --output-directory "$OUTPUT_DIR")
if [ "$ROCM_MAJOR" = "6" ] || [ "$ROCM_MAJOR" = "7" ]; then
    echo "Using explicit Perfetto output for ROCm $ROCM_MAJOR.x."
    TRACE_CMD+=(--output-format pftrace)
fi

echo ""
echo "Collecting full runtime trace (API calls, kernels, memory operations, and synchronization events)..."
echo ""

"${TRACE_CMD[@]}" -- "${BENCHMARK_CMD[@]}"

echo ""
echo "Profiling complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
print_generated_files "$OUTPUT_DIR" 3
echo ""

PFTRACE_FILE="$(find "$OUTPUT_DIR" -name "*.pftrace" | head -1)"
DB_FILE="$(find "$OUTPUT_DIR" -name "*.db" | head -1)"

if [ -n "$PFTRACE_FILE" ]; then
    echo "Perfetto trace file: $PFTRACE_FILE"
    echo "Size: $(du -h "$PFTRACE_FILE" | cut -f1)"
    echo "Open it in Perfetto UI: https://ui.perfetto.dev/"
elif [ -n "$DB_FILE" ]; then
    echo "SQLite database found: $DB_FILE"
    echo "Convert it to Perfetto format with:"
    echo "  rocpd2pftrace -i \"$DB_FILE\" -o trace.pftrace"
else
    echo "WARNING: No .pftrace or .db file was found under $OUTPUT_DIR"
fi
echo ""
