#!/bin/bash
# Collect a runtime trace for TinyTransformer V4 with rocprofv3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TINYTRANSFORMER_SCRIPT_DIR="$SCRIPT_DIR"
TINYTRANSFORMER_MODEL_SCRIPT="tiny_llama_v4.py"
TINYTRANSFORMER_WORKLOAD_NAME="tiny_llama_v4"
source "$SCRIPT_DIR/../profile_common.sh"

require_cmd rocprofv3
require_cmd "$PYTHON_BIN"
ensure_benchmark_script
build_benchmark_cmd

ROCM_VERSION="$(detect_rocm_version)"
ROCM_MAJOR="$(rocm_major_from_version "$ROCM_VERSION")"
OUTPUT_DIR="$(make_output_dir trace)"

echo "Starting rocprofv3 runtime trace for TinyTransformer V4..."
if [ -n "$ROCM_VERSION" ]; then
    echo "Detected ROCm version: $ROCM_VERSION"
fi
echo "Output directory: $OUTPUT_DIR"
print_workload_summary

TRACE_CMD=(rocprofv3 --runtime-trace --output-directory "$OUTPUT_DIR")
if [ "$ROCM_MAJOR" = "6" ] || [ "$ROCM_MAJOR" = "7" ]; then
    TRACE_CMD+=(--output-format pftrace)
fi

echo ""
"${TRACE_CMD[@]}" -- "${BENCHMARK_CMD[@]}"

echo ""
echo "Profiling complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
print_generated_files "$OUTPUT_DIR" 3
echo ""

PFTRACE_FILE="$(select_largest_match "$OUTPUT_DIR" "*.pftrace")"
DB_FILE="$(select_largest_match "$OUTPUT_DIR" "*.db")"

if [ -n "$PFTRACE_FILE" ]; then
    echo "Perfetto trace file: $PFTRACE_FILE"
    echo "Open it in Perfetto UI: https://ui.perfetto.dev/"
elif [ -n "$DB_FILE" ]; then
    echo "SQLite database found: $DB_FILE"
    echo "Convert it to Perfetto format with:"
    echo "  rocpd2pftrace -i \"$DB_FILE\" -o trace.pftrace"
else
    echo "WARNING: No .pftrace or .db file was found under $OUTPUT_DIR"
fi
