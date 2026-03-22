#!/bin/bash
# Collect a quick hotspot summary for TinyTransformer V4 with rocprofv3 --stats.

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

OUTPUT_DIR="$(make_output_dir hotspots)"

echo "Starting rocprofv3 hotspot summary for TinyTransformer V4..."
echo "Output directory: $OUTPUT_DIR"
print_workload_summary
echo ""

rocprofv3 \
    --kernel-trace \
    --stats \
    --output-directory "$OUTPUT_DIR" \
    -- "${BENCHMARK_CMD[@]}"

echo ""
echo "Profiling complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
print_generated_files "$OUTPUT_DIR" 3
echo ""

CSV_FILE="$(select_largest_match "$OUTPUT_DIR" "*_kernel_stats.csv")"
if [ -z "$CSV_FILE" ]; then
    CSV_FILE="$(select_largest_match "$OUTPUT_DIR" "*_domain_stats.csv")"
fi
if [ -n "$CSV_FILE" ]; then
    echo "Top rows from $CSV_FILE:"
    head -11 "$CSV_FILE"
else
    echo "WARNING: No hotspot CSV file was detected under $OUTPUT_DIR"
fi
