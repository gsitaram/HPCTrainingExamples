#!/bin/bash
# Collect a system trace for TinyTransformer V1 with rocprof-sys.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TINYTRANSFORMER_SCRIPT_DIR="$SCRIPT_DIR"
TINYTRANSFORMER_MODEL_SCRIPT="tiny_llama_v1.py"
TINYTRANSFORMER_WORKLOAD_NAME="tiny_llama_v1"
TINYTRANSFORMER_DEFAULT_NUM_STEPS=2
source "$SCRIPT_DIR/../profile_common.sh"

require_cmd rocprof-sys-run
require_cmd "$PYTHON_BIN"
ensure_benchmark_script
build_benchmark_cmd

OUTPUT_DIR="$(make_output_dir rocprof_sys)"

echo "Starting rocprof-sys trace for TinyTransformer V1..."
echo "Output directory: $OUTPUT_DIR"
print_workload_summary
echo ""

pushd "$OUTPUT_DIR" >/dev/null
rocprof-sys-run \
    --profile \
    --trace \
    -- "${BENCHMARK_CMD[@]}"
popd >/dev/null

echo ""
echo "Profiling complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
print_generated_files "$OUTPUT_DIR" 4
echo ""
echo "Open the trace in Perfetto:"
PROTO_FILE="$(select_largest_match "$OUTPUT_DIR" "*.proto")"
if [ -n "$PROTO_FILE" ]; then
    echo "  Perfetto trace file: $PROTO_FILE"
    echo "  Open it in Perfetto UI: https://ui.perfetto.dev/"
else
    echo "  WARNING: No .proto file was found under $OUTPUT_DIR"
    echo "  Inspect the output tree and open the generated trace in Perfetto UI if present."
fi
