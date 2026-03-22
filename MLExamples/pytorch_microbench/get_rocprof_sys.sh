#!/bin/bash
# Script to profile pytorch_microbench with rocprof-sys.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profile_common.sh"

require_cmd rocprof-sys-run
require_cmd "$PYTHON_BIN"
ensure_benchmark_script
build_benchmark_cmd

OUTPUT_DIR="$(make_output_dir rocprof_sys)"
mkdir -p "$OUTPUT_DIR"

echo "Starting rocprof-sys profiling for pytorch_microbench..."
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
PROTO_FILE="$(find "$OUTPUT_DIR" -name "*.proto" 2>/dev/null | head -1)"
if [ -n "$PROTO_FILE" ]; then
    echo "  Perfetto trace file: $PROTO_FILE"
    echo "  Open it in Perfetto UI: https://ui.perfetto.dev/"
else
    echo "  WARNING: No .proto file was found under $OUTPUT_DIR"
    echo "  Inspect the output tree and open the generated trace in Perfetto UI if present."
fi
