#!/bin/bash
# Script to profile inference_benchmark with rocprofv3 runtime trace
# This captures GPU API calls, kernel launches, and memory operations
#
# Compatible with ROCm 6.x and 7.x

set -e

# Detect ROCm version
ROCM_VERSION=""
ROCM_MAJOR=""

# Method 1: Check rocminfo
if command -v rocminfo &> /dev/null; then
    ROCM_VERSION=$(rocminfo | grep -i "ROCm Version" | head -1 | awk '{print $3}')
fi

# Method 2: Check ROCM_PATH
if [ -z "$ROCM_VERSION" ] && [ -n "$ROCM_PATH" ]; then
    if [ -f "$ROCM_PATH/.info/version" ]; then
        ROCM_VERSION=$(cat "$ROCM_PATH/.info/version")
    fi
fi

# Method 3: Check hipcc version (more reliable for module-loaded ROCm)
if [ -z "$ROCM_VERSION" ] && command -v hipcc &> /dev/null; then
    HIP_VERSION=$(hipcc --version 2>/dev/null | grep -i "HIP version" | head -1 | awk '{print $3}')
    if [ -n "$HIP_VERSION" ]; then
        ROCM_VERSION="$HIP_VERSION"
    fi
fi

# Extract major version
if [ -n "$ROCM_VERSION" ]; then
    ROCM_MAJOR=$(echo "$ROCM_VERSION" | cut -d. -f1)
    echo "Detected ROCm version: $ROCM_VERSION"
else
    echo "Warning: Could not detect ROCm version, assuming ROCm 7.x"
    ROCM_MAJOR="7"
fi

# Create output directory with timestamp
OUTPUT_DIR="profiling_results/trace_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Starting rocprofv3 runtime trace profiling for inference_benchmark..."
echo "Output directory: $OUTPUT_DIR"

# Build rocprofv3 command with appropriate flags for ROCm version
# ROCm 6.4+ and 7.x require explicit --output-format pftrace to generate Perfetto traces
# Earlier ROCm 6.x versions (6.0-6.3) generated pftrace by default
if [ "$ROCM_MAJOR" = "7" ] || [ "$ROCM_MAJOR" = "6" ]; then
    echo "Using ROCm 6.x/7.x: --output-format pftrace (generates Perfetto trace)"
    OUTPUT_FORMAT="--output-format pftrace"
else
    echo "Using ROCm 5.x or older: default format"
    OUTPUT_FORMAT=""
fi

echo ""
echo "Collecting full runtime trace (HIP/HSA API calls, kernels, memory operations)"
echo ""

# Run with rocprofv3 to collect full runtime trace
# Using resnet50 as the default network with standard batch size
# NOTE: Using --runtime-trace to capture complete timeline:
#       - HIP/HSA API calls
#       - Kernel execution on GPU
#       - Memory operations (H2D, D2H, D2D transfers)
#       - Synchronization events
#       This provides the comprehensive view needed for timeline analysis in Perfetto
rocprofv3 \
    --runtime-trace \
    $OUTPUT_FORMAT \
    --output-directory "$OUTPUT_DIR" \
    -- python micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 64 \
    --iterations 10

echo ""
echo "Profiling complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*/ 2>/dev/null || ls -lh "$OUTPUT_DIR"
echo ""

# Find and highlight the pftrace file
PFTRACE_FILE=$(find "$OUTPUT_DIR" -name "*.pftrace" | head -1)
DB_FILE=$(find "$OUTPUT_DIR" -name "*.db" | head -1)

if [ -n "$PFTRACE_FILE" ]; then
    echo "Perfetto trace file found: $PFTRACE_FILE"
    echo "Size: $(du -h "$PFTRACE_FILE" | cut -f1)"
    echo ""
    echo "To view the trace:"
    echo "  1. Visit: https://ui.perfetto.dev/"
    echo "  2. Open: $PFTRACE_FILE"
elif [ -n "$DB_FILE" ]; then
    echo "SQLite database found (ROCm 7.x without --output-format): $DB_FILE"
    echo "To convert to Perfetto format:"
    echo "  rocpd2pftrace -i $DB_FILE -o trace.pftrace"
    echo ""
    echo "Next time, use --output-format pftrace to generate Perfetto traces directly"
else
    echo "WARNING: No .pftrace or .db file found"
    echo "Check the output directory for profiling results"
fi
echo ""
