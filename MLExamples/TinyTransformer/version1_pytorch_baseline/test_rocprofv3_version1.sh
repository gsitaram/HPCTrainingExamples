#!/bin/bash
#
# rocprofv3 validation test for tiny_llama_v1.py
# Tests profiler capture on baseline PyTorch implementation
#

set -e

echo "=========================================="
echo "rocprofv3 Test Suite - Version 1 Baseline"
echo "=========================================="
echo ""

# Step 1: Environment Validation
echo "[STEP 1] Environment Validation"
echo "----------------------------------------"

echo "ROCm Version:"
rocm-smi --showproductname || echo "rocm-smi failed"
echo ""

echo "GPU Visibility:"
echo "  HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "  ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"
echo "  HSA_ENABLE_PROFILING=$HSA_ENABLE_PROFILING"
echo ""

echo "rocprofv3 location:"
which rocprofv3
echo ""

echo "PyTorch + ROCm Check:"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Device capability: {torch.cuda.get_device_capability(0)}')
else:
    print('WARNING: CUDA/ROCm not available!')
"
echo ""

# Step 2: Baseline Test (No Profiler)
echo "[STEP 2] Baseline Test - No Profiler"
echo "----------------------------------------"
echo "Running: python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 5 --validate-setup"
echo ""

python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 5 --validate-setup
BASELINE_EXIT=$?

if [ $BASELINE_EXIT -eq 0 ]; then
    echo "[SUCCESS] Baseline test passed"
else
    echo "[FAILED] Baseline test failed with exit code $BASELINE_EXIT"
    exit 1
fi
echo ""

# Step 3: rocprofv3 with runtime-trace (GitHub issue command pattern)
echo "[STEP 3] rocprofv3 Test - Runtime Trace + Perfetto"
echo "----------------------------------------"
echo "Running: rocprofv3 --runtime-trace --output-format pftrace -- python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 10 --enable-pytorch-profiler --profile-memory"
echo ""

OUTPUT_DIR="./rocprof_v1_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

rocprofv3 --runtime-trace --output-format pftrace -- python ../tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 10 --enable-pytorch-profiler --profile-memory
ROCPROF_EXIT=$?

echo ""
if [ $ROCPROF_EXIT -eq 0 ]; then
    echo "[SUCCESS] rocprofv3 completed"
else
    echo "[FAILED] rocprofv3 failed with exit code $ROCPROF_EXIT"
fi
echo ""

# Check generated files
echo "Generated files:"
ls -lh
echo ""

# Check for profiling data
if ls *.pftrace 1> /dev/null 2>&1; then
    echo "Found perfetto trace files:"
    ls -lh *.pftrace

    echo ""
    echo "Checking trace file size:"
    for f in *.pftrace; do
        size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
        if [ $size -gt 1000 ]; then
            echo "  $f: $size bytes (likely has data)"
        else
            echo "  $f: $size bytes (suspiciously small)"
        fi
    done
else
    echo "No .pftrace files found in current directory"
    echo "Checking subdirectories..."
    find . -name "*.pftrace" -ls
fi
echo ""

# Check for PyTorch profiler output
if [ -d "pytorch_profiles" ]; then
    echo ""
    echo "PyTorch Profiler output:"
    ls -lh pytorch_profiles/
    echo ""
    echo "TensorBoard traces available:"
    echo "  Launch: tensorboard --logdir pytorch_profiles"
else
    echo ""
    echo "Note: pytorch_profiles directory not found (script may need directory creation fix)"
fi

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Results directory: $OUTPUT_DIR"
echo ""
echo "Generated profiling data:"
echo "  1. rocprofv3 perfetto traces (.pftrace files)"
echo "  2. PyTorch profiler traces (pytorch_profiles/ if present)"
echo ""
echo "Next steps:"
echo "  1. Inspect generated files in $OUTPUT_DIR"
echo "  2. Open .pftrace in perfetto.dev or chrome://tracing"
echo "  3. View PyTorch traces with tensorboard --logdir pytorch_profiles"
echo "  4. Check for GPU kernel activity in both profilers"
echo "  5. Compare to GitHub issue #1386 output"
echo ""
echo "To view perfetto trace:"
echo "  Visit: https://ui.perfetto.dev/"
echo "  Click 'Open trace file' and select the .pftrace file"
echo ""
