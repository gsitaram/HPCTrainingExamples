#!/bin/bash
# Script to profile inference_benchmark with rocprofv3 kernel trace and hardware counters
# This captures detailed GPU hardware metrics for performance analysis
#
# Supports both ROCm 6.x (CSV output) and ROCm 7.x (SQLite database output)

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
OUTPUT_DIR="profiling_results/counters_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Starting rocprofv3 hardware counter profiling for inference_benchmark..."
echo "Output directory: $OUTPUT_DIR"

# Run with rocprofv3 to collect kernel trace with hardware counters
# Using resnet50 as the default network with standard batch size
rocprofv3 \
    --kernel-trace \
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

# Check if analyze script exists, if not create it
if [ ! -f "analyze_kernel_trace.py" ]; then
    echo "Creating analyze_kernel_trace.py script..."
    cat > analyze_kernel_trace.py << 'EOF'
#!/usr/bin/env python3
"""
Analyze rocprofv3 kernel trace results and summarize performance metrics.
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict

def analyze_kernel_trace(trace_file):
    """Parse and analyze kernel trace CSV file."""

    kernels = []

    try:
        with open(trace_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kernels.append(row)
    except Exception as e:
        print(f"Error reading trace file: {e}")
        return

    if not kernels:
        print("No kernel data found in trace file")
        return

    # Aggregate statistics by kernel name
    kernel_stats = defaultdict(lambda: {'count': 0, 'total_duration': 0.0, 'durations': []})

    for kernel in kernels:
        # Support both naming conventions
        name = kernel.get('Kernel_Name') or kernel.get('Name', 'Unknown')

        # Calculate duration from timestamps if DurationNs not available
        if 'DurationNs' in kernel:
            duration_ns = float(kernel.get('DurationNs', 0))
        elif 'Start_Timestamp' in kernel and 'End_Timestamp' in kernel:
            start = float(kernel.get('Start_Timestamp', 0))
            end = float(kernel.get('End_Timestamp', 0))
            duration_ns = end - start
        else:
            duration_ns = 0.0

        kernel_stats[name]['count'] += 1
        kernel_stats[name]['total_duration'] += duration_ns
        kernel_stats[name]['durations'].append(duration_ns)

    # Calculate statistics and sort by total duration
    results = []
    total_time = 0.0

    for name, stats in kernel_stats.items():
        avg_duration = stats['total_duration'] / stats['count']
        total_time += stats['total_duration']

        results.append({
            'name': name,
            'count': stats['count'],
            'total_duration_ms': stats['total_duration'] / 1e6,
            'avg_duration_us': avg_duration / 1e3,
            'min_duration_us': min(stats['durations']) / 1e3,
            'max_duration_us': max(stats['durations']) / 1e3,
        })

    results.sort(key=lambda x: x['total_duration_ms'], reverse=True)

    # Print summary
    print(f"\n{'='*100}")
    print(f"Kernel Trace Analysis Summary")
    print(f"{'='*100}")
    print(f"Total kernels executed: {sum(r['count'] for r in results)}")
    print(f"Unique kernel types: {len(results)}")
    print(f"Total GPU time: {total_time / 1e6:.2f} ms")
    print(f"{'='*100}\n")

    # Print top kernels
    print(f"{'Kernel Name':<60} {'Count':>8} {'Total(ms)':>12} {'Avg(us)':>12} {'Min(us)':>12} {'Max(us)':>12} {'%Time':>8}")
    print(f"{'-'*60} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")

    for result in results[:20]:  # Top 20 kernels
        pct = (result['total_duration_ms'] / (total_time / 1e6)) * 100 if total_time > 0 else 0.0
        name_short = result['name'][:58] if len(result['name']) > 58 else result['name']
        print(f"{name_short:<60} {result['count']:>8} {result['total_duration_ms']:>12.3f} "
              f"{result['avg_duration_us']:>12.3f} {result['min_duration_us']:>12.3f} "
              f"{result['max_duration_us']:>12.3f} {pct:>7.1f}%")

    if len(results) > 20:
        print(f"\n... and {len(results) - 20} more kernel types")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_kernel_trace.py <trace_directory>")
        sys.exit(1)

    trace_dir = Path(sys.argv[1])

    # Find kernel trace CSV file (may have PID prefix like "6055_kernel_trace.csv")
    trace_files = list(trace_dir.glob("**/kernel_trace.csv"))
    if not trace_files:
        trace_files = list(trace_dir.glob("**/*_kernel_trace.csv"))

    if not trace_files:
        print(f"No kernel_trace.csv found in {trace_dir}")
        sys.exit(1)

    print(f"Analyzing kernel trace: {trace_files[0]}")
    analyze_kernel_trace(trace_files[0])
EOF
    chmod +x analyze_kernel_trace.py
fi

# Run analysis based on ROCm version
echo "Running analysis on profiling results..."
if [ "$ROCM_MAJOR" = "7" ] || [ -n "$(find "$OUTPUT_DIR" -name "*.db" 2>/dev/null)" ]; then
    echo "Detected ROCm 7.x SQLite database format"
    DB_FILE=$(find "$OUTPUT_DIR" -name "*_results.db" | head -1)
    if [ -n "$DB_FILE" ]; then
        echo "Database file: $DB_FILE"
        echo ""

        # Run Python analysis if script exists
        if [ -f "analyze_rocpd_db.py" ]; then
            python analyze_rocpd_db.py "$DB_FILE"
        else
            echo "Note: analyze_rocpd_db.py not found. Manual analysis:"
            echo "  sqlite3 $DB_FILE"
            echo ""
            echo "Example query:"
            echo "  SELECT s.string AS kernel_name, COUNT(*) as count,"
            echo "         AVG(kd.end_timestamp - kd.start_timestamp) as avg_duration_ns"
            echo "  FROM rocpd_kernel_dispatch kd"
            echo "  JOIN rocpd_string s ON kd.kernel_id = s.id"
            echo "  GROUP BY kernel_name ORDER BY avg_duration_ns DESC LIMIT 20;"
        fi
    else
        echo "No database file found in $OUTPUT_DIR"
    fi
else
    echo "Detected ROCm 6.x CSV format"
    python analyze_kernel_trace.py "$OUTPUT_DIR"
fi
