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
