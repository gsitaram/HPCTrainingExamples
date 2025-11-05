#!/usr/bin/env python3
"""
Analyze kernel trace CSV from rocprofv3
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

def analyze_kernel_trace(csv_file):
    """Parse and summarize kernel trace data"""

    kernel_stats = defaultdict(lambda: {'count': 0, 'total_time': 0, 'times': []})
    total_kernels = 0

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Kind'] != 'KERNEL_DISPATCH':
                continue

            kernel_name = row['Kernel_Name']
            start = int(row['Start_Timestamp'])
            end = int(row['End_Timestamp'])
            duration_ns = end - start
            duration_us = duration_ns / 1000.0

            kernel_stats[kernel_name]['count'] += 1
            kernel_stats[kernel_name]['total_time'] += duration_us
            kernel_stats[kernel_name]['times'].append(duration_us)
            total_kernels += 1

    # Sort by total time
    sorted_kernels = sorted(kernel_stats.items(),
                           key=lambda x: x[1]['total_time'],
                           reverse=True)

    print("=" * 80)
    print("Kernel Trace Analysis")
    print("=" * 80)
    print(f"\nTotal kernel dispatches: {total_kernels}")
    print(f"Unique kernel types: {len(kernel_stats)}")
    print("")

    total_time = sum(s['total_time'] for s in kernel_stats.values())
    print(f"Total GPU time: {total_time:.2f} us ({total_time/1000:.2f} ms)")
    print("")

    print("Top kernels by total time:")
    print("-" * 80)
    print(f"{'Kernel Name':<60} {'Count':>8} {'Total(us)':>12} {'Avg(us)':>10}")
    print("-" * 80)

    for kernel_name, stats in sorted_kernels[:20]:
        short_name = kernel_name[:57] + "..." if len(kernel_name) > 60 else kernel_name
        avg_time = stats['total_time'] / stats['count']
        pct = (stats['total_time'] / total_time) * 100
        print(f"{short_name:<60} {stats['count']:>8} {stats['total_time']:>12.2f} {avg_time:>10.2f}")

    print("-" * 80)
    print("")

    # Timing statistics
    print("Timing Statistics (microseconds):")
    print("-" * 80)
    for kernel_name, stats in sorted_kernels[:10]:
        times = sorted(stats['times'])
        min_time = min(times)
        max_time = max(times)
        avg_time = sum(times) / len(times)
        median_time = times[len(times)//2]

        short_name = kernel_name.split('(')[0][-40:]
        print(f"\n{short_name}")
        print(f"  Count: {stats['count']}")
        print(f"  Min: {min_time:.2f} us, Max: {max_time:.2f} us")
        print(f"  Avg: {avg_time:.2f} us, Median: {median_time:.2f} us")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_kernel_trace.py <kernel_trace.csv>")
        sys.exit(1)

    csv_file = Path(sys.argv[1])
    if not csv_file.exists():
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    analyze_kernel_trace(csv_file)
