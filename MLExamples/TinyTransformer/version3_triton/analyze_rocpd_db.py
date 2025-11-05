#!/usr/bin/env python3
"""
Analyze ROCm 7.x rocpd SQLite database and summarize kernel performance metrics.
"""

import sys
import sqlite3
from pathlib import Path
from collections import defaultdict

def analyze_rocpd_database(db_file):
    """Parse and analyze rocpd SQLite database."""

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Check if required tables exist (with or without UUID suffix)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        # Find kernel_dispatch and string tables (may have UUID suffix in ROCm 7.x)
        kernel_dispatch_table = None
        string_table = None

        for table in tables:
            if table.startswith('rocpd_kernel_dispatch'):
                kernel_dispatch_table = table
            if table.startswith('rocpd_string'):
                string_table = table

        if not kernel_dispatch_table or not string_table:
            print(f"Error: Database missing required tables")
            print(f"Available tables: {', '.join(tables)}")
            conn.close()
            return

        print(f"Using tables: {kernel_dispatch_table}, {string_table}")

        # Query kernel dispatch data with kernel names
        # Join with info_kernel_symbol table for kernel names
        kernel_symbol_table = None
        for table in tables:
            if table.startswith('rocpd_info_kernel_symbol'):
                kernel_symbol_table = table
                break

        if not kernel_symbol_table:
            print(f"Error: Could not find kernel symbol table")
            conn.close()
            return

        query = f"""
        SELECT
            s.display_name AS kernel_name,
            kd.start,
            kd.end,
            (kd.end - kd.start) AS duration_ns
        FROM {kernel_dispatch_table} kd
        JOIN {kernel_symbol_table} s ON kd.kernel_id = s.id AND kd.guid = s.guid
        WHERE s.display_name IS NOT NULL
        ORDER BY duration_ns DESC
        """

        cursor.execute(query)
        kernels = cursor.fetchall()

        if not kernels:
            print("No kernel data found in database")
            conn.close()
            return

        # Aggregate statistics by kernel name
        kernel_stats = defaultdict(lambda: {'count': 0, 'total_duration': 0.0, 'durations': []})

        for kernel_name, start_ts, end_ts, duration_ns in kernels:
            kernel_stats[kernel_name]['count'] += 1
            kernel_stats[kernel_name]['total_duration'] += duration_ns
            kernel_stats[kernel_name]['durations'].append(duration_ns)

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
        print(f"ROCm 7.x Database Analysis Summary")
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

        conn.close()

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error analyzing database: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_rocpd_db.py <database_file_or_directory>")
        sys.exit(1)

    path = Path(sys.argv[1])

    # If directory provided, find database file
    if path.is_dir():
        db_files = list(path.glob("**/*_results.db"))
        if not db_files:
            print(f"No *_results.db database file found in {path}")
            sys.exit(1)
        db_file = db_files[0]
    else:
        db_file = path

    if not db_file.exists():
        print(f"Database file not found: {db_file}")
        sys.exit(1)

    print(f"Analyzing ROCm 7.x database: {db_file}")
    analyze_rocpd_database(db_file)
