import re
import sys
from statistics import mean, quantiles
from collections import defaultdict, Counter

# Updated Regex to capture:
# 1. Timings block
# 2. Total Time (last digit in block)
# 3. HTTP Status Code
# 4. Request Path
LOG_PATTERN = re.compile(r"(\d+/\d+/\d+/\d+/(\d+))\s+(\d{3}).*?\"[A-Z]+\s+([^?\s]+)")

def analyze_logs(file_path):
    # path -> { 'timings': [], 'codes': Counter() }
    stats = defaultdict(lambda: {'timings': [], 'codes': Counter()})

    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = LOG_PATTERN.search(line)
                if match:
                    total_time = int(match.group(2))
                    status_code = match.group(3)
                    path = match.group(4)
                    
                    stats[path]['timings'].append(total_time)
                    stats[path]['codes'][status_code] += 1
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # Table Header
    header = f"{'Endpoint Path':<30} | {'Count':<6} | {'Avg':<8} | {'Q1/Q3':<15} | {'Status Codes'}"
    print(header)
    print("-" * len(header))

    for path, data in sorted(stats.items()):
        timings = data['timings']
        cnt = len(timings)
        avg = mean(timings)
        
        # Calculate Quartiles
        if cnt > 1:
            q = quantiles(timings, n=4)
            q_str = f"{q[0]:.0f}/{q[2]:.0f}"
        else:
            q_str = f"{timings[0]}/{timings[0]}"

        # Format Status Codes: "200:5, 404:1"
        code_summary = ", ".join([f"{code}:{count}" for code, count in sorted(data['codes'].items())])

        print(f"{path:<30} | {cnt:<6} | {avg:<8.1f} | {q_str:<15} | {code_summary}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_logs(sys.argv[1])
    else:
        print("Usage: python analyze_logs.py <path_to_log_file>")
