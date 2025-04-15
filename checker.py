import os
import pandas as pd
import numpy as np

# Define directories
normal_dir = './generalt_tablak/'
perfect_dir = './perfect_tables/'

# Helper function to count number of 2s in a file
def count_twos_in_file(filepath):
    try:
        df = pd.read_excel(filepath, header=None)
        block = df.iloc[4:10, 2:7]  # C5:G10
        return (block == 2).sum().sum()
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0

# Gather Excel files from both directories
normal_files = sorted([f for f in os.listdir(normal_dir) if f.endswith('.xlsx')])
perfect_files = sorted([f for f in os.listdir(perfect_dir) if f.endswith('.xlsx')])

# Sanity check
assert len(normal_files) == len(perfect_files), "Mismatch in file counts!"

# Count 2s in both sets
normal_counts = []
perfect_counts = []

for norm, perf in zip(normal_files, perfect_files):
    normal_path = os.path.join(normal_dir, norm)
    perfect_path = os.path.join(perfect_dir, perf)

    normal_2s = count_twos_in_file(normal_path)
    perfect_2s = count_twos_in_file(perfect_path)

    normal_counts.append(normal_2s)
    perfect_counts.append(perfect_2s)

def print_stats(normal_counts, perfect_counts):
    norm = np.array(normal_counts)
    perf = np.array(perfect_counts)

    print("Comparison of '2' Counts Across Tables")
    print("--------------------------------------------------------")
    print(f"{'Metric':<18} | {'NORMAL':>10} | {'PERFECT':>10}")
    print("--------------------------------------------------------")
    print(f"{'Max 2s':<18} | {norm.max():>10} | {perf.max():>10}")
    print(f"{'Min 2s':<18} | {norm.min():>10} | {perf.min():>10}")
    print(f"{'Avg 2s':<18} | {norm.mean():>10.2f} | {perf.mean():>10.2f}")
    print(f"{'Std deviation':<18} | {norm.std():>10.2f} | {perf.std():>10.2f}")
    print("--------------------------------------------------------")

print_stats(normal_counts, perfect_counts)