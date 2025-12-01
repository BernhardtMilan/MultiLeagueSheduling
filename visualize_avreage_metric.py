import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

improved_greedy_metric = [17648.4, 17730.5, 17689.3, 17728.9, 17689.5, 17762.0, 17824.4, 17757.8, 17683.6, 17619.6, 17698.3]
evo_best_metric = [18000.0, 18004.0, 17998.5, 18010.0, 18013.5, 17996.5, 18016.0, 18013.0, 17998.5, 18004.0, 18014.0]
ats_best_metric = [17875.3, 17824.4, 17971.3, 17999.8, 17938.7, 17879.4, 17785.1, 17904.2, 17866.9, 17914.8, 17898.4]
ga_best_metric = [17795.2, 17876.6, 17919.4, 17746.3, 17888.3, 17776.5, 17826.9, 17804.6, 17763.7, 17638.9, 17754.2]

data = [
    improved_greedy_metric,
    evo_best_metric,
    ats_best_metric,
    ga_best_metric
]

labels = ['Impr. Greedy', 'Proposed method', 'ATS', 'GA']

stats = {
    "Min": [np.min(d) for d in data],
    "Max": [np.max(d) for d in data],
    "Mean": [np.mean(d) for d in data],
    "Range": [np.ptp(d) for d in data],
    "Std. Dev.": [np.std(d, ddof=0) for d in data]
}

# --- Print LaTeX-formatted table ---
print("\\textbf{Statistic} & \\textbf{ImprovedGreedy} & \\textbf{Proposed Method} & \\textbf{ATS} & \\textbf{GA} \\\\")
for stat, values in stats.items():
    vals = [round(v, 1) for v in values]
    max_idx = int(np.argmax(vals))  # best (highest) value
    formatted = " & ".join([f"\\textbf{{{v}}}" if i == max_idx else f"{v}" for i, v in enumerate(vals)])
    print(f"{stat:<11}& {formatted} \\\\")

plt.figure(figsize=(8, 5))
parts = plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)

for pc in parts['bodies']:
    pc.set_facecolor("#87A7EB")  # light blue
    pc.set_edgecolor('black')
    pc.set_alpha(0.8)

plt.xticks(range(1, len(labels) + 1), labels)
plt.ylabel('Metric Value')
plt.title('Comparison of Metric Values Across Methods')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Combine into a tidy DataFrame (Seaborn expects this format)
data = pd.DataFrame({
    'Improved Greedy': improved_greedy_metric,
    'Proposed Method': evo_best_metric,
    'ATS': ats_best_metric,
    'GA': ga_best_metric
})

# Melt for long-form plotting
df_long = data.melt(var_name='Method', value_name='Metric')

# --- Plot ---
plt.figure(figsize=(9, 6))

# Boxplot: shows median, quartiles, outliers
sns.boxplot(x='Method', y='Metric', data=df_long, width=0.5, fliersize=0, color="#87A7EB", boxprops={'alpha':0.7})

# Swarmplot: shows all data points
sns.swarmplot(x='Method', y='Metric', data=df_long, color='black', alpha=0.7, size=5)

# --- Style ---
plt.title("Comparison of Metric Values Across Methods", fontsize=14,)
plt.xlabel("")
plt.ylabel("Metric Value")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()