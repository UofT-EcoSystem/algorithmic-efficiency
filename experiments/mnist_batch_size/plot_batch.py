"""
This script will plot training loss vs training step using for a recorded CSV.

Author: Daniel Snider <danielsnider12@gmail.com>

Usage: python3 experiments/mnist_batch_size/plot_batch.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read Data
input_file = '/home/dans/algorithmic-efficiency/best_parameters_full.csv'
input_file = './out.csv'
df = pd.read_csv(input_file)

# Plot
sns.set_theme()
fig, ax = plt.subplots()
sns.lineplot(data=df, ax=ax, x='batch_size', y='step_to_threshold', hue='architecture')

# Style
ax.set_ylabel('Steps')
ax.set_xlabel('Batch Size')
plt.xscale('log', base=2)
plt.yscale('log', base=2)


# Save
fig.savefig('plot_batch_test.png', transparent=False, dpi=160, bbox_inches="tight")
