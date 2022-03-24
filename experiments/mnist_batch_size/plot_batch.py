"""
This script will plot training loss vs training step using for a recorded CSV.

Author: Daniel Snider <danielsnider12@gmail.com>

Usage: python3 experiments/mnist_batch_size/plot_batch.py in.csv out.png
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

# Read Data
input_file = '/home/dans/algorithmic-efficiency/best_parameters_full.csv'
input_file = './out.csv'
input_file = sys.argv[1] if len(sys.argv) > 1 else input_file
output_file = sys.argv[2] if len(sys.argv) > 2 else 'plot.png'
df = pd.read_csv(input_file)


for arch in df['architecture'].unique():
    loc = df['architecture'] == arch
    num_steps_at_min_batch = df[loc].step_to_threshold.iloc[0]
    df.loc[loc, 'step_to_threshold'] = df.loc[loc, 'step_to_threshold'] / num_steps_at_min_batch

# Plot
sns.set_theme()
fig, ax = plt.subplots()
sns.lineplot(data=df, ax=ax, x='batch_size', y='step_to_threshold', hue='architecture')

# Style
ax.set_ylabel('Steps / (# Steps at B=4)')
ax.set_xlabel('Batch Size')
plt.xscale('log', base=2)
plt.yscale('log', base=2)


# Save
fig.savefig(output_file, transparent=False, dpi=160, bbox_inches="tight")
