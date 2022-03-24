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

df_orig = df

for arch in df_orig['architecture'].unique():
    arch_loc = df_orig['architecture'] == arch
    min_batch =  df_orig[arch_loc].batch_size.min()
    min_batch_loc = (df_orig['architecture'] == arch) & (df_orig['batch_size'] == min_batch)
    num_steps_at_min_batch = df_orig[min_batch_loc].step_to_threshold.iloc[0]
    df.loc[arch_loc, 'step_to_threshold'] = df_orig.loc[arch_loc, 'step_to_threshold'] / num_steps_at_min_batch
    print('min_batch')
    print(min_batch)


# Plot
sns.set_theme()
fig, ax = plt.subplots()
g = sns.lineplot(data=df, ax=ax, x='batch_size', y='step_to_threshold', hue='architecture')

# Style
ax.set_ylabel(f'Steps / (# Steps at B={min_batch})')
ax.set_xlabel('Batch Size')
plt.xscale('log', base=2)
plt.yscale('log', base=2)
g.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Save
fig.savefig(output_file, transparent=False, dpi=160, bbox_inches="tight")
