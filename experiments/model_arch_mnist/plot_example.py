"""
This script will plot training loss vs training step using for a recorded CSV.

Author: Daniel Snider <danielsnider12@gmail.com>

Usage: python3 experiments/model_arch_mnist/plot_example.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read Data
input_file = './experiments/model_arch_mnist/logs/activation_relu-width_16-depth_1-dropout_0-batch_1024/configurable_mnist_jax/trial_1/measurements.csv'
df = pd.read_csv(input_file)

# Plot
sns.set_theme()
fig, ax = plt.subplots()
sns.lineplot(data=df, ax=ax, x='global_step', y='loss')

# Style
ax.set_ylabel('Loss')
ax.set_xlabel('Global Step')

# Save
fig.savefig('plot_loss.png', transparent=False, dpi=160, bbox_inches="tight")
