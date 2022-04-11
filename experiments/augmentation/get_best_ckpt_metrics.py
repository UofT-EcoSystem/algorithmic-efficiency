import numpy as np
import pandas as pd
import os

experiment = 'cifar10_baseline'
df = pd.read_csv(os.path.join('saved', experiment, 'all_measurements.csv'))

trial_best = pd.DataFrame()
num_trials = 5
for i in range(1, num_trials + 1):
  best = df.loc[df['trial_idx'] == i]['loss'].idxmin()
  trial_best = pd.concat([trial_best, df.iloc[[best]][['loss', 'accuracy', 'epoch', 'global_step', 'trial_idx']]])

print('Loss: %.4f+-%.4f' % (np.mean(trial_best['loss']), np.std(trial_best['loss'])))
print('Accuracy: %.4f+-%.4f' % (np.mean(trial_best['accuracy']), np.std(trial_best['accuracy'])))

print('\nBest Checkpoints:')
for i in range(num_trials):
  print('Trial%i - step=%i, epoch=%i' % (trial_best['trial_idx'].values[i], trial_best['global_step'].values[i], trial_best['epoch'].values[i]))
