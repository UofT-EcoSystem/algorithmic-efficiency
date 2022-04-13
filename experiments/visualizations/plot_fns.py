import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_trial_mean_std(csvs, labels, metric):
  for i, csv in enumerate(csvs):
    df = pd.read_csv(csv)
    gp = df.groupby('epoch')
    mean = gp.mean()[metric]
    sigma = gp.std()[metric]
    sns.lineplot(data=mean,
      label=labels[i] + ' mean'
      )
    plt.fill_between(mean.index, 
					mean - sigma, 
					mean + sigma, 
					alpha=0.2, 
					label=labels[i] + " Std")

  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel(metric)

# csvs = [
#   './experiments/augmentation/saved/cifar10_baseline/all_measurements.csv',
#   './experiments/augmentation/saved/cifar10_aug_offline/all_measurements.csv',
#   './experiments/augmentation/saved/aug_cifar10_all_tests_redo/with_aug/100%_data/all_measurements.csv'
# ]
# labels = [
#   'baseline',
#   'offline augmentation',
#   'online augmentation'
# ]
# plot_trial_mean_std(csvs, labels, metric='loss')
# plt.xlim(right=30)
# plt.show()

csvs = [
  './experiments/augmentation/saved/aug_cifar10_all_tests_redo/with_aug/100%_data/all_measurements.csv',
  './experiments/augmentation/saved/aug_cifar10_all_tests_redo/with_aug/90%_data/all_measurements.csv',
  './experiments/augmentation/saved/aug_cifar10_all_tests_redo/with_aug/80%_data/all_measurements.csv',
  './experiments/augmentation/saved/aug_cifar10_all_tests_redo/with_aug/70%_data/all_measurements.csv',
  './experiments/augmentation/saved/aug_cifar10_all_tests_redo/with_aug/60%_data/all_measurements.csv',
  './experiments/augmentation/saved/aug_cifar10_all_tests_redo/with_aug/50%_data/all_measurements.csv',
  './experiments/augmentation/saved/aug_cifar10_all_tests_redo/with_aug/40%_data/all_measurements.csv',
  './experiments/augmentation/saved/aug_cifar10_all_tests_redo/with_aug/30%_data/all_measurements.csv',
  './experiments/augmentation/saved/aug_cifar10_all_tests_redo/with_aug/20%_data/all_measurements.csv',
  './experiments/augmentation/saved/aug_cifar10_all_tests_redo/with_aug/10%_data/all_measurements.csv',
]
labels = [
  '100%',
  '90%',
  '80%',
  '70%',
  '60%',
  '50%',
  '40%',
  '30%',
  '20%',
  '10%',
]

plot_trial_mean_std(csvs, labels, metric='loss')
plt.show()