import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

def plot_comparisons_exp_1(baselines, augmented, fig_dir):
	'''
	'''
	baselines_df = pd.read_csv(baselines)
	augmented_df = pd.read_csv(augmented)
	# Calc average loss per trial
	# -------- Plots for trial-specific curves ----------

	baseline_gp = baselines_df.groupby('trial_idx')
	augmented_gp = augmented_df.groupby('trial_idx')
	
	for trial_idx, logs in baseline_gp:
		sns.lineplot(data=logs, 
					x='epoch', 
					y='accuracy', 
					alpha=0.4, 
					color='r', 
					label="CIFAR10 Baseline Runs" if trial_idx == 1 else "")
	for trial_idx, logs in augmented_gp:
		sns.lineplot(data=logs, 
					x='epoch', 
					y='accuracy', 
					alpha=0.4, 
					color='g', 
					label="CIFAR10 Augmented Runs" if trial_idx == 1 else "")
	plt.title("CIFAR10 Validation Accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Validation Accuracy")
	plt.legend()
	plt.savefig(os.path.join(fig_dir, "accuracy curves.png"), dpi=150)
	plt.clf()

	for trial_idx, logs in baseline_gp:
		sns.lineplot(data=logs, 
					x='epoch', 
					y='loss', 
					alpha=0.4, 
					color='r', 
					label="CIFAR10 Baseline Runs" if trial_idx == 1 else "")
	for trial_idx, logs in augmented_gp:
		sns.lineplot(data=logs, 
					x='epoch', 
					y='loss', 
					alpha=0.4, 
					color='g', 
					label="CIFAR10 Augmented Runs" if trial_idx == 1 else "")
	plt.title("CIFAR10 Validation Loss Curves")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.savefig(os.path.join(fig_dir, "loss curves.png"), dpi=150)
	plt.clf()
	
	sns.lineplot(x=[0.1,0.75],
				 y=[0.1,0.75],
				 color="b",
				 lw=2,
				 label="Equvalient Accuracy")

	all_differences = []
	min_len = 100
	for baseline_idx,baseline_logs in baseline_gp:
		for aug_trial_idx, aug_logs in augmented_gp:
			plt_first = min(len(baseline_logs['accuracy']), len(aug_logs['accuracy']))
			all_differences.append((sorted(aug_logs['accuracy'][:plt_first]), sorted(baseline_logs['accuracy'][:plt_first])))
			min_len = min(min_len, plt_first)
			sns.lineplot(x=sorted(baseline_logs['accuracy'][:plt_first]),
						    y=sorted(aug_logs['accuracy'][:plt_first]),
						    alpha=0.05,
						    color='r',
						    label="Accuracy Correlations" if baseline_idx == 1 and aug_trial_idx == 1 else "")

	aug_diff, baseline_diff = [], []
	for diff in all_differences:
		aug_diff.append(diff[0][:min_len])
		baseline_diff.append(diff[1][:min_len])
	
	aug_diff, baseline_diff = np.array(aug_diff), np.array(baseline_diff)
	aug_mean = np.mean(aug_diff, axis=0)
	baseline_mean = np.mean(baseline_diff, axis=0)
	sns.lineplot(x=baseline_mean, y=aug_mean, color='black', lw=1.5, label="Mean Comparison")

	plt.title("Comparison Between Validation Accuracy at Fixed Epochs")
	plt.xlabel("Baseline Accuracy")
	plt.ylabel("Augmented Accuracy")
	plt.legend()
	plt.savefig(os.path.join(fig_dir, "accuracy_comparisons.png"), dpi=150)
	plt.clf()
	
	# ------- Plots for per-epoch information ----------

	baseline_epochs = baselines_df.groupby('epoch')
	augmented_epochs = augmented_df.groupby('epoch')
	
	baseline_sigma = baseline_epochs.std()['accuracy']
	baseline_mean = baseline_epochs.mean()['accuracy']	
	augmented_sigma = augmented_epochs.std()['accuracy']
	augmented_mean = augmented_epochs.mean()['accuracy']	

	sns.lineplot(data=baseline_mean, 
				label="Baseline Mean Accuracy", 
				color="r")
	plt.fill_between(baseline_mean.index, 
					baseline_mean - baseline_sigma, 
					baseline_mean + baseline_sigma, 
					alpha=0.2, 
					color="r", 
					label="Baseline Std")
	sns.lineplot(data=augmented_mean, 
				label="Augmented Mean Accuracy",
				color="g")
	plt.fill_between(augmented_mean.index,
					augmented_mean - augmented_sigma, 
					augmented_mean + augmented_sigma, 
					alpha=0.2, 
					color="g", 
					label="Augmented Std")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.title("Mean CIFAR10 Validation Accuracy")
	plt.legend()
	plt.savefig(os.path.join(fig_dir, "accuracy mean_std"), dpi=150)
	plt.clf()
	# Do same thing for loss
	baseline_sigma = baseline_epochs.std()['loss']
	baseline_mean = baseline_epochs.mean()['loss']	
	augmented_sigma = augmented_epochs.std()['loss']
	augmented_mean = augmented_epochs.mean()['loss']	

	sns.lineplot(data=baseline_mean, 
				label="Baseline Mean Loss", 
				color="r")
	plt.fill_between(baseline_mean.index, 
					baseline_mean - baseline_sigma, 
					baseline_mean + baseline_sigma, 
					alpha=0.2, 
					color="r", 
					label="Baseline Std")
	sns.lineplot(data=augmented_mean, 
				label="Augmented Mean Loss",
				color="g")
	plt.fill_between(augmented_mean.index,
					augmented_mean - augmented_sigma, 
					augmented_mean + augmented_sigma, 
					alpha=0.2, 
					color="g", 
					label="Augmented Std")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Mean CIFAR10 Validation Loss")
	plt.legend()
	plt.savefig(os.path.join(fig_dir, "loss mean_std"), dpi=150)
	plt.clf()

def plot_comparisons_exp_2(result_dir, fig_dir):

	aug = pd.DataFrame()
	no_aug = pd.DataFrame()
	
	for pct_data in range(10,110,10):
		pct_dir = os.path.join(result_dir, "no_aug/{}%_data".format(pct_data),"all_measurements.csv")
		vals = pd.read_csv(pct_dir)
		vals['pct'] = pct_data
		no_aug = no_aug.append(vals)
		
		pct_dir = os.path.join(result_dir, "with_aug/{}%_data".format(pct_data),"all_measurements.csv")
		vals = pd.read_csv(pct_dir)
		vals['pct'] = pct_data
		aug = aug.append(vals)

	num_trials=5
	min_idx_aug = {} # min loss, indexed (pct data, trial num)
	min_idx_noaug = {}
	for pct in range(10,110,10):
		for i in range(1,num_trials+1):
			min_idx_noaug[(pct, i)] = no_aug[(no_aug['pct'] == pct) &
				(no_aug['trial_idx'] == i)]['loss'].idxmin()

			min_idx_aug[(pct, i)] = aug[(aug['pct'] == pct) &
				(aug['trial_idx'] == i)]['loss'].idxmin()

	diffs_mean_std = []
	for pct in range(10,110,10):
		diffs = []
		for i in range(1,num_trials+1):
			diffs.append(aug[(aug['pct'] == pct) & (aug['trial_idx'] == i)]['accuracy'][min_idx_aug[(pct,i)]] - \
					no_aug[(no_aug['pct'] == pct) & (no_aug['trial_idx'] == i)]['accuracy'][min_idx_noaug[(pct,i)]])

		diffs_mean_std.append([np.mean(diffs), np.std(diffs)])
	
	diffs_mean_std = np.array(diffs_mean_std)

	sns.lineplot(x= list(range(10,110,10)), y= diffs_mean_std[:,0], color="r")
	plt.fill_between(list(range(10,110,10)), 
					diffs_mean_std[:,0] - diffs_mean_std[:,1], 
					diffs_mean_std[:,0] + diffs_mean_std[:,1],
					color='r',
					alpha=0.2)
	plt.xlabel("Subset of Data Used (%)")
	plt.ylabel("Validation Accuracy Improvement Augmented vs Baseline(%)")
	plt.savefig(os.path.join(fig_dir, "Validation Accuracy at Minimum Loss Meanstd"), dpi=150)


	for pct_data in range(10,110,10):
		pct_dir = os.path.join(result_dir, "no_aug/{}%_data".format(pct_data))
		measurement_csv = os.path.join(pct_dir, "all_measurements.csv")

		vals = pd.read_csv(measurement_csv)
		per_epoch = vals.groupby('global_step')
		epoch_mean = per_epoch.mean()['accuracy']
		epoch_std = per_epoch.std()['accuracy']
		sns.lineplot(data=epoch_mean, label="{}% data".format(pct_data), alpha=0.6)
		plt.fill_between(epoch_mean.index,
			epoch_mean - epoch_std, epoch_mean + epoch_std, alpha=0.3)
	plt.ylabel("Accuracy(%)")
	plt.xlabel("Step")
	plt.savefig(os.path.join(fig_dir, "Validation Accuracy (No Aug)"), dpi=150)
	plt.clf()

	for pct_data in range(10,110,10):
		pct_dir = os.path.join(result_dir, "no_aug/{}%_data".format(pct_data))
		measurement_csv = os.path.join(pct_dir, "all_measurements.csv")

		vals = pd.read_csv(measurement_csv)
		per_epoch = vals.groupby('global_step')
		epoch_mean = per_epoch.mean()['loss']
		epoch_std = per_epoch.std()['loss']
		sns.lineplot(data=epoch_mean, label="{}% data".format(pct_data), alpha=0.6)
		plt.fill_between(epoch_mean.index,
			epoch_mean - epoch_std, epoch_mean + epoch_std, alpha=0.3)
	plt.ylabel("Validation Loss")
	plt.xlabel("Step")
	plt.savefig(os.path.join(fig_dir, "Validation Loss (No Aug)"), dpi=150)
	plt.clf()

	for pct_data in range(10,110,10):
		pct_dir = os.path.join(result_dir, "with_aug/{}%_data".format(pct_data))
		measurement_csv = os.path.join(pct_dir, "all_measurements.csv")

		vals = pd.read_csv(measurement_csv)
		per_epoch = vals.groupby('global_step')
		epoch_mean = per_epoch.mean()['accuracy']
		epoch_std = per_epoch.std()['accuracy']
		sns.lineplot(data=epoch_mean, label="{}% data".format(pct_data), alpha=0.6)
		plt.fill_between(epoch_mean.index,
			epoch_mean - epoch_std, epoch_mean + epoch_std, alpha=0.3)	
	plt.ylabel("Accuracy(%)")
	plt.xlabel("Step")
	plt.savefig(os.path.join(fig_dir, "Validation Accuracy (Aug)"), dpi=150)
	plt.clf()

	
	for pct_data in range(10,110,10):
		pct_dir = os.path.join(result_dir, "with_aug/{}%_data".format(pct_data))
		measurement_csv = os.path.join(pct_dir, "all_measurements.csv")

		vals = pd.read_csv(measurement_csv)
		per_epoch = vals.groupby('global_step')
		epoch_mean = per_epoch.mean()['loss']
		epoch_std = per_epoch.std()['loss']
		sns.lineplot(data=epoch_mean, label="{}% data".format(pct_data), alpha=0.6)
		plt.fill_between(epoch_mean.index,
			epoch_mean - epoch_std, epoch_mean + epoch_std, alpha=0.3)	
	plt.ylabel("Validation Loss")
	plt.xlabel("Step")
	plt.savefig(os.path.join(fig_dir, "Validation Loss (Aug)"), dpi=150)
	plt.clf()

	for i, pct_data in enumerate([10,20,30,50,70,100]):
		pct_dir_noaug = os.path.join(result_dir, "no_aug/{}%_data".format(pct_data))
		measurement_csv_noaug = os.path.join(pct_dir_noaug, "all_measurements.csv")
		pct_dir = os.path.join(result_dir, "with_aug/{}%_data".format(pct_data))
		measurement_csv = os.path.join(pct_dir, "all_measurements.csv")

		vals_noaug = pd.read_csv(measurement_csv_noaug)
		vals_aug = pd.read_csv(measurement_csv)

		per_epoch_noaug = vals_noaug.groupby('global_step').mean()['accuracy']
		per_epoch_aug = vals_aug.groupby('global_step').mean()['accuracy']

		min_val = min(np.shape(per_epoch_noaug)[0], np.shape(per_epoch_aug)[0])

		# sns.lineplot(data=per_epoch_noaug[:min_val],alpha=0.3)
		# sns.lineplot(data=per_epoch_aug[:min_val], alpha=0.3)

		# print(per_epoch_noaug[:min_val])
		# print(per_epoch_aug[:min_val])
		plt.fill_between(per_epoch_aug.index[:min_val], 
					per_epoch_noaug[:min_val], 
					per_epoch_aug[:min_val],
					alpha=0.4,
					label="{}% data".format(pct_data))

	plt.ylim([0.4,0.80])
	plt.ylabel("Accuracy (%)")
	plt.xlabel("Step")
	plt.legend(loc='lower right')
	plt.savefig(os.path.join(fig_dir, "Validation Accuracy Improvements"), dpi=150)
	plt.clf()


if __name__=="__main__":
	# change to args input if you want
	baselines_file = "./experiments/augmentation/saved/cifar10_baseline/all_measurements.csv"
	augmented_file = "./experiments/augmentation/saved/cifar10_aug_offline/all_measurements.csv"
	fig_dir = "./experiments/augmentation/figures"
	plot_comparisons_exp_1(baselines_file, augmented_file, fig_dir)

	results_dir = "./experiments/augmentation/saved/aug_cifar10_all_tests_redo"
	plot_comparisons_exp_2(results_dir, fig_dir)