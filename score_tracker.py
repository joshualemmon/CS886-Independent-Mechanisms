from matplotlib import pyplot as plt
import torch
import numpy as np
import seaborn as sns

class ScoreTracker:
	def __init__(self, transforms):
		self.score_dict = {}
		self.transforms = transforms
		self.num_experts = len(transforms)
		for t in self.transforms:
			temp_dict = {}
			for i in range(self.num_experts):
				temp_dict[i] = []
			self.score_dict[t] = temp_dict

	def update_score(self, expert, score, transform):
		self.score_dict[transform][expert].append(score)

	def running_avg(self, vals, n):
		return np.convolve(vals, np.ones((n,))/n)[(n-1):]

	def plot_scores(self, iters, save_loc):
		sns.set_theme()
		# fig = plt.figure()
		line_styles = {0:['g', 'solid'], 1:['b', 'solid'], 2:['r', 'solid'], 3:['m', 'solid'], \
		               4:['c', 'solid'], 5:['y', 'solid'], 6:['g', 'dashed'],7:['b', 'dashed'], \
		               8:['r', 'dashed'],9:['m', 'dashed'], 10:['c', 'dashed']}
		if self.num_experts == 10:
			fig, axes = plt.subplots(5,2, sharex=True, figsize=(10,10))
		else:
			fig, axes = plt.subplots(6,2, figsize=(10,10))
		axes = axes.flat
		for j, key in enumerate(self.score_dict.keys()):
			tf_dict = self.score_dict[key]
			# if self.num_experts == 10:
			# 	ax = fig.add_subplot(5,2,j+1)
			# else:
			# 	ax = fig.add_subplot(6,2,j+1)

			# ax.set_xlabel('Iterations')
			# ax.set_ylabel('Score')
			# ax.set_title(key)
			if (j + 1) == len(axes) or (j+2) == len(axes):
				axes[j].set_xlabel('Iterations')
			axes[j].set_ylabel('Score')
			axes[j].set_title(key)
			for i, ex in enumerate(tf_dict.keys()):
				avg = self.running_avg(tf_dict[ex], 50)
				col, sty = line_styles[i]
				axes[j].plot(avg[:-50], linestyle=sty, color=col)
		# fig.tight_layout()
		# plt.subplots_adjust(wspace=0.3, hspace=0.3)
		plt.savefig(f'{save_loc}/score_plot.png', bbox_inches='tight')


