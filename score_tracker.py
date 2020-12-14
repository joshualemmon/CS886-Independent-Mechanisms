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

	def plot_scores(self, save_loc):
		sns.set_theme()
		line_styles = {0:['g', 'solid'], 1:['b', 'solid'], 2:['r', 'solid'], 3:['m', 'solid'], \
		               4:['c', 'solid'], 5:['y', 'solid'], 6:['g', 'dashed'],7:['b', 'dashed'], \
		               8:['r', 'dashed'],9:['m', 'dashed'], 10:['c', 'dashed']}
		if self.num_experts == 10:
			fig, axes = plt.subplots(2,5, figsize=(60,12))
		else:
			fig, axes = plt.subplots(3,4, figsize=(60,24))
		axes = axes.flat
		for j, key in enumerate(self.score_dict.keys()):
			tf_dict = self.score_dict[key]
			axes[j].set_xlabel('Iterations')

			axes[j].set_ylabel('Score')
			axes[j].set_title(key)
			for i, ex in enumerate(tf_dict.keys()):
				avg = self.running_avg(tf_dict[ex], 50)
				col, sty = line_styles[i]
				axes[j].plot(avg[:-50], linestyle=sty, color=col)
		if self.num_experts == 11:
			fig.delaxes(axes[-1])

		plt.savefig(f'{save_loc}/score_plot.png', bbox_inches='tight')


