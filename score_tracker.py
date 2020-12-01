from matplotlib import pyplot as plt
import torch
import numpy as np

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
		fig = plt.figure()
		line_styles = {0:['g', 'solid'], 1:['b', 'solid'], 2:['r', 'solid'], 3:['m', 'solid'], 5:['y', 'solid'], 6:['c', 'solid'], 7:['g', 'dashed'],8:['b', 'dashed'],9:['r', 'dashed'],10:['m', 'dashed']}
		for j, key in enumerate(self.score_dict.keys()):
			tf_dict = self.score_dict[key]
			ax = fig.add_subplot(5,2,j+1)
			ax.set_xlabel('Iterations')
			ax.set_ylabel('Disc. Score')
			ax.set_title(key)
			for i, ex in enumerate(tf_dict.keys()):
				avg = running_avg(tf_dict[ex], 50)
				col, sty = line_styles[i]
				ax.plot(avg, linestyle=sty, color=col)
		plt.savefig(f'{save_loc}/score_plot.png', bbox_inches='tight')


