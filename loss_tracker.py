from matplotlib import pyplot as plt
import torch
import numpy as np
import seaborn as sns

class LossTracker:
	def __init__(self, num_experts):
		self.num_experts = num_experts
		self.disc_losses = []
		self.expert_losses = {}
		for i in range(num_experts):
			self.expert_losses[i] = []

	def update_expert_loss(self, expert, loss):
		self.expert_losses[expert].append(loss)

	def update_disc_loss(self, loss):
		self.disc_losses.append(loss)

	def plot_losses(self, save_loc):
		sns.set_theme()
		line_styles = {0:['g', 'solid'], 1:['b', 'solid'], 2:['r', 'solid'], 3:['m', 'solid'], \
		               4:['c', 'solid'], 5:['y', 'solid'], 6:['g', 'dashed'],7:['b', 'dashed'], \
		               8:['r', 'dashed'],9:['m', 'dashed'], 10:['c', 'dashed']}
		fig, axes = plt.subplots(2,1, sharex=True)
		axes = axes.flat
		axes[0].set_ylabel('Loss')
		axes[0].set_title('Expert Losses')
		axes[1].set_ylabel('Loss')
		axes[1].set_title('Discriminator Loss')
		for i in range(self.num_experts):
			col, sty = line_styles[i]
			axes[0].plot(self.expert_losses[i], linestyle=sty, color=col)
		axes[1].plot(self.disc_losses, linestyle='solid', color='b')
		plt.savefig(f'{save_loc}/loss_plot.png', bbox_inches='tight')


