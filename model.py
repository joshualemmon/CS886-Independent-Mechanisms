import torch
import torch.nn as nn
import numpy as np

class Expert(nn.Module):
	def __init__(self):
		super(Expert, self).__init__()
		self.model = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ELU(),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ELU(),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ELU(),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ELU(),
			nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
			nn.Sigmoid())
	def forward(self, x):
		return self.model(x)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
			nn.ELU(),
			nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
			nn.ELU(),
			nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
			nn.ELU(),
			nn.AvgPool2d(kernel_size=2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
			nn.ELU(),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
			nn.ELU(),
			nn.AvgPool2d(kernel_size=2),
			)
		self.layer3 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
			nn.ELU(),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
			nn.ELU(),
			nn.AvgPool2d(kernel_size=2))
		self.layer4 = nn.Sequential(
			nn.Linear(in_features=4*4*64, out_features=1024),
			nn.ELU(),
			nn.Linear(in_features=1024, out_features=1),
			nn.Sigmoid())
		self.flatten = nn.Flatten()
	def forward(self, x, mb):
		out1 = self.layer1(x)
		out2 = self.layer2(out1)
		out3 = self.layer3(out2)
		out3 = self.flatten(out3)
		out4 = self.layer4(out3)
		return out4
