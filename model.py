import torch
import torch.nn as nn
import numpy as np
import argparse
from matplotlib import pyplot as plt

class Expert(nn.Module):
	def __init__(self):
		super(Expert, self).__init__()
		self.model = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
			nn.BatchNorm2d(32)
			nn.ELU(inplace=True)
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
			nn.BatchNorm2d(32)
			nn.ELU(inplace=True)
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
			nn.BatchNorm2d(32)
			nn.ELU(inplace=True)
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
			nn.BatchNorm2d(32)
			nn.ELU(inplace=True)
			nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
			nn.Sigmoid()
			)
	def forward(self, x):
		return self.model(x)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.model = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
			nn.ELU(inplace=True)
			nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
			nn.ELU(inplace=True)
			nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
			nn.ELU(inplace=True)
			nn.AvgPool2d(kernel_size=2)
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
			nn.ELU(inplace=True)
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
			nn.ELU(inplace=True)
			nn.AvgPool2d(kernel_size=2)
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
			nn.ELU(inplace=True)
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
			nn.ELU(inplace=True)
			nn.AvgPool2d(kernel_size=2)
			nn.Linear(in_features=64, out_features=1024)
			nn.ELU(inplace=True)
			nn.Linear(in_features=1024, out_features=1)
			nn.Sigmoid()
			)

	def forward(self, x):
		return self.model(x)


def main(args):
	pass

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	main(ap.parse_args())