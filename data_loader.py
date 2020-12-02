import numpy as np
import torchvision as tv
import torch
import torch.utils.data as udata
import PIL
from PIL import Image
import numpy as np
import random

class DataLoader:
	def __init__(self, tf_type, tf_data, cl_data, device):
		self.tf_type = tf_type
		if tf_type == 0:
			self.transforms = ['UP', 'UPRIGHT', 'RIGHT', 'DOWNRIGHT', 'DOWN', 'DOWNLEFT', 'LEFT', 'UPLEFT', 'NOISE', 'INVERT']
		else:
			self.transforms = ['R45', 'R90', 'R135', 'R180', 'R225', 'R270', 'R315', 'XSTR', 'YSTR', 'XCOM', 'YCOM']
		self.num_transforms = len(self.transforms)
		self.transform_data = tf_data
		self.clean_data = cl_data
		self.device = device

	def sample_each_transform(self):
		imgs = []
		rand_imgs = random.sample(self.transform_data, 200)
		for tf in self.transforms:
			img = random.sample([t[0] for t in rand_imgs if t[1] == tf], 1)[0]
			# img = tv.transforms.ToTensor()(img)
			img = img.to(self.device)
			imgs.append(img.unsqueeze(0))
		return torch.stack([i for i in imgs]), self.transforms

	def get_sample(self, mb):
		clean = random.sample(self.clean_data, mb)
		cl_imgs, cl_lbls = zip(*clean)
		transformed = random.sample(self.transform_data, mb)
		tf_imgs, tf_types, tf_lbls = zip(*transformed)

		return tf_imgs, cl_imgs

