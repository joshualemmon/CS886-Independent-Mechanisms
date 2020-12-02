import numpy as np
import torchvision as tv
import torch
import torch.utils.data as udata
import PIL
from PIL import Image
import numpy as np
import random

class DataTransform:
	def __init__(self, tf_type, data, labels, gpu):
		self.tf_type = tf_type
		if tf_type == 0:
			self.transforms = ['UP', 'UPRIGHT', 'RIGHT', 'DOWNRIGHT', 'DOWN', 'DOWNLEFT', 'LEFT', 'UPLEFT', 'NOISE', 'INVERT']
		else:
			self.transforms = ['R45', 'R90', 'R135', 'R180', 'R225', 'R270', 'R315', 'XSTR', 'YSTR', 'XCOM', 'YCOM']
		self.num_transforms = len(self.transforms)
		mid = int(len(data)/2)
		self.transform_data = data[:mid]
		self.transform_labels = labels[:mid]
		self.clean_data = data[mid:]
		self.clean_labels = labels[mid:]
		self.gpu = True if gpu else False

	def sample_each_transform(self):
		if self.tf_type == 0:
			return self.sample_each_paper_transform()
		else:
			return self.sample_each_expanded_transform()

	def sample_each_paper_transform(self):
		tf_imgs = []
		imgs = random.sample(self.transform_data, self.num_transforms)
		for i, img in enumerate(imgs):
			if i%10 == 0:
				tf_img = self.translate(img, (0,-4)) # translate up 4 pixels
			elif i%10 == 1:
				tf_img = self.translate(img, (4,-4)) # translate up and right 4 pixels
			elif i%10 == 2:
				tf_img = self.translate(img, (4,0)) # translate right 4 pixels
			elif i%10 == 3:
				tf_img = self.translate(img, (4,4)) # translate down and right 4 pixels
			elif i%10 == 4:
				tf_img = self.translate(img, (0,4)) # translate down 4 pixels
			elif i%10 == 5:
				tf_img = self.translate(img, (-4,4)) # translate down and left 4 pixels
			elif i%10 == 6:
				tf_img = self.translate(img, (-4,0)) # translate left 4 pixels
			elif i%10 == 7:
				tf_img = self.translate(img, (-4,-4)) # translate up and left 4 pixels
			elif i%10 == 8:
				tf_img = self.add_noise(img) # add gaussian noise with mean 0 and variance 0.25
			elif i%10 == 9:
				tf_img = self.invert(img) # invert pixel values as p' = 1 - p
			tf_imgs.append(tf_img.unsqueeze(0))
		return torch.stack([r for r in tf_imgs]), self.transforms

	def sample_each_expanded_transform(self):
		tf_imgs = []
		imgs = random.sample(self.transform_data, self.num_transforms)
		for i, img in enumerate(imgs):
			if i%11 == 0:
				tf_img = self.rotate(img, 45)
			elif i%11 == 1:
				tf_img =  self.rotate(img, 90)
			elif i%11 == 2:
				tf_img =  self.rotate(img, 135)
			elif i%11 == 3:
				tf_img = self.rotate(img, 180)
			elif i%11 == 4:
				tf_img = self.rotate(img, 225)
			elif i%11 == 5:
				tf_img = self.rotate(img, 270)
			elif i%11 == 6:
				tf_img = self.rotate(img, 315)
			elif i%11 == 7:
				tf_img = self.stretch(img, 'x', 1.75)
			elif i%11 == 8:
				tf_img = self.stretch(img, 'y', 1.75)
			elif i%11 == 9:
				tf_img = self.compress(img, 'x', 1.75)
			elif i%11 == 10:
				tf_img = self.compress(img, 'y', 1.75)
			tf_imgs.append(tf_img)
		return tf_imgs, self.transforms

	def get_transformed_sample(self, sample_size):
		clean = random.sample(self.clean_data, sample_size)
		transformed = random.sample(self.transform_data, sample_size) 
		if self.tf_type == 0:
			transformed = self.transform_paper(transformed)
		else:
			transformed = self.transform_expanded(transformed)
		return transformed, clean

	def transform_paper(self, imgs):
		tf_imgs = []
		for img in imgs:
			i = np.random.randint(self.num_transforms)
			if i%10 == 0:
				tf_img = self.translate(img, (0,-4)) # translate up 4 pixels
			elif i%10 == 1:
				tf_img = self.translate(img, (4,-4)) # translate up and right 4 pixels
			elif i%10 == 2:
				tf_img = self.translate(img, (4,0)) # translate right 4 pixels
			elif i%10 == 3:
				tf_img = self.translate(img, (4,4)) # translate down and right 4 pixels
			elif i%10 == 4:
				tf_img = self.translate(img, (0,4)) # translate down 4 pixels
			elif i%10 == 5:
				tf_img = self.translate(img, (-4,4)) # translate down and left 4 pixels
			elif i%10 == 6:
				tf_img = self.translate(img, (-4,0)) # translate left 4 pixels
			elif i%10 == 7:
				tf_img = self.translate(img, (-4,-4)) # translate up and left 4 pixels
			elif i%10 == 8:
				tf_img = self.add_noise(img) # add gaussian noise with mean 0 and variance 0.25
			elif i%10 == 9:
				tf_img = self.invert(img) # invert pixel values as p' = 1 - p
			tf_imgs.append(tf_img)
		return tf_imgs

	def transform_expanded(self, imgs):
		tf_imgs = []
		for img in imgs:
			i = np.random.randint(self.num_transforms)
			if i%11 == 0:
				tf_img = self.rotate(img, 45)
			elif i%11 == 1:
				tf_img =  self.rotate(img, 90)
			elif i%11 == 2:
				tf_img =  self.rotate(img, 135)
			elif i%11 == 3:
				tf_img = self.rotate(img, 180)
			elif i%11 == 4:
				tf_img = self.rotate(img, 225)
			elif i%11 == 5:
				tf_img = self.rotate(img, 270)
			elif i%11 == 6:
				tf_img = self.rotate(img, 315)
			elif i%11 == 7:
				tf_img = self.stretch(img, 'x', 1.75)
			elif i%11 == 8:
				tf_img = self.stretch(img, 'y', 1.75)
			elif i%11 == 9:
				tf_img = self.compress(img, 'x', 1.75)
			elif i%11 == 10:
				tf_img = self.compress(img, 'y', 1.75)
			tf_imgs.append(tf_img)
		return tf_imgs

	def add_noise(self, img):
		img = img.squeeze(0)
		img = img + torch.randn(img.size())*0.25
		img = img - img.min()
		img = img/img.max()
		if self.gpu:
			img.cuda()
		return img

	def translate(self, img, d):
		img = tv.transforms.ToPILImage()(img.squeeze(0))
		img = tv.transforms.functional.affine(img, angle=0, translate=d, scale=1, shear=0, fillcolor=1) #fillcolor not working?
		img = tv.transforms.ToTensor()(img).squeeze(0)

		# fillcolour not working, fill in black borders manually
		if d[0] < 0: #LEFT
			if d[1] < 0: #UP
				img[-4:,:] = 1.0
				img[:,-4:] = 1.0
			elif d[1] > 0: #DOWN
				img[:4, :] = 1.0
				img[:,-4:] = 1.0
			else:
				img[:,-4:] = 1.0
		elif d[0] > 0: #RIGHT
			if d[1] < 0: #UP
				img[-4:,:] = 1.0
				img[:,:4] = 1.0
			elif d[1] > 0: #DOWN 
				img[:4,:] = 1.0
				img[:,:4] = 1.0
			else:
				img[:,:4] = 1.0
		else:
			if d[1] < 0:
				img[-4:,:] = 1.0
			elif d[1] > 0:
				img[:4, :]= 1.0
		if self.gpu:
			img.cuda()
		return img

	# Invert pixel values
	def invert(self, img):
		img = torch.ones_like(img) - img
		if self.gpu:
			img.cuda()
		return img.squeeze(0)
		# return tv.transforms.ToPILImage()(img.squeeze(0))

	# Rotate image by angle clockwise
	def rotate(self, img, angle):
		img = tv.transforms.ToPILImage()(img.squeeze(0))
		img = tv.transforms.functional.pad(img, 100, padding_mode='edge')
		img = tv.transforms.functional.rotate(img, angle=-angle)
		img = tv.transforms.ToTensor()(img)
		img = torch.narrow(img, 1, 100, 32)
		img = torch.narrow(img, 2, 100, 32)
		if self.gpu:
			img.cuda()
		return img

	def stretch(self, img, axis, scale):
		img = tv.transforms.ToPILImage()(img.squeeze(0))
		x, y = img.size
		stretched = x*scale
		if axis == 'x':
			img = tv.transforms.functional.resize(img, size=(x,int(scale*y)))
			img = tv.transforms.ToTensor()(img)
			img = torch.narrow(img, 2, int((stretched-32)/2), 32)
		else:
			img = tv.transforms.functional.resize(img, size=(int(1.75*x),y))
			img = tv.transforms.ToTensor()(img)
			img = torch.narrow(img, 1, int((stretched-32)/2), 32)
		img = tv.transforms.ToTensor()(img)
		if self.gpu:
			img.cuda()
		return img

	def compress(self, img, axis, scale):
		img = tv.transforms.ToPILImage()(img.squeeze(0))
		x, y = img.size
		compressed = int(x/scale)
		if axis == 'x':
			img = tv.transforms.functional.resize(img, size=(x,int(y/1.75)))
			img = tv.transforms.functional.pad(img, (int((32-compressed)/2), 0), padding_mode='edge')
		else:
			img = tv.transforms.functional.resize(img, size=(int(x/1.75),y))
			img = tv.transforms.functional.pad(img, (0,int((32-compressed)/2)), padding_mode='edge')

		img = tv.transforms.ToTensor()(img)
		if self.gpu:
			img.cuda()
		return img