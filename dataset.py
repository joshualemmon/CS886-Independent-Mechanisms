import numpy as np
import os
import argparse
import torchvision as tv
import torch
import torch.utils.data as udata
import PIL
from PIL import Image

def main(args):
	ds = args.dataset if args.dataset else 'MNIST'
	dl = args.download if args.download else False
	tf = args.transform if args.transform else 0

	images, labels  = get_dataset(ds, dl)
	transform_datasets(images, labels, ds, tf)

def get_dataset(ds, dl):
	transform = tv.transforms.Compose([tv.transforms.Pad(2),tv.transforms.ToTensor()])
	if ds == 'MNIST':
		dataset = tv.datasets.MNIST(root='./datasets/MNIST/original', train=True, download=dl, transform=transform)
	elif ds == 'omniglot':
		dataset = tv.datasets.Omniglot(root='./datasets/omniglot/original', train=True, download=dl, transform=transform)
	loader = loader = udata.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
	imgs, lbls = iter(loader).next()
	for i in range(len(imgs)):
		imgs[i] = torch.ones_like(imgs[i]) - imgs[i]

	return imgs, lbls

def transform_datasets(images, labels, ds, tf):
	mid = int(len(images)/2)
	training_transformed = images[:mid]
	labels_transformed = labels[:mid]
	training_clean = images[mid:]
	labels_clean = labels[:mid]
	if tf == 0:
		training_transformed = transform_paper(training_transformed, labels_transformed)
	elif tf == 1:
		training_transformed = transform_new(training_transformed, labels_transformed)
	save_dataset(training_transformed, training_clean, labels_clean, ds, tf)

def save_dataset(transf, c_imgs, c_lbls, ds, tf):
	label_map = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine'}
	tf_type = 'paper_transformations' if tf == 0 else 'new_transformations' 
	for i in range(len(transf)):
		img = transf[i][0]
		tf_class = transf[i][1]
		label = transf[i][2]
		img.save(f'./datasets/{ds}/{tf_type}/transformed/{label_map[label.item()]}_{tf_class}_{i}.png')
	for i in range(len(c_imgs)):
		img = tv.transforms.ToPILImage()(c_imgs[i])
		label = c_lbls[i]
		img.save(f'./datasets/{ds}/{tf_type}/clean/{label_map[label.item()]}_{i}.png')

def transform_paper(imgs, lbls):
	tf_imgs = []
	tf_type = '' 
	for i, img in enumerate(imgs):
		if i%10 == 0:
			tf_img = translate(img, (0,-4)) # translate up 4 pixels
			tf_type = 'UP'
		elif i%10 == 1:
			tf_img = translate(img, (4,-4)) # translate up and right 4 pixels
			tf_type = 'UPRIGHT'
		elif i%10 == 2:
			tf_img = translate(img, (4,0)) # translate right 4 pixels
			tf_type = 'RIGHT'
		elif i%10 == 3:
			tf_img = translate(img, (4,4)) # translate down and right 4 pixels
			tf_type = 'DOWNRIGHT'
		elif i%10 == 4:
			tf_img =translate(img, (0,4)) # translate down 4 pixels
			tf_type = 'DOWN'
		elif i%10 == 5:
			tf_img = translate(img, (-4,4)) # translate down and left 4 pixels
			tf_type = 'DOWNLEFT'
		elif i%10 == 6:
			tf_img =translate(img, (-4,0)) # translate left 4 pixels
			tf_type = 'LEFT'
		elif i%10 == 7:
			tf_img = translate(img, (-4,-4)) # translate up and left 4 pixels
			tf_type = 'UPLEFT'
		elif i%10 == 8:
			tf_img = add_noise(img) # add gaussian noise with mean 0 and variance 0.25
			tf_type = 'NOISE'
		elif i%10 == 9:
			tf_img = invert(img) # invert pixel values as p' = 1 - p
			tf_type = 'INVERT'
		tf_imgs.append((tf_img, tf_type, lbls[i]))
	return tf_imgs

def transform_new(imgs, lbls):
	tf_imgs = []
	tf_type = ''
	for i, img in enumerate(imgs):
		if i%11 == 0:
			tf_img = rotate(img, 45)
			tf_type = 'R45'
		elif i%11 == 1:
			tf_img =  rotate(img, 90)
			tf_type = 'R90'
		elif i%11 == 2:
			tf_img =  rotate(img, 135)
			tf_type = 'R135'
		elif i%11 == 3:
			tf_img = rotate(img, 180)
			tf_type = 'R180'
		elif i%11 == 4:
			tf_img = rotate(img, 225)
			tf_type = 'R225'
		elif i%11 == 5:
			tf_img = rotate(img, 270)
			tf_type = 'R270'
		elif i%11 == 6:
			tf_img = rotate(img, 315)
			tf_type = 'R315'
		elif i%11 == 7:
			tf_img = stretch(img, 'x', 1.75)
			tf_type = 'XSTR'
		elif i%11 == 8:
			tf_img = stretch(img, 'y', 1.75)
			tf_type = 'YSTR'
		elif i%11 == 9:
			tf_img = compress(img, 'x', 1.75)
			tf_type = 'XCOM'
		elif i%11 == 10:
			tf_img = compress(img, 'y', 1.75)
			tf_type = 'YCOM'
		tf_imgs.append((tf_img, tf_type, lbls[i]))
	return tf_imgs

def add_noise(img):
	img = img.squeeze(0)
	img = img + torch.randn(img.size())*0.25
	img -= img.min()
	img /= img.max()
	return tv.transforms.ToPILImage()(img)

def translate(img, d):
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

	return tv.transforms.ToPILImage()(img)

# Invert pixel values
def invert(img):
	img = torch.ones_like(img) - img
	return tv.transforms.ToPILImage()(img.squeeze(0))

# Rotate image by angle clockwise
def rotate(img, angle):
	img = tv.transforms.ToPILImage()(img.squeeze(0))
	img = tv.transforms.functional.pad(img, 100, padding_mode='edge')
	img = tv.transforms.functional.rotate(img, angle=-angle)
	img = tv.transforms.ToTensor()(img)
	img = torch.narrow(img, 1, 100, 32)
	img = torch.narrow(img, 2, 100, 32)
	
	return tv.transforms.ToPILImage()(img.squeeze(0))	

def stretch(img, axis, scale):
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
	return tv.transforms.ToPILImage()(img.squeeze(0))	

def compress(img, axis, scale):
	img = tv.transforms.ToPILImage()(img.squeeze(0))
	x, y = img.size
	compressed = int(x/scale)
	if axis == 'x':
		img = tv.transforms.functional.resize(img, size=(x,int(y/1.75)))
		img = tv.transforms.functional.pad(img, (int((32-compressed)/2), 0), padding_mode='edge')
	else:
		img = tv.transforms.functional.resize(img, size=(int(x/1.75),y))
		img = tv.transforms.functional.pad(img, (0,int((32-compressed)/2)), padding_mode='edge')

	return img	

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument('-ds', '--dataset', type=str)
	ap.add_argument('-dl', '--download', type=bool)
	ap.add_argument('-tf', '--transform', type=int)

	main(ap.parse_args())