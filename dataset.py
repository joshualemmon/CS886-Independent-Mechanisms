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

	images, labels  = get_dataset(ds, dl)
	save_dataset(images, labels, ds)

def get_dataset(ds, dl):
	transform = tv.transforms.Compose([tv.transforms.Pad(2),tv.transforms.ToTensor()])
	if ds == 'MNIST':
		dataset = tv.datasets.MNIST(root='./datasets/MNIST/download', train=True, download=dl, transform=transform)
	elif ds == 'omniglot':
		dataset = tv.datasets.Omniglot(root='./datasets/omniglot/download', train=True, download=dl, transform=transform)
	loader = loader = udata.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
	imgs, lbls = iter(loader).next()
	for i in range(len(imgs)):
		imgs[i] = torch.ones_like(imgs[i]) - imgs[i]

	return imgs, lbls

def save_dataset(imgs, lbls, ds):
	label_map = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine'}
	for i in range(len(imgs)):
		img = tv.transforms.ToPILImage()(imgs[i])
		label = lbls[i]
		img.save(f'./datasets/{ds}/images/{label_map[label.item()]}_{i}.png')

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument('-ds', '--dataset', type=str)
	ap.add_argument('-dl', '--download', type=bool)

	main(ap.parse_args())