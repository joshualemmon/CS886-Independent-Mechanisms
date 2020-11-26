import torch
import torch.nn as nn
import torchvision as tv
import numpy as np
import argparse
import model.py as models
import argparse
import os
import PIL
from PIL import Image
from matplotlib import plt
import random


def main(args):
	dataset = args.dataset if args.dataset else 0
	transforms = args.transforms if args.transforms else 0
	maxiter = args.iter if args.iter else 2000
	aii_iter = args.aprox_iter if args.aprox_iter else 500
	aii_mse = args.aprox_mse if args.aprox_mse else 0.002
	num_exp = args.num_experts if args.num_experts else 10
	mb = args.mini_batch if args.minibatch else 32

	tf_data, cl_data = load_data(dataset, transforms)
	get_tf_sample(tf_data, mb)
	return
	experts, disc = get_models(num_exp)
	ex_opts, disc_opt = get_optimizers(experts, disc)
	experts, ex_opts = approx_id_init(experts, ex_opts, tf_data[0], aii_iter, aii_mse)
	experts, disc, metrics = train(experts, ex_opts, disc, disc_opt, tf_data, cl_data, maxiter, mb)

def load_data(dataset, transforms):
	trans_data = []
	clean_data = []

	t_dir = f'./datasets/{"MNIST" if dataset == 0 else "omniglot"}/{"paper_transformations" if transforms == 0 else "new_transformations"}/transformed'
	c_dir = f'./datasets/{"MNIST" if dataset == 0 else "omniglot"}/{"paper_transformations" if transforms == 0 else "new_transformations"}/clean'

	for fname in os.listdir(t_dir):
		lbl = fname.split('_')[0]
		tf_type = fname.split('_')[1]
		img = tv.transforms.ToTensor()(Image.open(t_dir + '/' + fname))
		trans_data.append((img, lbl, tf_type))
	for fname in os.listdir(c_dir):
		lbl = fname.split('_')[0]
		img = tv.transforms.ToTensor()(Image.open(c_dir + '/' + fname))
		clean_data.append((img, lbl))

	return trans_data, clean_data

def get_models(num_exp):
	experts = []
	for i in range(num_exp):
		experts.append(models.Expert())
	disc = models.Discriminator()

	return experts, discrim

def get_optimizers(experts, discrim):
	ex_opts = []

	for exp in experts:
		ex_opts.append(torch.optim.Adam(exp.parameters()))

	disc_opt = torch.optim.Adam(discrim.parameters())

	return ex_opts, disc_opt

def save_tensor_img(img, fname):
	img = tv.transforms.ToPILImage()(img.squeeze(0))
	img.save(fname)

def approx_id_init(experts, opts, data, maxiter, err_th):
	init_experts = []
	init_opts = []

	loss = nn.MSELoss()
	for i in range(maxiter):
		sample = random.choice(data)
		indexes = []
		for j in range(len(experts)):
			opts[j].zero_grad()
			out = experts[j](sample)

			err = loss(sample, out)
			if err < err_th:
				init_experts.append(experts[j])
				init_opts.append(opts[j])
				indexes.append(j)
				save_tensor_img(sample, f'expert_{i}_in.png')
				save_tensor_img(out, f'expert_{i}_out.png')
			else:
				err.backward()
				opt[j].step()

		if len(indexes > 0):
			indexes.reverse()
			for j in indexes:
				del experts[j]
				del opts[j]
		if len(experts) == 0:
			break

	for i, e in enumerate(experts):
		init_experts.append(e)
		init_opts.append(opts[i])

	return init_experts, init_opts 

def disc_loss(x, c):
	e_sum = torch.zeros(1)
	for j in range(len(c)):
		e_sum += torch.log(1-c[j])

	return torch.log(x) + 1/len(c)*e_sum

def expert_loss(c):
	return torch.log(c)

def get_tf_sample(data, mb):
	sample = random.sample(data, mb)
	print(len(sample))

def train(experts, ex_opts, disc, disc_opt, tf_data, cl_data, maxiter, mb):
	metrics = {}

	for i in range(maxiter):
		t_sample = get_tf_sample(tf_data, mb)
		c_sample = get_cl_sample(cl_data, mb)


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-ds', '--dataset', type=int)
	ap.add_argument('-tf', '--transforms', type=int)
	ap.add_argument('-i', '--iter', type=int)
	ap.add_argument('-aii', '--aprox_iter', type=int)
	ap.add_argument('-aim', '--aprox_mse', type=float)
	ap.add_argument('-ne', '--num_experts', type=int)
	ap.add_argument('-mb', '--mini_batch', type=int)

	main(ap.parse_arge())
