import torch
import torch.nn as nn
import torchvision as tv
import numpy as np
import argparse
from model import Expert, Discriminator
import argparse
import os
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import random
import time
from score_tracker import ScoreTracker
from transform import DataTransform

def main(args):
	dataset = args.dataset if args.dataset else 0
	transforms = args.transforms if args.transforms else 0
	maxiter = args.iter if args.iter else 2000
	aii_iter = args.aprox_iter if args.aprox_iter else 500
	aii_mse = args.aprox_mse if args.aprox_mse else 0.002
	num_exp = args.num_experts if args.num_experts else 10
	mb = args.mini_batch if args.mini_batch else 32
	runs = args.runs if args.runs else 1
	track_time = args.time if args.time else 0
	use_gpu = args.gpu if args.gpu else None

	if use_gpu:
		torch.cuda.set_device(0)

	print('Loading data')
	imgs, lbls = load_data(dataset, use_gpu)
	data_loader = DataTransform(transforms, imgs, lbls, use_gpu)
	print('Beginning run')
	for r in range(runs):
		if track_time == 1:
			start_time = time.time()
		experts, disc = get_models(num_exp)
		if use_gpu:
			disc.cuda()
			for e in experts:
				e.cuda()
		ex_opts, disc_opt = get_optimizers(experts, disc)
		print('Running AII')
		experts, ex_opts = approx_id_init(experts, ex_opts, data_loader , aii_iter, aii_mse)
		print('Training')
		experts, disc, metrics = train(experts, ex_opts, disc, disc_opt, data_loader, transforms, maxiter, mb)
		if track_time == 1:
			end_time = time.time()
			print(f'Run #{r+1} finished after {end_time-start_time} seconds')
		print('Testing models')

		ds = 'MNIST' if dataset == 0 else 'omniglot'
		tf_type = 'paper' if transforms == 0 else 'expanded'
		run_dir = f'./experiments/{ds}/{tf_type}/run_{r}'
		os.makedirs(run_dir)
		metrics.plot_scores(maxiter, run_dir)
		test(experts, disc, data_loader, mb, run_dir)

def load_data(dataset, transforms, use_gpu=0):
	imgs = []
	lbls = []
	ds = "MNIST" if dataset == 0 else "omniglot"
	d_dir = f'./datasets/{ds}/images'
	for fname in os.listdir(d_dir):
		lbl = fname.split('_')[0]
		img = tv.transforms.ToTensor()(Image.open(d_dir + '/' + fname))
		if use_gpu:
			img.cuda()
		imgs.append(img)
		lbls.append(lbl)

	return imgs, lbls

def get_models(num_exp):
	experts = []
	for i in range(num_exp):
		experts.append(Expert())
	disc = Discriminator()

	return experts, disc

def get_optimizers(experts, discrim):
	ex_opts = []

	for exp in experts:
		ex_opts.append(torch.optim.Adam(exp.parameters()))

	disc_opt = torch.optim.Adam(discrim.parameters())

	return ex_opts, disc_opt

def save_tensor_img(img, fname):
	img = tv.transforms.ToPILImage()(img.squeeze(0))
	img.save(fname)

def approx_id_init(experts, opts, data_loader, maxiter, err_th):
	init_experts = []
	init_opts = []

	loss = nn.MSELoss()
	for i in range(maxiter):
		sample, _  = data_loader.get_transformed_sample(1)
		sample = sample[0].unsqueeze(0).unsqueeze(0)
		indexes = []
		for j in range(len(experts)):
			opts[j].zero_grad()
			out = experts[j](sample)

			err = loss(sample, out)
			if err < err_th:
				init_experts.append(experts[j])
				init_opts.append(opts[j])
				indexes.append(j)
				# save_tensor_img(sample, f'expert_{i}_in.png')
				# save_tensor_img(out, f'expert_{i}_out.png')
			else:
				err.backward()
				opts[j].step()

		if len(indexes) > 0:
			indexes.reverse()
			for j in indexes:
				del experts[j]
				del opts[j]
		if len(experts) == 0:
			break

	for i, e in enumerate(experts):
		init_experts.append(e)
		init_opts.append(opts[i])

	for opt in init_opts:
		opt.zero_grad()
	return init_experts, init_opts 

def disc_loss(x, c):
	l = torch.log(x).sum() 
	l = l + torch.mean(torch.log(1-c), dim=1).sum()
	return -1*l

def expert_loss(c):
	return -1*torch.log(c)

def get_mb_sample(data, mb):
	samples = []
	for s in random.sample(data, mb):
		samples.append(list(s))
	return samples

def get_metric_tracker(tfs):
	if tfs == 0:
		transforms = ['UP', 'UPRIGHT', 'RIGHT', 'DOWNRIGHT', 'DOWN', 'DOWNLEFT', 'LEFT', 'UPLEFT', 'NOISE', 'INVERT']
	else:
		transforms = ['R45', 'R90', 'R135', 'R180', 'R225', 'R270', 'R315', 'XSTR', 'YSTR', 'XCOM', 'YCOM']

	return ScoreTracker(transforms)

def train(experts, ex_opts, disc, disc_opt, data_loader, tfs, maxiter, mb):
	metrics = get_metric_tracker(tfs)
	
	for i in range(maxiter):
		t_sample, c_sample = data_loader.get_transformed_sample(mb)
		t_inputs = torch.stack([r.unsqueeze(0) for r in t_sample])
		c_inputs = torch.stack([r for r in c_sample])

		disc_opt.zero_grad()
		for j in range(len(ex_opts)):
			ex_opts[j].zero_grad()

		t_outs = [ex(t_inputs) for ex in experts]


		clean_scores = disc(c_inputs, mb)

		ex_scores = []
		for out in t_outs:
			ex_scores.append(disc(out.detach(), mb))
		ex_scores = torch.cat([s for s in ex_scores], 1)


		score_copy = ex_scores.clone().detach()

		d_loss = disc_loss(clean_scores, ex_scores)
		d_loss.sum().backward(retain_graph=True)
		max_inds = torch.argmax(score_copy, dim=1)
		for j in range(mb):
			e_loss = expert_loss(ex_scores[j][max_inds[j].item()])
			e_loss.backward(retain_graph=True)
			ex_opts[max_inds[j].item()].step()

		disc_opt.step()

		m_in, m_tfs = data_loader.sample_each_transform()
		for j, m in enumerate(m_in):
			for k in range(len(experts)):
				m_out = experts[k](m.unsqueeze(0))
				m_score = disc(m_out, 1)
				metrics.update_score(k, m_score.item(), m_tfs[j])

		if (i+1) % 100 == 0:
			print(f'Iteration {i+1}/{maxiter}')

	return experts, disc, metrics


def test(experts, disc, data_loader, mb, run_dir):
	t_sample, _ = data_loader.get_transformed_sample(mb)
	t_inputs = torch.stack([r.unsqueeze(0) for r in t_sample])

	for i, t in enumerate(t_inputs):
		ex_scores = []
		t_outs = [ex(t.unsqueeze(0)) for ex in experts]
		for out in t_outs:
			ex_scores.append(disc(out, mb))
		ex_scores = torch.cat([s for s in ex_scores], 1)
		max_ind = torch.argmax(ex_scores, dim=1)
		save_tensor_img(t, f'{run_dir}/{i}_in.png')
		save_tensor_img(t_outs[max_ind], f'{run_dir}/{i}_best.png')



if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-ds', '--dataset', type=int)
	ap.add_argument('-tf', '--transforms', type=int)
	ap.add_argument('-i', '--iter', type=int)
	ap.add_argument('-aii', '--aprox_iter', type=int)
	ap.add_argument('-aim', '--aprox_mse', type=float)
	ap.add_argument('-ne', '--num_experts', type=int)
	ap.add_argument('-mb', '--mini_batch', type=int)
	ap.add_argument('-r', '--runs', type=int)
	ap.add_argument('-t', '--time', type=int)
	ap.add_argument('-g', '--gpu', type=int)

	main(ap.parse_args())
