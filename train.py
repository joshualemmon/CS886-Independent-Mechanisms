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
from data_loader import DataLoader
from loss_tracker import LossTracker

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
	# use_gpu = args.gpu if args.gpu else None
	use_gpu = torch.cuda.is_available()
	device = torch.device("cuda" if use_gpu else "cpu")
	# device = torch.device("cpu")

	print('Loading data')
	tf_data, cl_data = load_data(dataset, transforms, device)
	print('Creating data loader')
	data_loader = DataLoader(transforms, tf_data, cl_data, device)
	print('Beginning run')
	for r in range(runs):
		if track_time == 1:
			start_time = time.time()
		experts, disc = get_models(num_exp)
		disc = disc.to(device)
		for e in experts:
			e = e.to(device)
		ex_opts, disc_opt = get_optimizers(experts, disc)

		print('Running AII')
		experts, ex_opts = approx_id_init(experts, ex_opts, data_loader , aii_iter, aii_mse)
		print('Training')
		experts, disc, scores, losses = train(experts, ex_opts, disc, disc_opt, data_loader, transforms, maxiter, mb, device)
		if track_time == 1:
			end_time = time.time()
			print(f'Run #{r+1} finished after {end_time-start_time} seconds')
		print('Testing models')

		ds = 'MNIST' if dataset == 0 else 'omniglot'
		tf_type = 'paper' if transforms == 0 else 'expanded'
		run_dir = f'./experiments/{ds}/{tf_type}/run_{r}'
		os.makedirs(run_dir)
		scores.plot_scores(run_dir)
		losses.plot_losses(run_dir)

		test(experts, disc, data_loader, mb, run_dir)

def load_data(dataset, transforms, device):
	tf_imgs = []
	tf_types = []
	tf_lbls = []

	cl_imgs = []
	cl_lbls = []
	ds = "MNIST" if dataset == 0 else "omniglot"
	tf = "expanded" if transforms == 1 else "paper"
	d_dir = f'./datasets/{ds}/{tf}/'
	for fname in os.listdir(d_dir+'/transformed'):
		lbl = fname.split('_')[0]
		tf = fname.split('_') [1]
		img = tv.transforms.ToTensor()(Image.open(d_dir + '/transformed/' + fname))
		img = img.to(device)
		tf_imgs.append(img)
		tf_types.append(tf)
		tf_lbls.append(lbl)
	for fname in os.listdir(d_dir+'/clean'):
		lbl = fname.split('_')[0]
		img = tv.transforms.ToTensor()(Image.open(d_dir + '/clean/' + fname))
		img = img.to(device)
		cl_imgs.append(img)
		cl_lbls.append(lbl)

	tf_data = list(zip(tf_imgs, tf_types, tf_lbls))
	cl_data = list(zip(cl_imgs, cl_lbls))
	return tf_data, cl_data

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
	img = img.cpu()
	img = tv.transforms.ToPILImage()(img.squeeze(0))
	img.save(fname)

def approx_id_init(experts, opts, data_loader, maxiter, err_th):
	init_experts = []
	init_opts = []

	loss = nn.MSELoss()
	for i in range(maxiter):
		sample, _  = data_loader.get_sample(1)
		sample = sample[0].unsqueeze(0)
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

def train(experts, ex_opts, disc, disc_opt, data_loader, tfs, maxiter, mb, device):
	score_tracker = get_metric_tracker(tfs)
	loss_tracker = LossTracker(len(experts))
	loss = nn.BCELoss(reduction='mean')


	for e in experts:
		for param in e.parameters():
			param.requires_grad = False
	
	for i in range(maxiter):
		t_sample, c_sample = data_loader.get_sample(mb)
		t_inputs = torch.stack([r for r in t_sample])
		c_inputs = torch.stack([r for r in c_sample])

		disc_opt.zero_grad()
		for j in range(len(ex_opts)):
			ex_opts[j].zero_grad()

		t_outs = [ex(t_inputs) for ex in experts]


		clean_scores = disc(c_inputs)

		ex_scores = []
		for out in t_outs:
			ex_scores.append(disc(out.detach()))
		ex_scores = torch.cat([s for s in ex_scores], 1)
		max_inds = torch.argmax(ex_scores, dim=1)

		clean_labels = torch.ones(mb).unsqueeze(1).to(device)
		trans_labels = torch.zeros(mb).unsqueeze(1).to(device)


		d_loss_real = loss(clean_scores, clean_labels)
		d_loss_fake = loss(ex_scores, trans_labels)
		d_loss = d_loss_real + d_loss_fake

		d_l = d_loss.clone().detach()
		loss_tracker.update_disc_loss(d_l.detach().item())
		d_loss.backward(retain_graph=True)

		disc_opt.step()
		for j in range(mb):
			for param in experts[max_inds[j]].parameters():
				param.requires_grad = True
			e_label = torch.ones(1).unsqueeze(1).to(device)
			e_loss = loss(ex_scores[j][max_inds[j]],e_label)
			e_loss.backward(retain_graph=True)
			ex_opts[max_inds[j].item()].step()
			for param in experts[max_inds[j]].parameters():
				param.requires_grad = False

		m_in, m_tfs = data_loader.sample_each_transform()
		for k in range(len(experts)):
			e_loss = 0
			for j, m in enumerate(m_in):
				m_out = experts[k](m)
				m_score = disc(m_out)
				score_tracker.update_score(k, m_score.detach().item(), m_tfs[j])
				e_label = torch.ones(1).unsqueeze(1).to(device)
				e_loss = loss(m_score.detach(), e_label)
				e_loss = e_loss/len(m_in)
			loss_tracker.update_expert_loss(k, e_loss.detach().item())


		if (i+1) % 100 == 0:
			print(f'Iteration {i+1}/{maxiter}')

	return experts, disc, score_tracker, loss_tracker


def test(experts, disc, data_loader, mb, run_dir):
	t_sample, _ = data_loader.sample_each_transform()

	for i, t in enumerate(t_sample):
		ex_scores = []
		t_outs = [ex(t) for ex in experts]
		for out in t_outs:
			ex_scores.append(disc(out))
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