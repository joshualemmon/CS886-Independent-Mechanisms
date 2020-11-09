import numpy as np
import os
import argparse
import torchvision as tv
import cv2

def main(args):
	ds = args.dataset if args.dataset else 'MNIST'
	dl = args.download if args.download else False
	tf = args.transform if args.transform else 0

	if dl:
		download_dataset(ds)
	else:
		transform_datasets(ds, tf)

def download_dataset(ds):
	if ds == 'MNIST':
		tv.datasets.MNIST(root='./datasets/MNIST/original')
	elif ds == 'omniglot':
		tv.datasets.Omniglot(root='./datasets/omniglot/original')

def transform_datasets(ds, tf):
	images = read_images(ds)
	training_transformed = images[:len(images)]
	training_clean = images[len(images):]

	if tf == 0:
		training_transformed = transform_paper(training_transformed)
	elif tf == 1:
		training_transformed = transform_new(training_transformed)
	save_dataset(training_transformed, training_clean, ds, tf)

def save_dataset(transf, clean, ds, tf):
	for i, img in enumerate(transf):
		cv2.imwrite(f'./datasets/{ds}/{'paper_transformations' if tf == 0 else 'new_transformations'}/transformed/tf_{i}.jpg', img)
	for i, img in enumerate(clean):
		cv2.imwrite(f'./datasets/{ds}/{'paper_transformations' if tf == 0 else 'new_transformations'}/clean/cl_{i}.jpg', img)

def transform_paper(imgs):
	tf_imgs = []
	for i, img in enumerate(images):
		if i%10 == 0:
			tf_imgs.append(translate(img, [1, 0, 0, 0]))
			cv2.imshow(tf_imgs[-1])
			return
		elif i%10 == 1:
			tf_imgs.append(translate(img, [1, 1, 0, 0]))
		elif i%10 == 2:
			tf_imgs.append(translate(img, [0, 1, 0, 0]))
		elif i%10 == 3:
			tf_imgs.append(translate(img, [0, 1, 1, 0]))
		elif i%10 == 4:
			tf_imgs.append(translate(img, [0, 0, 1, 0]))
		elif i%10 == 5:
			tf_imgs.append(translate(img, [0, 0, 1, 1]))
		elif i%10 == 6:
			tf_imgs.append(translate(img, [0, 0, 0, 1]))
		elif i%10 == 7:
			tf_imgs.append(translate(img, [1, 0, 0, 1]))
		elif i%10 == 8:
			tf_imgs.append(add_noise(img))
		elif i%10 == 9:
			tf_imgs.append(invert(img))
	return tf_imgs

def transform_new(imgs):
	tf_imgs - []


def read_images(ds):
	images = []
	for fname in sorted(os.listdir(f'./datasets/{ds}/original')):
		img = cv2.imread(os.path.join(f'./datasets/{ds}/original', fname))
		img /= np.amax(img)
		print(np.amin(img), np.amax(img))
		cv2.imshow(img)
		images.append(img)

	return images

def add_noise(img):
	return img + np.random.rand(0, 0.25, img.shape[:2])

def translate(img, d):
	# d: 1x4 array for directions [UP RIGHT DOWN LEFT]
	T = np.array([1, 0, 4*d[1] - 4*d[3]], [0, 1, 4*d[0] - 4*d[2]])
	return cv2.warpAffine(img, T, img.shape[:2], borderValue=1.0)

def invert(img):
	return np.ones_like(img) - img

def rotate(img, angle):
	pass

def stretch(img, scale):
	pass

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument('-ds', '--dataset', type=str)
	ap.add_argument('-dl', '--download', type=bool)
	ap.add_argument('-tf', '--transform', type=int)

	main(ap.parse_args())