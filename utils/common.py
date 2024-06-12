import numpy as np
import cv2


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class ListAverageMeter(object):
	"""Computes and stores the average and current values of a list"""
	def __init__(self):
		self.len = 10000  # set up the maximum length
		self.reset()

	def reset(self):
		self.val = [0] * self.len
		self.avg = [0] * self.len
		self.sum = [0] * self.len
		self.count = 0

	def set_len(self, n):
		self.len = n
		self.reset()

	def update(self, vals, n=1):
		assert len(vals) == self.len, 'length of vals not equal to self.len'
		self.val = vals
		for i in range(self.len):
			self.sum[i] += self.val[i] * n
		self.count += n
		for i in range(self.len):
			self.avg[i] = self.sum[i] / self.count
			

def read_img(filename):
	img = cv2.imread(filename)
	return img[:, :, ::-1].astype('float32') / 255.0

# crop borders
def crop_borders(Sour_im, tar_im):
	crop_border_h = Sour_im.shape[0]%8
	crop_border_w = Sour_im.shape[1]%8
 	
	if not crop_border_h == 0:
		tar_im = tar_im[:-crop_border_h, :, :]
		Sour_im = Sour_im[:-crop_border_h, :, :]
  
	if not crop_border_w == 0:
		tar_im = tar_im[:, :-crop_border_w, :]
		Sour_im = Sour_im[:, :-crop_border_w, :]

	cropped_sour = Sour_im
	cropped_tar = tar_im
	if not tar_im.shape == Sour_im.shape:
		raise ValueError('The input and target image shape not same!')
	
	return cropped_sour, cropped_tar



def write_img(filename, img):
	img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
	cv2.imwrite(filename, img)


def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).copy()
