"""
Different utilities.

Copyright (C) 2021, Chengshuai Yang <integrityyang@gmail>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import subprocess
import glob
import logging
from random import choices # requires Python >= 3.6
import numpy as np
import wandb
import scipy.io as sio
import cv2
from skimage.transform import resize
import torch
from tensorboardX import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio,structural_similarity

IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif') # Supported image types
import random
import numpy as np
import skimage.color as sc

def get_patch(sample, patch_size=96, video_length=20, scale=1, multi=False, input_large=False):
    temporal_length,ih, iw = sample.shape   
    number1=int((video_length-temporal_length)/(temporal_length-1))+1
    length=video_length
    if not input_large:
        p = 1 if multi else 1
        tp = p * patch_size
        ip = tp // 1
    else:
        tp = patch_size
        ip = patch_size
    random.seed(0)
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    iz = random.randrange(0, temporal_length)

    if not input_large:
        tx, ty = 1 * ix, 1 * iy
    else:
        tx, ty = ix, iy

    ret = sample[:,iy:iy + ip, ix:ix + ip]
    ret=np.roll(ret, iz, axis=0)
    ret_re=ret[::-1,:,:]
    ret_useful_1=np.zeros([video_length+temporal_length,ip, ip], dtype=np.uint8, order='C')
    ret_useful_1[0:temporal_length,:,:]=ret

    for iter in range(number1):
        if iter%2==0:
            ret_useful_1[(iter+1)*(temporal_length-1)+1:(iter+2)*(temporal_length-1)+1,:,:]=ret_re[1:temporal_length,:,:]
        else:
            ret_useful_1[(iter+1)*(temporal_length-1)+1:(iter+2)*(temporal_length-1)+1,:,:]=ret[1:temporal_length,:,:]
           
    ret_useful=ret_useful_1[0:video_length,:,:]
    return ret_useful

def SCI_forward(img_train, batch_size, args):
	'''Simulate color sci working principle.
	'''
	seg_direct = sio.loadmat(args['code_dir'])
	batch_size=img_train.shape[0]
	code=seg_direct['code']
	nrow, ncol, ntemp=img_train.shape[2],img_train.shape[3],img_train.shape[1]
	mask_train=code[0:nrow,0:ncol,0:ntemp]
	mask_train=torch.from_numpy(mask_train)
	mask_train=mask_train.type(torch.FloatTensor)
	mask=mask_train.permute(2, 0, 1)
	mask1=mask.unsqueeze(0).repeat(batch_size,1,1,1)
	meas=torch.sum(img_train*mask1,1,keepdim=True)
	#mask_train=mask_train.permute(3, 4, 1,2,0)	
	return	mask1, meas

def save_data(recon,list,testnum,args):
	'''save data and images.
	'''
	
	for k in range(6):
		subset=[]
		data=recon[k]
		ver,hor=data.shape[0],data.shape[1]
		for i in range(ver):
			for j in range(hor):			
				subset.append(wandb.Image(
							data[i,j,:,:], caption="Video: {} frame: {}".format(i, j)))
		wandb.log({list[testnum*6+k]:subset})

		save_data_name=args['log_dir']+'/savedata/'+list[testnum*6+k]+'.mat'
		sio.savemat(save_data_name,{'recon':recon[k].data.cpu().numpy().astype(np.float32)})


def init_logging(argdict):
	"""Initilizes the logging and the SummaryWriter modules
	"""
	if not os.path.exists(argdict['log_dir']):
		os.makedirs(argdict['log_dir'])
	writer = SummaryWriter(argdict['log_dir'])
	logger = init_logger(argdict['log_dir'], argdict)
	return writer, logger

def get_imagenames(seq_dir, pattern=None):
	""" Get ordered list of filenames
	"""
	files = []
	for typ in IMAGETYPES:
		files.extend(glob.glob(os.path.join(seq_dir, typ)))

	# filter filenames
	if not pattern is None:
		ffiltered = []
		ffiltered = [f for f in files if pattern in os.path.split(f)[-1]]
		files = ffiltered
		del ffiltered

	# sort filenames alphabetically
	files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	return files


def open_sequence_train(seq_dir, gray_mode,orderdir,count_num,choose_num,patchsize, expand_if_needed=False, max_num_fr=100):
	r""" Opens a sequence of images and expands it to even sizes if necesary
	Args:
		fpath: string, path to image sequence
		gray_mode: boolean, True indicating if images is to be open are in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
		max_num_fr: maximum number of frames to load
	Returns:
		seq: array of dims [num_frames, C, H, W], C=1 grayscale or C=3 RGB, H and W are even.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
	"""
	# Get ordered list of filenames
	seg_direct = sio.loadmat(orderdir)
	order=seg_direct['order']
	files = get_imagenames(seq_dir)
	seq_list = []
	seq_lista = []
	#print("\tOpen sequence in folder: ", seq_dir)
	#np.random.seed(0)
	#start_img_id = np.random.randint(0, len(files) - max_num_fr)
	num_len=len(files)
	compress_ratio=24
	ver=int(order[count_num,0]*(32/244))
	hor=int(order[count_num,1]*(256/598))

	for fpath in files[0:num_len]:
		img,img1, expanded_h, expanded_w = open_image(fpath, \
												 gray_mode=gray_mode, \
												 expand_if_needed=expand_if_needed, \
												 expand_axis0=False,normalize_data=False)
		seq_list.append(img)
		seq_lista.append(img1)
		#seq_list.append(np.rot90(img))
	seq = np.stack(seq_list, axis=0)
	seqa = np.stack(seq_lista, axis=0)
	seq_list1 = []
	for ii in range(0,seq.shape[0]-compress_ratio+1,compress_ratio):  #  seq.shape[0]-compress_ratio
		for jj in range(2):
			ori1=seq[ii:ii+compress_ratio,order[count_num,0]:order[count_num,0]+patchsize,order[count_num,1]:order[count_num,1]+patchsize]
			ori2=seqa[ii:ii+compress_ratio,ver:ver+patchsize,hor:hor+patchsize]			
			ori=ori1
			seq_list1.append(ori)
			ori_ro90=np.rot90(ori1,k=1,axes=(1,2))
			seq_list1.append(ori_ro90)
			ori_ro180=np.rot90(ori_ro90,k=1,axes=(1,2))
			seq_list1.append(ori_ro180)
			ori_ro270=np.rot90(ori_ro180,k=1,axes=(1,2))
			seq_list1.append(ori_ro270)
			mirror=ori[:,::-1,:]
			seq_list1.append(mirror)
			mirror_ro90=np.rot90(mirror,k=1,axes=(1,2))
			seq_list1.append(mirror_ro90)
			mirror_ro180=np.rot90(mirror_ro90,k=1,axes=(1,2))
			seq_list1.append(ori_ro180)
			mirror_ro270=np.rot90(mirror_ro180,k=1,axes=(1,2))
			seq_list1.append(mirror_ro270)
			ori=ori2
			seq_list1.append(ori)
			ori_ro90=np.rot90(ori1,k=1,axes=(1,2))
			seq_list1.append(ori_ro90)
			ori_ro180=np.rot90(ori_ro90,k=1,axes=(1,2))
			seq_list1.append(ori_ro180)
			ori_ro270=np.rot90(ori_ro180,k=1,axes=(1,2))
			seq_list1.append(ori_ro270)
			mirror=ori[:,::-1,:]
			seq_list1.append(mirror)
			mirror_ro90=np.rot90(mirror,k=1,axes=(1,2))
			seq_list1.append(mirror_ro90)
			mirror_ro180=np.rot90(mirror_ro90,k=1,axes=(1,2))
			seq_list1.append(ori_ro180)
			mirror_ro270=np.rot90(mirror_ro180,k=1,axes=(1,2))
			seq_list1.append(mirror_ro270)
			
			count_num =count_num +1
	for ii in range(seq.shape[0]-compress_ratio,-1,-compress_ratio):  #  seq.shape[0]-compress_ratio
		for jj in range(2):
			ori1=seq[ii:ii+compress_ratio,order[count_num,0]:order[count_num,0]+patchsize,order[count_num,1]:order[count_num,1]+patchsize]
			ori2=seqa[ii:ii+compress_ratio,ver:ver+patchsize,hor:hor+patchsize]			
			ori=ori1
			seq_list1.append(ori)
			ori_ro90=np.rot90(ori1,k=1,axes=(1,2))
			seq_list1.append(ori_ro90)
			ori_ro180=np.rot90(ori_ro90,k=1,axes=(1,2))
			seq_list1.append(ori_ro180)
			ori_ro270=np.rot90(ori_ro180,k=1,axes=(1,2))
			seq_list1.append(ori_ro270)
			mirror=ori[:,::-1,:]
			seq_list1.append(mirror)
			mirror_ro90=np.rot90(mirror,k=1,axes=(1,2))
			seq_list1.append(mirror_ro90)
			mirror_ro180=np.rot90(mirror_ro90,k=1,axes=(1,2))
			seq_list1.append(ori_ro180)
			mirror_ro270=np.rot90(mirror_ro180,k=1,axes=(1,2))
			seq_list1.append(mirror_ro270)
			ori=ori2
			seq_list1.append(ori)
			ori_ro90=np.rot90(ori1,k=1,axes=(1,2))
			seq_list1.append(ori_ro90)
			ori_ro180=np.rot90(ori_ro90,k=1,axes=(1,2))
			seq_list1.append(ori_ro180)
			ori_ro270=np.rot90(ori_ro180,k=1,axes=(1,2))
			seq_list1.append(ori_ro270)
			mirror=ori[:,::-1,:]
			seq_list1.append(mirror)
			mirror_ro90=np.rot90(mirror,k=1,axes=(1,2))
			seq_list1.append(mirror_ro90)
			mirror_ro180=np.rot90(mirror_ro90,k=1,axes=(1,2))
			seq_list1.append(ori_ro180)
			mirror_ro270=np.rot90(mirror_ro180,k=1,axes=(1,2))
			seq_list1.append(mirror_ro270)
			
			count_num =count_num +1
	return seq_list1, count_num,expanded_h, expanded_w

def open_sequence(seq_dir, gray_mode, orderdir,count_num,patchsize,expand_if_needed=False, max_num_fr=100):
	r""" Opens a sequence of images and expands it to even sizes if necesary
	Args:
		fpath: string, path to image sequence
		gray_mode: boolean, True indicating if images is to be open are in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
		max_num_fr: maximum number of frames to load
	Returns:
		seq: array of dims [num_frames, C, H, W], C=1 grayscale or C=3 RGB, H and W are even.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
	"""
	# Get ordered list of filenames
	files = get_imagenames(seq_dir)

	seq_list = []
	print("\tOpen sequence in folder: ", seq_dir)
	# np.random.seed(0)
	# start_img_id = np.random.randint(0,len(files)-max_num_fr)

	for fpath in files[0:max_num_fr]:

		img,_, expanded_h, expanded_w = open_image(fpath,\
												   gray_mode=gray_mode,\
												   expand_if_needed=expand_if_needed,\
												   expand_axis0=False)
		seq_list.append(img)
	seq = np.stack(seq_list, axis=0)
	seq_list1 = []
	for ii in range(0,seq.shape[0]-8,8):
		for jj in range(9):
			ori=seq[ii:ii+8,order[count_num,0]:order[count_num,0]+patchsize,order[count_num,1]:order[count_num,1]+patchsize]
			ori_ro90=np.rot90(ori,k=1,axes=(1,2))
			ori_ro180=np.rot90(ori_ro90,k=1,axes=(1,2))
			ori_ro270=np.rot90(ori_ro180,k=1,axes=(1,2))
			mirror=ori[:,::-1,:]
			mirror_ro90=np.rot90(mirror,k=1,axes=(1,2))
			mirror_ro180=np.rot90(mirror_ro90,k=1,axes=(1,2))		
			mirror_ro270=np.rot90(mirror_ro180,k=1,axes=(1,2))	
			seq_list1.append(ori)
			seq_list1.append(ori_ro90)
			seq_list1.append(ori_ro180)
			seq_list1.append(ori_ro270)
			seq_list1.append(mirror)
			seq_list1.append(mirror_ro90)
			seq_list1.append(mirror_ro180)
			seq_list1.append(mirror_ro270)
	return seq_list1, count_num,expanded_h, expanded_w

def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, normalize_data=True):
	r""" Opens an image and expands it if necesary
	Args:
		fpath: string, path of image file
		gray_mode: boolean, True indicating if image is to be open
			in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
	Returns:
		img: image of dims NxCxHxW, N=1, C=1 grayscale or C=3 RGB, H and W are even.
			if expand_axis0=False, the output will have a shape CxHxW.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
	"""
	if not gray_mode:
		# Open image as a CxHxW torch.Tensor
		img = cv2.imread(fpath)[0:480,0:854,:]
		# from HxWxC to CxHxW, RGB image
		img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
	else:
		# from HxWxC to  CxHxW grayscale image (C=1)
		img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

	if expand_axis0:
		img = np.expand_dims(img, 0)

	# Handle odd sizes
	expanded_h = False
	expanded_w = False
	sh_im = img.shape
	if expand_if_needed:
		if sh_im[-2]%2 == 1:
			expanded_h = True
			if expand_axis0:
				img = np.concatenate((img, \
					img[:, :, -1, :][:, :, np.newaxis, :]), axis=2)
			else:
				img = np.concatenate((img, \
					img[:, -1, :][:, np.newaxis, :]), axis=1)


		if sh_im[-1]%2 == 1:
			expanded_w = True
			if expand_axis0:
				img = np.concatenate((img, \
					img[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
			else:
				img = np.concatenate((img, \
					img[:, :, -1][:, :, np.newaxis]), axis=2)

	if normalize_data:
		img = normalize(img)
	img1=(resize(img[:,:],(288,512))*255).astype(np.uint8)

	return img,img1, expanded_h, expanded_w

def batch_psnr_ssim(img, imclean, data_range):
	r"""
	Computes the PSNR along the batch dimension (not pixel-wise)

	Args:
		img: a `torch.Tensor` containing the restored image
		imclean: a `torch.Tensor` containing the reference image
		data_range: The data range of the input image (distance between
			minimum and maximum possible values). By default, this is estimated
			from the image data-type.
	"""
	img_cpu = img.data.cpu().numpy().astype(np.float32)
	imgclean = imclean.data.cpu().numpy().astype(np.float32)
	psnr = 0
	ssim= 0	
	for i in range(img_cpu.shape[0]):
		for j in range(img_cpu.shape[1]):			
			psnr += peak_signal_noise_ratio(imgclean[i,j, :, :], img_cpu[i,j, :, :], \
						data_range=data_range)
			ssim += structural_similarity(imgclean[i,j, :, :], img_cpu[i,j, :, :], \
						data_range=data_range)
			#ssim+ =structural_similarity(imgclean[i,j, :, :, :], img_cpu[i, :, :, :],\
			#		data_range=data_range,multichannel=True)
		#ssim += peak_signal_noise_ratio(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
		#			   data_range=data_range)
	return psnr/(img_cpu.shape[0]*img_cpu.shape[1]),ssim/(img_cpu.shape[0]*img_cpu.shape[1])


def get_git_revision_short_hash():
	r"""Returns the current Git commit.
	"""
	return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def init_logger(log_dir, argdict):
	r"""Initializes a logging.Logger to save all the running parameters to a
	log file

	Args:
		log_dir: path in which to save log.txt
		argdict: dictionary of parameters to be logged
	"""
	from os.path import join

	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(log_dir, 'log.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	try:
		logger.info("Commit: {}".format(get_git_revision_short_hash()))
	except Exception as e:
		logger.error("Couldn't get commit number: {}".format(e))
	logger.info("Arguments: ")
	for k in argdict.keys():
		logger.info("\t{}: {}".format(k, argdict[k]))

	return logger

def init_logger_test(result_dir):
	r"""Initializes a logging.Logger in order to log the results after testing
	a model

	Args:
		result_dir: path to the folder with the denoising results
	"""
	from os.path import join

	logger = logging.getLogger('testlog')
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(result_dir, 'log.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger

def close_logger(logger):
	'''Closes the logger instance
	'''
	x = list(logger.handlers)
	for i in x:
		logger.removeHandler(i)
		i.flush()
		i.close()





