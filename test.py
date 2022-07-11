"""

"""
import time
import argparse
import torch
import wandb
import os
import scipy.io as sio
import numpy as np
import torch.nn as nn
from utils import init_logging, SCI_forward,save_data
from train_common import resume_training,validate_and_log


def main(**args):
	r"""Performs the main training loop
	"""
	torch.backends.cudnn.benchmark = True
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:{}".format(args['GPU']) if use_cuda else "cpu")
	wandb.init(config=args,project="test_a40")#new
	# Load dataset
	
	
	seg_direct = sio.loadmat(args['orig_dir'])
	current_lr=args['lr']
	
	writer, logger = init_logging(args)
	save_data_file=args['log_dir']+'/savedata'	
	if not os.path.exists(save_data_file):
		os.makedirs(save_data_file)
	# Define GPU devices
	
	# Define loss
	criterion = nn.MSELoss()	
	if torch.cuda.device_count() > 1:
		print("have", torch.cuda.device_count(), "GPUs!")

	list=['aerial32','crash32','drop40','kobe32','runner40','traffic48']
	list_seg=['aerial32_256','crash32_256','drop40_256','kobe32_256','runner40_256','traffic48_256']	
	# Resume training or start a new	
	SCI_backward, optimizer, start_epoch = resume_training(args)	
	start_time = time.time()			
	psnr_al=0
	ssim_al=0
	recon=[]
	for number in range(0,6):
		if number==2 or number==4:
			data1=seg_direct[list_seg[number]]
			data=np.expand_dims(data1[:,:,:,0], axis=3)
		else:
			data=seg_direct[list_seg[number]]
		data_name=list[number]											
		#save_data_name=args['log_dir']+'/'+data_name+'.mat'
		batchsize=data.shape[3]
		img_out_ori=torch.ones(batchsize,data.shape[2],data.shape[0],data.shape[1]).to(device)				
		data=torch.from_numpy(data)
		data=data.permute(3,2, 0, 1) #torch.float32
		data=data.float()				
		img_val=data/255.
		SCI_backward.eval()					
		mask,measurement=SCI_forward(img_val,batchsize, args)
		mask,measurement,img_train=mask.to(device),measurement.to(device),img_val.to(device)				
		with torch.no_grad():
			img_out,_=SCI_backward(mask,measurement,img_out_ori)
		name=data_name
		
		#validate_and_log(img_out, img_val, name, writer, logger, epoch,args)					
		psnr_val,ssim_val=validate_and_log(img_out[-1], img_val, name, writer, logger, number,args)				
		psnr_al=psnr_al+psnr_val
		ssim_al=ssim_al+ssim_val
		recon.append(img_out[-1])
	psnr_mean=psnr_al/6
	ssim_mean=ssim_al/6
	logger.info("\t"+"[Last] PSNR_mean: {:1.4f} SSIM_mean: {:1.4f}".format(psnr_mean, ssim_mean))
	wandb.log({'PSNR_mean': psnr_mean, 'SSIM_mean': ssim_mean})
	
	save_data(recon,list,args)	
				
	# Print elapsed time
	elapsed_time = time.time() - start_time
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))		

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train the denoiser")
	parser.add_argument("--batch_size", type=int, default=3, 	\
					 help="Training batch size")#130
	parser.add_argument("--batch_v_size", type=int, default=50, 	\
					 help="Validating batch size")
	parser.add_argument("--temporal_length", type=int, default=8, 	\
					 help="Training temporal_length")
	parser.add_argument("--patchsize", type=int, default=256, 	\
					help="Training batch size")
	parser.add_argument("--iter__number", type=int, default=8, 	\
					help="AL iteration number")##14   3  9 
	
	parser.add_argument("--GPU", type=int, default=0, 	\
					help="the GPU ID")##
	parser.add_argument("--init_channels", type=int, default=64, \
					 help="Number of init_channels")# 72
	parser.add_argument("--pres_channels", type=int, default=64, \
					 help="Number of pres_channels")
	parser.add_argument("--init_input", type=int, default=8, \
					 help="Number of init_input")# 18
	parser.add_argument("--pres_input", type=int, default=8, \
					 help="Number of pres_input") 
	parser.add_argument("--priors", type=int, default=6, \
					 help="Number of priors")  
	parser.add_argument("--epochs", "--e", type=int, default=320, \
					 help="Number of total training epochs")					
	parser.add_argument("--resume_training", "--r", action='store_true',default=True,
						help="resume training from a previous checkpoint")
	parser.add_argument("--lr", type=float, default=2e-5, \
					 help="Initial learning rate")

	parser.add_argument("--save_every_epochs", type=int, default=30,\
						help="Number of training epochs to save state")	# Dirs
	parser.add_argument("--log_dir", type=str, default="/home/chengshuai/data/ELPunfolding/result/main_a40", \
					 help='path of log files')
	parser.add_argument("--code_dir", type=str, default='/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-train/code',						
						help='path of split set file')
	parser.add_argument("--orig_dir", type=str, default='/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-train/origd24n',						
						help='path of orig')

	argspar = parser.parse_args()

	print("\n### Transformer Unfolding model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))
