"""

"""
import time
import argparse

import torch
from tqdm import  tqdm
import wandb
import os
import scipy.io as sio
import numpy as np
import torch.nn as nn
from dataset import TrainDataset
from utils import init_logging, SCI_forward,save_data
from train_common import resume_training, log_train_psnr, \
					validate_and_log, save_model_checkpoint

from torch.utils.data import DataLoader

def main(**args):
	r"""Performs the main training loop
	"""
	torch.backends.cudnn.benchmark = True
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	wandb.init(config=args,project="main_a40")#new
	# Load dataset
	print('> Loading datasets ...')

	train_set = TrainDataset(train_files_names=args['dataset_split_files'],trainsetdir=args['trainset_dir'],valsetdir=args['valset_dir'],valsetdircha=args['valset_cha_dir'], \
			orderdir=args['order_dir'],patchsize=args['patchsize'],epoach_num=0, gray_mode=True) ###args['patchsize']
	loader_train = DataLoader(train_set,batch_size=args['batch_size'],shuffle=True, pin_memory=True) #,transforms=[transforms.])
	
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
	for param_group in optimizer.param_groups:
		lr_rate=param_group["lr"]
	# Train and Validate	
	psnr=0	
	start_time = time.time()
	for epoch in range(start_epoch, args['epochs']):

		if epoch>10 and epoch%2==0:
			
			order_num=epoch
			train_set = TrainDataset(train_files_names=args['dataset_split_files'],trainsetdir=args['trainset_dir'], valsetdir=args['valset_dir'], valsetdircha=args['valset_cha_dir'],\
			orderdir=args['order_dir'],patchsize=args['patchsize'],epoach_num=order_num, gray_mode=True,present_set=train_set) ###args['patchsize']
			loader_train = DataLoader(train_set,batch_size=args['batch_size'],shuffle=True, pin_memory=True) #,transforms=[transforms.])

		if epoch>1:
			if (epoch-5)%15==0:
				ratio=(epoch-5)//15
				lr_rate=current_lr*(0.9**(ratio))

		for param_group in optimizer.param_groups:
			param_group["lr"] = lr_rate
		#train 
		for iter, data in tqdm(enumerate(loader_train)):			
			
			batch_size=data.shape[0]
			SCI_backward.train()
			img_out_ori=torch.ones(batch_size,args['temporal_length'],args['patchsize'],args['patchsize']).to(device)
			optimizer.zero_grad()				
			img_train=data/255.
			mask,measurement=SCI_forward(img_train,batch_size, args)
			mask,measurement,img_train=mask.to(device),measurement.to(device),img_train.to(device)
			img_out,v_list=SCI_backward(mask,measurement,img_out_ori)
			gt_train =img_train.to(device)				
			loss =torch.sqrt(criterion(img_out[-1], gt_train)) #+ 0.5*torch.sqrt(criterion(img_out[-2], gt_train))+ 0.5*torch.sqrt(criterion(img_out[-3], gt_train))	
			loss.backward()
			optimizer.step()			
				
		if epoch%1==0:			

			name='Train_id {}]_'.format(iter+1)+'[video_id {}]_'.format(1)
			log_train_psnr(img_out[-1],	gt_train, loss,	writer,	logger,name, epoch, lr_rate, iter)			
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
			
			if psnr<psnr_mean:
				psnr=psnr_mean
				save_data(recon,list,args)
				save_model_checkpoint(SCI_backward, optimizer, epoch,args)
				
	# Print elapsed time
	elapsed_time = time.time() - start_time
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))		

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train the denoiser")
	parser.add_argument("--batch_size", type=int, default=3, 	\
					 help="Training batch size")#130	
	parser.add_argument("--temporal_length", type=int, default=8, 	\
					 help="Training temporal_length")
	parser.add_argument("--patchsize", type=int, default=256, 	\
					help="Training batch size")
	parser.add_argument("--iter__number", type=int, default=8, 	\
					help="AL iteration number")##14   3  9 
	  
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
	parser.add_argument("--resume_training", "--r", action='store_true',default=False,
						help="resume training from a previous checkpoint")
	parser.add_argument("--use_first_stage", "--r1", action='store_true',default=False,
						help="resume training from a previous checkpoint")
	parser.add_argument("--lr", type=float, default=1e-4, \
					 help="Initial learning rate")

	parser.add_argument("--save_every_epochs", type=int, default=30,\
						help="Number of training epochs to save state")	# Dirs
	parser.add_argument("--log_dir", type=str, default="/home/chengshuai/data/ELPunfolding/result/main_a40", \
					 help='path of log files')
	parser.add_argument("--trainset_dir", type=str, default='/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-train/JPEGImages/480p/',
					 help='path of trainset')
	parser.add_argument("--valset_dir", type=str, default='/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-test/JPEGImages/480p/',
						 help='path of validation set')
	parser.add_argument("--valset_cha_dir", type=str, default='/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-test-challenge/JPEGImages/480p/',
						 help='path of validation set')
	parser.add_argument("--dataset_split_files", type=str, default='/home/chengshuai/data/ELPunfolding/data/traindata/datasetname/',						
						help='path of split set file')						
	parser.add_argument("--code_dir", type=str, default='/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-train/code',						
						help='path of split set file')
	parser.add_argument("--orig_dir", type=str, default='/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-train/orig3',						
						help='path of orig')
	parser.add_argument("--order_dir", type=str, default='/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-train/order_color',						
						help='path of split set file')

	argspar = parser.parse_args()

	print("\n### Transformer Unfolding model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))
