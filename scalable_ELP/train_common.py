"""
Different common functions for training the models.

Copyright (C) 2021, Chengshuai Yang <integrityyang@gmail>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import torch
import torch.optim as optim
from utils import batch_psnr_ssim
from SCI_Modelcollect import SCI_backwardcollect
from scalable import Scalable
from scalable1 import Scalable1

from data_parallel import BalancedDataParallel


def	resume_training(argdict):
	""" Resumes previous training or starts anew
	"""
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	#l1,l2=argdict['patchsize'],argdict['patchsize']
	if_train=True	
	SCI_backward=SCI_backwardcollect(argdict)
	SCI_backward.to(device)	
	#optimizer = optim.Adam(color_SCI_backward.parameters(), lr=argdict['lr'])
	#resumef = os.path.join(argdict['log_dir'], 'ckpt.pth')
	#checkpoint = torch.load(resumef, map_location=device)
	#color_SCI_backward.all[0].load_state_dict(checkpoint['color_SCI_backward_dict'],strict=False)
	scale=Scalable(argdict)			
	scale1=Scalable1(argdict)
	#color_SCI_backward.all[0]=scale(color_SCI_backward.all[0],argdict)
	#color_SCI_backward.append(color_SCI_backwardinit)	
	#for i in range(0,argdict['priors']-1):		
	#	color_SCI_backward.all[i+1]=scale1(color_SCI_backward.all[i+1],color_SCI_backward.all[0],argdict)

	SCI_backward.to(device)

	optimizer = optim.Adam(SCI_backward.parameters(), lr=argdict['lr'])	
	epoch=0
	if argdict['resume_training_pre']:
		resumef = os.path.join(argdict['log_dir'], 'ckptall.pth')
		if os.path.isfile(resumef):
			checkpoint = torch.load(resumef, map_location=device)
			print("> Resuming previous training")			
			SCI_backward.load_state_dict(checkpoint['color_SCI_backward_dict'],strict=False)
			SCI_backward.all[0]=scale(SCI_backward.all[0],argdict)
			#color_SCI_backward.append(color_SCI_backwardinit)	
			for i in range(1,argdict['priors']):		
				SCI_backward.all[i]=scale1(SCI_backward.all[i],argdict)			
			optimizer.load_state_dict(checkpoint['optimizer'])
			for param_group in optimizer.param_groups:
				current_lr=param_group["lr"]
			current_lr=1e-4
			#new_epoch = argdict['epochs']					
			#optimizer = optim.SGD(color_SCI_backward.parameters(), lr=current_lr)
			optimizer = optim.Adam(SCI_backward.parameters(), lr=current_lr)	
			#epoch=checkpoint['epoch']
		else:
			raise Exception("Couldn't resume training with checkpoint {}".\
				   format(resumef))
	if argdict['resume_training']:
		resumef = os.path.join(argdict['log_dir'], 'ckptallS.pth')
		if os.path.isfile(resumef):
			checkpoint = torch.load(resumef, map_location=device)
			print("> Resuming previous training")			
			SCI_backward.load_state_dict(checkpoint['color_SCI_backward_dict'],strict=False)
			#color_SCI_backward.all[0]=scale(color_SCI_backward.all[0],argdict)
			#color_SCI_backward.append(color_SCI_backwardinit)	
			#for i in range(0,argdict['priors']-1):		
			#	color_SCI_backward.all[i+1]=scale1(color_SCI_backward.all[i+1],color_SCI_backward.all[0],argdict)			
			optimizer.load_state_dict(checkpoint['optimizer'])
			for param_group in optimizer.param_groups:
				current_lr=param_group["lr"]
			#current_lr=1e-4
			#new_epoch = argdict['epochs']					
			#optimizer = optim.SGD(color_SCI_backward.parameters(), lr=current_lr)
			optimizer = optim.Adam(SCI_backward.parameters(), lr=current_lr)	
			#epoch=checkpoint['epoch']
		else:
			raise Exception("Couldn't resume training with checkpoint {}".\
				   format(resumef))
	## parallel train
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		#color_SCI_backward=DataParallelModel(color_SCI_backward,device_ids=[0, 1]) #DataParallelModel(model, device_ids=[0, 1, 2])
		#color_SCI_backward=nn.DataParallel(color_SCI_backward,device_ids=[6,7,8])	 
		gpu0_bsz = 3
		acc_grad = 1
		SCI_backward=BalancedDataParallel(gpu0_bsz // acc_grad,SCI_backward,device_ids=[0])#		
			
	return SCI_backward, optimizer, epoch


def	log_train_psnr(img_out,	gt_train, loss,	writer,	logger,name, epoch, current_lr, iter):
	'''Logs trai loss.
	'''
	#Compute pnsr of the whole batch
	#psnr_train = batch_psnr(torch.clamp(result, 0., 1.), imsource, 1.)
	name1='['+name
	if epoch == 0:
		writer.add_image(name+'frame_id {}'.format(0+1)+'_orginal', gt_train[0,0,:,:].unsqueeze(0).repeat(3,1,1), epoch)
	writer.add_image(name+'frame_id {}'.format(0+1)+'_reconstructed', img_out[0,0,:,:].unsqueeze(0).repeat(3,1,1), epoch)
	psnr_train,ssim_train = batch_psnr_ssim(img_out, gt_train, 1.)
	# Log the scalar values
	writer.add_scalar(name+'loss', loss.item(), epoch)
	writer.add_scalar(name+'lr', current_lr, epoch)
	writer.add_scalar(name+'psnr', psnr_train, epoch)
	writer.add_scalar(name+'ssim', ssim_train, epoch)
# 	writer.add_scalar('PSNR on training data', psnr_train, \
# 		  training_params['step'])
	# idx: i in dataloader
	#print("\n"+name+"[epoch {}] loss: {:1.4f} PSNR_train: {:1.4f} PSNR_train: {:1.4f}".\
	#	  format(epoch+1,  loss.item(), psnr_train))
	logger.info("\t"+name1+"[epoch {}] loss: {:1.7f} lr: {:1.9f} PSNR_train: {:1.4f} SSIM_train: {:1.4f}".\
			format(epoch+1,  loss.item(),current_lr, psnr_train, ssim_train))
	

def save_model_checkpoint(color_SCI_backward, optimizer, epoch, argdict):
	"""Stores the model parameters under 'argdict['log_dir'] + '/net.pth'
	Also saves a checkpoint under 'argdict['log_dir'] + '/ckpt.pth'
	"""
	#torch.save(model.state_dict(), os.path.join(argdict['log_dir'], 'net.pth'))
	save_dict = { 		
		'color_SCI_backward_dict': color_SCI_backward.state_dict(), \
		'optimizer' : optimizer.state_dict(), \
		'epoch': epoch+1, \
		}
	torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckptallS.pth'))

	if epoch % argdict['save_every_epochs'] == 0:
		torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckptall_e{}.pth'.format(epoch+1)))
	del save_dict

def validate_and_log(img_out, img_val, name, writer,logger, epoch,args):
	"""Validation step 
	"""
	name1='['+name
	if epoch == 0:
		writer.add_image(name+'frame_id {}'.format(0+1)+'_Orginal', img_val[0,0,:,:].unsqueeze(0).repeat(3,1,1), epoch)
	psnr_val,ssim_val = batch_psnr_ssim(img_out, img_val, 1.)
	#print("\n"+ name+"[epoch %d] PSNR_val: %.4f," % (epoch+1, psnr_val))
	logger.info("\t"+name1+"[epoch {}] PSNR_val: {:1.4f} SSIM_val: {:1.4f}".format(epoch+1, psnr_val, ssim_val))
	writer.add_scalar(name+'psnr', psnr_val, epoch)
	writer.add_scalar(name+'ssim', ssim_val, epoch)
	writer.add_image(name+'frame_id {}'.format(0+1)+'_Reconstructed', img_out[0,0,:,:].unsqueeze(0).repeat(3,1,1), epoch)

	return psnr_val,ssim_val
	