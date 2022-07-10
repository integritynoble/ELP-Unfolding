"""
Take the pretrained strategy, which can change the maximum compressed ratio. scalable is suitable for the first stage denoising prior.
scalable1 is suitable for the non-first stage denoising prior.

Copyright (C) 2021, Chengshuai Yang <integrityyang@gmail>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch
import numpy as np
import torch.nn as nn

class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 4
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames, out_ch*self.interm_ch, \
					  kernel_size=3, padding=1, groups=1, bias=False),
			#nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class CvBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(CvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		return self.convblock(x)

class DownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
			#nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			CvBlock(out_ch, out_ch)
		)

	def forward(self, x):
		return self.convblock(x)

class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch),
			nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2)
		)

	def forward(self, x):
		return self.convblock(x)

def	transfer(in_a,in_b):
		c=[]
		for num1, par1 in enumerate(in_a.parameters()):
			c.append(par1)		
		
		for num1, par1 in enumerate(in_b.parameters()):
			
			data1=c[num1].data
			#vertical replace
			nu=data1.shape[0]
			hor=data1.shape[1]
			nu1=par1.data.shape[0]
			if nu1>nu:
				num=nu1//nu
				remain=nu1%nu
				for real_num in range(num):
					if real_num==0:
						real_data=data1[:,0:hor,:,:]
					else:
						real_data=torch.cat((real_data,data1[:,0:hor,:,:]),0)
				if remain>0:
					real_data1=data1[0:remain,0:hor,:,:]
					real_data=torch.cat((real_data,real_data1),0)
				
			else:
				num=nu//nu1
				remain=nu%nu1
				for real_num in range(num):
					if real_num==0:
						real_data=data1[0:nu1,0:hor,:,:]
					else:
						real_data=real_data+data1[nu1*real_num:nu1*(real_num+1),0:hor,:,:]#torch.cat((real_data,data1),1)
				if remain>0:
					real_data[0:remain,0:hor,:,:]=real_data[0:remain,0:hor,:,:]+data1[nu1*num:nu1*num+remain,0:hor,:,:]

				real_data=data1[0:nu1,0:hor,:,:]							
			
			#horizontal replace
			nu=data1.shape[1]
			nu1=par1.data.shape[1]
			if nu1>nu:							
				num=nu1//nu
				remain=nu1%nu
				for real_num in range(num):
					if real_num==0:
						real_data1=real_data
					else:
						real_data1=torch.cat((real_data1,real_data),1)
				if remain>0:
					real_data2=real_data[:,0:remain,:,:]
					real_data1=torch.cat((real_data1,real_data2),1)
				
			else:
				num=nu//nu1
				remain=nu%nu1
				for real_num in range(num):
					if real_num==0:
						real_data1=real_data[:,0:nu1,:,:]
					else:
						real_data1=real_data1+real_data[:,nu1*real_num:nu1*(real_num+1),:,:]#torch.cat((real_data,data1),1)
				if remain>0:
					real_data1[:,0:remain,:,:]=real_data1[:,0:remain,:,:]+real_data[:,nu1*num:nu1*num+remain,:,:]

				real_data1=real_data[:,0:nu1,:,:]
			
			num_data=real_data1.data.cpu().numpy().astype(np.float32)
			device=real_data1.device
			torch_data=torch.from_numpy(num_data).to(device)
			par1.data=torch_data
		

		return in_b	

class Scalable1(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self,argdict):
		super(Scalable1, self).__init__()
		self.init_channels=argdict['init_channels']
		self.pres_channels=argdict['pres_channels']
		self.init_input=argdict['init_input']
		self.pres_input=argdict['pres_input']
		self.iter__number=argdict['iter__number']	

	def forward(self, color_SCI_backward,argdict):
		C_dim=argdict['pres_channels']
		C_dima=C_dim//2
		C_dimb=C_dima//2
		number=argdict['pres_input']
		
		initial_judge=2
		count=0			

		in_a=color_SCI_backward.AL[count].visualtransformer.inc
		in_b= InputCvBlock(num_in_frames=number+2, out_ch=C_dimb)				
		color_SCI_backward.AL[count].visualtransformer.inc=transfer(in_a,in_b)

		in_a=color_SCI_backward.AL[count].visualtransformer.downc0
		in_b= DownBlock(in_ch=C_dimb*initial_judge, out_ch=C_dima)	
		color_SCI_backward.AL[count].visualtransformer.downc0=transfer(in_a,in_b)

		in_a=color_SCI_backward.AL[count].visualtransformer.downc1
		in_b= DownBlock(in_ch=C_dima*initial_judge, out_ch=C_dim)	
		color_SCI_backward.AL[count].visualtransformer.downc1=transfer(in_a,in_b)

		in_a=color_SCI_backward.AL[count].visualtransformer.upc2
		in_b=UpBlock(in_ch=C_dim*initial_judge, out_ch=C_dima)	
		color_SCI_backward.AL[count].visualtransformer.upc2=transfer(in_a,in_b)

		in_a=color_SCI_backward.AL[count].visualtransformer.upc1
		in_b= UpBlock(in_ch=C_dima*initial_judge, out_ch=C_dimb)	
		color_SCI_backward.AL[count].visualtransformer.upc1=transfer(in_a,in_b)			

		in_a=color_SCI_backward.AL[count].visualtransformer.outc
		in_b=  OutputCvBlock(in_ch=C_dimb*initial_judge, out_ch=number)		
		color_SCI_backward.AL[count].visualtransformer.outc=transfer(in_a,in_b)
				
		return color_SCI_backward

