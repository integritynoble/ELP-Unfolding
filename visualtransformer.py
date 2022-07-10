# -*- coding: utf-8 -*-
"""

"""

import torch
import torch.nn.functional as F
from torch import nn

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

        
class Visual_Transformer(nn.Module):
    def __init__(self, C_dim=64,in_ch=16,initial_judge=2): #dim = 128, num_tokens = 8
        super(Visual_Transformer, self).__init__()                
        
        C_dima=C_dim//2
        C_dimb=C_dima//2       
        self.number=in_ch
       
        self.inc = InputCvBlock(num_in_frames=self.number+2, out_ch=C_dimb)
        self.downc0 = DownBlock(in_ch=C_dimb*initial_judge, out_ch=C_dima)
        self.downc1 = DownBlock(in_ch=C_dima*initial_judge, out_ch=C_dim)
       # self.dence = OutputCvBlock(in_ch=C_dim, out_ch=C_dim)
        self.upc2 = UpBlock(in_ch=C_dim*initial_judge, out_ch=C_dima)
        self.upc1 = UpBlock(in_ch=C_dima*initial_judge, out_ch=C_dimb)
        self.outc = OutputCvBlock(in_ch=C_dimb*initial_judge, out_ch=self.number)       
        #self.transformer3a=Transformer(hw_number//16, attention_dim//4, nrow//4, mlp_dim//4, C_dim, depth,  dropout)
        #self.transformer3b=Transformer(hw_number//16, attention_dim//4, nrow//4, mlp_dim//4, C_dim, depth,  dropout)  
        
    def forward(self, iter_,img, x0,num,last1=None, last2 = None,last3=None, last4 = None,last5 = None): #100,3,32,32
        C=img.shape[1]
        count=num//C
        remain=num%C
        img1=img.repeat(1,count,1,1)
        img2=img[:,0:remain,:,:]
        img3=torch.cat((img1,img2,x0),1)
        x1=self.inc(img3)        
        if iter_==0:
            x2=self.downc0(x1)
            x3=self.downc1(x2)
            x4=self.upc2(x3)
            x5=self.upc1(x4+x2)
            x6=self.outc(x5+x1)
        else:
            x2=self.downc0(torch.cat((x1,last1),1))

            x3=self.downc1(torch.cat((x2,last2),1))
                       
            x4=self.upc2(torch.cat((x3,last3),1))

            x5=self.upc1(torch.cat((x4+x2,last4),1))

            x6=self.outc(torch.cat((x5+x1,last5),1))
            x1,x2,x3,x4,x5=x1+last1,x2+last2,x3+last3,x4+last4,x5+last5

        for i in range(C):
            z=torch.sum(x6[:,i::C,:,:],1,keepdim=False)  #sum(x6[:,0::C,:,:],)
            if i<remain:
                z1=(x6[:,count*C+i,:,:]+z)/(count+1)
            else:
                z1=z/count
            z1=z1.unsqueeze(1)
            if i==0:
                z2=z1
            else:
                z2=torch.cat((z2,z1),1)
        x=z2+img       
        return  x,x1,x2,x3,x4,x5



