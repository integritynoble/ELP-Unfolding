import torch
from torch import nn
import torch.nn.init as init
from einops import rearrange
from visualtransformer import Visual_Transformer


class AL_net(nn.Module):
    def __init__(self,args,number1, dropout = 0.1):
        super(AL_net,self).__init__()
       
        in_ch=args['init_input']        
        self.pres_ch=args['pres_input']     
        self.gamma1 = torch.nn.Parameter(torch.Tensor([1]))
        self.gamma2 = torch.nn.Parameter(torch.Tensor([1]))
        
        self.visualtransformer=Visual_Transformer(C_dim=args['init_channels'], in_ch=in_ch,initial_judge=number1)
                     

    def forward(self,v,x, lamda_1,lamda_2,mask,mask_sum,measurement,measurement_mean,iter_,x1=None,x2=None,x3=None,x4=None,x5=None):

        if iter_==0:
            mid_b=(lamda_2-mask*lamda_1)/self.gamma2            
            gamma=self.gamma2/self.gamma1
            x=(v+mid_b)+mask*(torch.div((measurement-torch.sum(mask*(v+mid_b),1,keepdim=True)),(mask_sum+gamma)))
        batch, C, H, W = mask.shape
        noise_map=self.gamma2.expand((batch, 1, H, W))
        img=x-lamda_2/self.gamma2
        x0=torch.cat((noise_map,measurement_mean),1)
        # x0=torch.cat((img,noise_map),1)
        if iter_==0:
            v,x1,x2,x3,x4,x5=self.visualtransformer(iter_,img,x0,self.pres_ch)
        else:
            v,x1,x2,x3,x4,x5=self.visualtransformer(iter_,img,x0,self.pres_ch,x1,x2,x3,x4,x5)
        lamda_1=lamda_1-self.gamma1*(measurement-torch.sum(mask*x,1,keepdim=True))
        lamda_2=lamda_2-self.gamma2*(x-v)
        mid_b=(lamda_2-mask*lamda_1)/self.gamma2            
        gamma=self.gamma2/self.gamma1
        x=(v+mid_b)+mask*(torch.div((measurement-torch.sum(mask*(v+mid_b),1,keepdim=True)),(mask_sum+gamma)))

        return	v,x,x1,x2,x3,x4,x5,lamda_1,lamda_2,self.gamma1,self.gamma2

class SCIbackwardinit(nn.Module):
    def __init__(self,args, dropout = 0.1):
        super().__init__()       
        
        alnet=[]
        self.iter_number=args['iter__number']        
       
        for i in range(self.iter_number):
            if i==0:
                alnet.append(AL_net(args,1))
            else:
                alnet.append(AL_net(args,2))	
           # up_samples.append(Upsample(BasicBlock,3,3*ntemp,4*ntemp))

        self.AL=nn.ModuleList(alnet)

    def forward(self, mask, measurement,img_out_ori):
        #nrow, ncol, ntemp, batch_size,iter_number=args['patchsize'],args['patchsize'],args['temporal_length'],args['batch_size'],args['iter__number']
       #T_or=torch.ones(batch_size,head_num,L_num)	
       #  yb = A(theta+b); 	v = (theta+b)+At((y-yb)./(mask_sum+gamma)) 
       #  yb = A(v+b); 	x = (v+b)+At((y-yb)./(mask_sum+gamma)), b=λ_2−𝐴^𝑇 λ_1, gamma=γ_2/γ_1        
        mask_sum1=torch.sum(mask,1,keepdim=True)
        mask_sum1[mask_sum1==0]=1
        measurement_mean=measurement/mask_sum1
        mask_sum=torch.sum(mask**2,1,keepdim=True)
        mask_sum[mask_sum==0]=1
        x_list,v_list = [],[]
        batch, C, H, W = mask.shape
        lamda_1=torch.zeros_like(measurement)
        lamda_2=torch.zeros_like(mask)
        v=img_out_ori.clone()      
        x=v
       
        for iter_ in range(self.iter_number):
            if iter_==0:
                v,x,x1,x2,x3,x4,x5,lamda_1,lamda_2,gamma1,gamma20=self.AL[iter_](v,x,lamda_1,lamda_2,mask,mask_sum,measurement,measurement_mean,iter_)

            else:
                v,x,x1,x2,x3,x4,x5,lamda_1,lamda_2,gamma1,gamma20=self.AL[iter_](v,x,lamda_1,lamda_2,mask,mask_sum,measurement,measurement_mean,iter_,x1,x2,x3,x4,x5)

            x_list.append(x),v_list.append(v)
        

        #output = v_list[-3:]
        gamma2=gamma20.unsqueeze(0).repeat(batch,1)
        gamma1=gamma1.unsqueeze(0).repeat(batch,1)

        return	x_list,v_list,x1,x2,x3,x4,x5,gamma1,gamma2,lamda_1,lamda_2
