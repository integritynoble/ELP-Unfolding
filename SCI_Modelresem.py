import torch
from torch import nn
import torch.nn.init as init
from einops import rearrange
from visualtransformer import Visual_Transformer

class AL_net(nn.Module):
    def __init__(self,args,number1,number, dropout = 0.1):
        super(AL_net,self).__init__()
        
        in_ch=args['init_input']       
        self.pres_ch=args['pres_input']
        self.prior=number
             
        self.gamma1 = torch.nn.Parameter(torch.Tensor([1]))
        self.gamma2 = torch.nn.Parameter(torch.Tensor([1]))
        
        self.visualtransformer=Visual_Transformer(C_dim=args['init_channels'], in_ch=in_ch,initial_judge=number1)
                     

    def forward(self,v0,x, lamda_1,lamda_20,gamma1,gamma20,mask,mask_sum,measurement,measurement_mean,iter_,x1=None,x2=None,x3=None,x4=None,x5=None):
        
        lamda_2=lamda_20[:,:,:,:,-1]
        img=x-lamda_2/self.gamma2
        batch, C, H, W = mask.shape
        noise_map=self.gamma2.expand((batch, 1, H, W))
        x0=torch.cat((noise_map,measurement_mean),1)
        v,x1,x2,x3,x4,x5=self.visualtransformer(iter_,img,x0,self.pres_ch,x1,x2,x3,x4,x5)

        lamda_1=lamda_1-gamma1[0,0]*(measurement-torch.sum(mask*x,1,keepdim=True))
        lamda_2=lamda_2-self.gamma2*(x-v)        
        gamma0=torch.zeros_like(self.gamma2)
        xb=-mask*lamda_1
        for p in range(self.prior-1):
            gamma0+=gamma20[0,:,p]
            xb=xb+lamda_20[:,:,:,:,p]+gamma20[0,:,p]*v0[:,:,:,:,p]
        gamma0+=self.gamma2
        xb=xb+lamda_2+self.gamma2*v
        gamma=gamma0/gamma1[0,0]
        x_b=xb/gamma0
        x=x_b+mask*(torch.div((measurement-torch.sum(mask*x_b,1,keepdim=True)),(mask_sum+gamma)))        

        return	v,x,x1,x2,x3,x4,x5,self.gamma2,lamda_1,lamda_2

class SCIbackwardresem(nn.Module):
    def __init__(self,args, number):
        super().__init__()       
        
        alnet=[]
        self.iter_number=1        
       
        for i in range(self.iter_number):            
            alnet.append(AL_net(args,2,number))	
           # up_samples.append(Upsample(BasicBlock,3,3*ntemp,4*ntemp))

        self.AL=nn.ModuleList(alnet)

    def forward(self, x0,v0,lamda_10,lamda_20,gamma10,gamma20,mask,measurement,x10,x20,x30,x40,x50):
        #nrow, ncol, ntemp, batch_size,iter_number=args['patchsize'],args['patchsize'],args['temporal_length'],args['batch_size'],args['iter__number']
       #T_or=torch.ones(batch_size,head_num,L_num)	
       #  yb = A(theta+b); 	v = (theta+b)+At((y-yb)./(mask_sum+gamma)) 
       #  yb = A(v+b); 	x = (v+b)+At((y-yb)./(mask_sum+gamma)), b=λ_2−𝐴^𝑇 λ_1, gamma=γ_2/γ_1        
        mask_sum=torch.sum(mask,1,keepdim=True)
        mask_sum[mask_sum==0]=1
        measurement_mean=measurement/mask_sum
        x_list,v_list = [],[]
        batch, C, H, W = mask.shape
        #lamda_1=torch.zeros_like(measurement)
        v,x=v0.clone(),x0.clone()      
        lamda_1,lamda_2=lamda_10.clone(),lamda_20.clone()      
        gamma1,gamma2=gamma10.clone(),gamma20.clone()      
        x1,x2,x3,x4,x5=x10.clone(),x20.clone(),x30.clone(),x40.clone(),x50.clone()      

        #lamda_2.append(lamda_2[-1])                
        #xv.append(xv[-1])
       
        for iter_ in range(8,8+self.iter_number):
            v_,x,x1,x2,x3,x4,x5,gamma20,lamda_1,lamda_21=self.AL[iter_-8](v,x,lamda_1,lamda_2,gamma1,gamma2,mask,mask_sum,measurement,measurement_mean,iter_,x1,x2,x3,x4,x5)
            x_list.append(x),v_list.append(v_)
            #lamda_2[:,:,:,:,-1]=lamda_21

        gamma2=gamma20.unsqueeze(0).repeat(batch,1)
        

        return	x_list,v_list,x1,x2,x3,x4,x5,gamma1,gamma2,lamda_1,lamda_21