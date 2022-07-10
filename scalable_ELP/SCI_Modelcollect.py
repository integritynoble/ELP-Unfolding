"""
Combine the first period and the second period.

Copyright (C) 2021, Chengshuai Yang <integrityyang@gmail>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch
import torch.nn as nn
from SCI_Modelinit import SCIbackwardinit
from SCI_Modelresem import SCIbackwardresem

class SCI_backwardcollect(nn.Module):
    def __init__(self,argdict):
        super().__init__()       
        SCI_backward=[]
        SCI_backwardinit=SCIbackwardinit(argdict)        #
        SCI_backward.append(SCI_backwardinit)
         
        self.prior=argdict['priors']
        for i in range(0,self.prior-1):             
            SCI_backwardresem=SCIbackwardresem(argdict,i+1)          
            SCI_backward.append(SCI_backwardresem)
        self.all=nn.ModuleList(SCI_backward)
        

    def forward(self, mask,measurement,img_out_ori):
        #nrow, ncol, ntemp, batch_size,iter_number=args['patchsize'],args['patchsize'],args['temporal_length'],args['batch_size'],args['iter__number']
        v0,gamma20,lamda_20=[],[],[]
        for prior in range(self.prior):
            if prior==0:
                x_list,v_list,x1,x2,x3,x4,x5,gamma1,gamma2,lamda_1,lamda_2=self.all[prior](mask,measurement,img_out_ori)
            else:
                x_list,v_list,x1,x2,x3,x4,x5,gamma1,gamma2,lamda_1,lamda_2=self.all[prior](x,v1,lamda_1,lamda_21,gamma1,gamma21,mask,measurement,x1,x2,x3,x4,x5)
            v0.append(v_list[-1]),gamma20.append(gamma2),lamda_20.append(lamda_2)
            v1,gamma21,lamda_21=torch.stack(v0, axis=4),torch.stack(gamma20, axis=2),torch.stack(lamda_20, axis=4)
            x=x_list[-1]        

        return	x_list,v_list