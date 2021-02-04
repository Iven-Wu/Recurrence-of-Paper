import argparse
import copy
import numpy as np
import os
import random
#from sklearn.utils import shuffle
#import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from load import load_cla_data
#from evaluator import evaluate
from tensorboardX import SummaryWriter

writer = SummaryWriter()

class AWLSTM(nn.Module):
    def __init__(self,grad = None):

        super(AWLSTM,self).__init__()
        units = 32
        self.grad = grad
        self.fea_dim = 11
        self.last_linear = nn.Linear(2*units,1)
        self.epsilon = 0.001

        self.in_lat = nn.Linear(self.fea_dim,10)
        # 这里还有一个lstm，和这个in_lat连着，输出
        #self.lst = nn.LSTM(10,units,5)
        self.lst = nn.LSTM(10, units)
        self.av_W = nn.Parameter(torch.rand((units,units)),requires_grad=True)
        self.av_b = nn.Parameter(torch.rand(units),requires_grad=True)
        self.av_u = nn.Parameter(torch.rand(units),requires_grad=True)

        self.ta = nn.Tanh()

    def forward(self,x,grad=None):
        x = self.in_lat(x)
        #print(x)
        x = x.view(2,1,10)
        output = self.lst(x)
        #print(output)
        #out = torch.tensor([out])
        out,(hn,cn) = output
        self.hn = hn
        #out = hn
        self.a_laten = torch.tanh(torch.matmul(out,self.av_W) + self.av_b)
        self.a_scores = torch.matmul(self.a_laten, self.av_u)


        self.a_alphas = torch.softmax(self.a_scores,dim=0)

        self.a_con = torch.sum(self.a_alphas*out.view(2,-1),axis=0)
        self.fea_con = torch.cat((self.hn.view(-1),self.a_con))
        #print(self.fea_con.shape)
        pred = self.last_linear(self.fea_con)


        ### adv train
        if self.grad:

            # adv_delta = out.grad
            adv_var = self.fea_con + self.epsilon * self.grad
            adv_pred = self.last_linear(adv_var)

            return adv_pred

        return pred



tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt = load_cla_data(
        'data/stocknet-dataset/price/ourpped','2014-01-02', '2015-08-03', '2015-10-01')

train_data = torch.tensor(tra_pv,requires_grad=True,dtype=torch.float32).cuda()
train_gt = torch.tensor(tra_gt).cuda()
test_data = torch.tensor(tes_pv,dtype=torch.float32).cuda()
test_gt = torch.tensor(tes_gt).cuda()
#print(train_data.shape)
#print(train_gt.shape)
model = AWLSTM()
if torch.cuda.is_available():
    model = model.cuda()
m_optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
criterion = nn.MSELoss()

# print("start now")
# print(tra_gt)

epochs = 5
for epoch in range(epochs):
    for i in range(len(train_data)):

        m_optimizer.zero_grad()

        #print(train_gt[i])
        pred = model(train_data[i])
        #print(pred)
        #loss = criterion(pred,train_gt[i])
        loss = (pred-train_gt[i])**2
        loss.backward(retain_graph=True)
        grad = model.fea_con.grad


        adv_pred = model(train_data[i],grad)
        #adv_loss = criterion(train_gt[i],adv_pred)
        adv_loss = (adv_pred-train_gt[i])**2

        all_loss = loss+ 0.5*adv_loss
        all_loss.backward()
        m_optimizer.step()
        if (i+1)%1000==0:
            print("train epoch",epoch," ",i+1," loss",all_loss.item())
        writer.add_scalar('loss_new', all_loss.item(), global_step=i + epoch*len(train_data))


count = 0
ans = []
for i in range(len(test_data)):
    res = model(test_data[i])
    if res>=0.5 and test_gt[i]==1:
        count+=1
    elif res<0.5 and test_gt[i]==0:
        count+=1


    t = [res.item(),tes_gt[i]]
    ans.append(t)
    if (i+1)%100==0:
        print("test sample ",i," accuracy ",count/len(test_data))

print(count/len(test_data))
print(ans)





