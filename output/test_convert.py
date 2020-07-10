#coding=utf-8

import os.path as osp
import sys
import copy
import os 
import cv2
import numpy as np
import caffe
import torch
from ssrnet import MTSSRNet
caffe.set_mode_cpu()
result_net = caffe.Net('ssrnet_merge_bn_connect.prototxt',"ssrnet_merge_bn.caffemodel", caffe.TEST)

#im = cv2.imread('test.jpg')
#im.resize(64, 64, 3)
#im = im.transpose(2, 0, 1)[np.newaxis, :, :, :]
x = np.ones(shape = (1,3,64,64)).astype(np.float32)
data = result_net.blobs['data']
data.reshape(*x.shape)
data.data[...] = x
pred = result_net.forward()

local_s1 = torch.from_numpy(pred["local_s1_1"])
pred_a_s1 = torch.from_numpy(pred["pred_a_s1"])
local_s3 = torch.from_numpy(pred["local_s3_1"])
pred_a_s3 = torch.from_numpy(pred["pred_a_s3"])
delta_s1 = torch.from_numpy(pred["delta_s1_1"])
delta_s2 = torch.from_numpy(pred["delta_s2_1"])
pred_a_s2 = torch.from_numpy(pred["pred_a_s2"])
local_s2 = torch.from_numpy(pred["local_s2_1"])
delta_s3 = torch.from_numpy(pred["delta_s3_1"])


a = pred_a_s1[:,0]*0 # (n,3)
b = pred_a_s1[:,0]*0
c = pred_a_s1[:,0]*0
#print(a.size())
di = 1
dj = 1
dk = 1


for i in range(0,3):
    a = a+(i - di + local_s1)*pred_a_s1[:,:,i] # (n,3)

a = a / (3 * (1 + 1 * delta_s1))

for j in range(0,3):
    b = b+(j - dj + local_s2)*pred_a_s2[:,:,j]
b = b / (3 * (1 + 1 * delta_s1)) / (3 * (1 + 1 * delta_s2))

for k in range(0,3):
    c = c+(k - dk + local_s3)*pred_a_s3[:,:,k]
c = c / (3 * (1 + 1 * delta_s1)) / (3 * (1 + 1 * delta_s2)) / (3 * (1 + 1 * delta_s3))


V = 99.

age = (a+b+c)*V
print("caffe inference:", age)
#print result_net.blobs


model = MTSSRNet()
model.load_state_dict(torch.load("../pred_15.pth", map_location = "cpu"))

model.eval()

x = torch.ones(1,3,64,64, dtype = torch.float)
print("pytorch inference:", model(x))

