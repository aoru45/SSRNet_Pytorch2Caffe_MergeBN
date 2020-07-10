import torch
import torch.nn as nn
class MTSSRNet(nn.Module):
    def __init__(self,num_classes=3,stage_num=[3,3,3],lambda_d=1):
        super(MTSSRNet,self).__init__()
        #self.crop = CropAffine()
        #self.crop = AttentionCropLayer()
        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d
        self.x_layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size = 3,stride =1,padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(2)
        )
        self.x_layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size = 3,stride =1,padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(32,32,kernel_size = 3,stride =1,padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(2)
        )
        self.x_layer3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size = 3,stride =1,padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64,64,kernel_size = 3,stride =1,padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(2)
        )
        self.x_layer4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size = 3,stride =1,padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128,128,kernel_size = 3,stride =1,padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128,64,kernel_size = 1,stride =1,padding = 0),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(2)
        )
        self.s_layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size = 3,stride =1,padding = 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2)
        )
        self.s_layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size = 3,stride =1,padding = 1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32,32,kernel_size = 3,stride =1,padding = 1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.MaxPool2d(2)
        )
        self.s_layer2t = nn.Sequential(
            nn.Conv2d(32,64,kernel_size = 1,stride =1,padding = 0),
            nn.Tanh(),
            nn.MaxPool2d(2)
        )
        self.x_layer2t = nn.Sequential(
            nn.Conv2d(32,64,kernel_size = 1,stride =1,padding = 0),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(2)
        )
        self.s_layer3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size = 3,stride =1,padding = 1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64,64,kernel_size = 3,stride =1,padding = 1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(2)
        )
        self.s_layer3t = nn.Sequential(
            nn.Conv2d(64,64,kernel_size = 1,stride =1,padding = 0),
            nn.Tanh(),
            nn.MaxPool2d(2)
        )
        self.x_layer3t = nn.Sequential(
            nn.Conv2d(64,64,kernel_size = 1,stride =1,padding = 0),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(2)
        )
        self.s_layer4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size = 3,stride =1,padding = 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128,128,kernel_size = 3,stride =1,padding = 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128,64,kernel_size = 1,stride =1,padding = 0),
            nn.Tanh(),
            nn.MaxPool2d(2)
        )
        self.feat_delta_s1 = nn.Sequential(
            nn.Linear(64*4*4,2*self.num_classes),
            nn.Tanh()
        )
        self.delta_s1 = nn.Sequential(
            nn.Linear(2*self.num_classes,self.num_classes),
            nn.Tanh()
        )
        self.feat_local_s1 = nn.Sequential(
            nn.Linear(64*4*4,2*self.num_classes),
            nn.Tanh()
        )
        self.local_s1 = nn.Sequential(
            nn.Linear(2*self.num_classes,self.num_classes),
            nn.Tanh()
        )
        self.feat_pred_s1 = nn.Sequential(
            nn.Linear(64*4*4,self.stage_num[0]*self.num_classes),
            nn.ReLU(inplace= True)
        )
        self.feat_delta_s2 = nn.Sequential(
            nn.Linear(64*4*4,2*self.num_classes),
            nn.Tanh()
        )
        self.delta_s2 = nn.Sequential(
            nn.Linear(2*self.num_classes,self.num_classes),
            nn.Tanh()
        )
        self.feat_local_s2 = nn.Sequential(
            nn.Linear(64*4*4,2*self.num_classes),
            nn.Tanh()
        )
        self.local_s2 = nn.Sequential(
            nn.Linear(2*self.num_classes,self.num_classes),
            nn.Tanh()
        )
        self.feat_pred_s2 = nn.Sequential(
            nn.Linear(64*4*4,self.stage_num[0]*self.num_classes),
            nn.ReLU(inplace= True)
        )
        self.feat_delta_s3 = nn.Sequential(
            nn.Linear(64*8*8,2*self.num_classes),
            nn.Tanh()
        )
        self.delta_s3 = nn.Sequential(
            nn.Linear(2*self.num_classes,self.num_classes),
            nn.Tanh()
        )
        self.feat_local_s3 = nn.Sequential(
            nn.Linear(64*8*8,2*self.num_classes),
            nn.Tanh()
        )
        self.local_s3 = nn.Sequential(
            nn.Linear(2*self.num_classes,self.num_classes),
            nn.Tanh()
        )
        self.feat_pred_s3 = nn.Sequential(
            nn.Linear(64*8*8,self.stage_num[2]*self.num_classes),
            nn.ReLU(inplace= True)
        )
    def forward(self,x):
        _,_,img_size,_ = x.size()
        #x,pos = self.crop(x)
        x_layer1 = self.x_layer1(x) # 32
        x_layer2 = self.x_layer2(x_layer1) # 16
        x_layer3 = self.x_layer3(x_layer2) # 8
        x_layer4 = self.x_layer4(x_layer3) # 4
        
        s_layer1 = self.s_layer1(x)
        s_layer2 = self.s_layer2(s_layer1)
        s_layer3 = self.s_layer3(s_layer2)
        s_layer4 = self.s_layer4(s_layer3)

        feat_s1_pre = (x_layer4 * s_layer4).view(-1,64*4*4)
        feat_delta_s1 = self.feat_delta_s1(feat_s1_pre)
        delta_s1 = self.delta_s1(feat_delta_s1)
        feat_local_s1 = self.feat_local_s1(feat_s1_pre)
        local_s1 = self.local_s1(feat_local_s1)
        feat_pred_s1 = self.feat_pred_s1(feat_s1_pre)
        pred_a_s1 = feat_pred_s1.view(-1,self.num_classes,self.stage_num[0])

        s_layer3_ = self.s_layer3t(s_layer3)
        x_layer3_ = self.x_layer3t(x_layer3)
        
        feat_s2_pre = (x_layer3_ * s_layer3_).view(-1,64*4*4)
        feat_delta_s2 = self.feat_delta_s2(feat_s2_pre)
        delta_s2 = self.delta_s2(feat_delta_s2)
        feat_local_s2 = self.feat_local_s2(feat_s2_pre)
        local_s2 = self.local_s2(feat_local_s2)
        feat_pred_s2 = self.feat_pred_s2(feat_s2_pre)
        pred_a_s2 = feat_pred_s2.view(-1,self.num_classes,self.stage_num[1])

        s_layer2_ = self.s_layer2t(s_layer2)
        x_layer2_ = self.x_layer2t(x_layer2)

        feat_s3_pre = (x_layer2_ * s_layer2_).view(-1,64*8*8)
        feat_delta_s3 = self.feat_delta_s3(feat_s3_pre)
        delta_s3 = self.delta_s3(feat_delta_s3)
        feat_local_s3 = self.feat_local_s3(feat_s3_pre)
        local_s3 = self.local_s3(feat_local_s3)
        feat_pred_s3 = self.feat_pred_s3(feat_s3_pre)
        pred_a_s3 = feat_pred_s3.view(-1,self.num_classes,self.stage_num[2])

        a = pred_a_s1[:,0]*0 # (n,3)
        b = pred_a_s1[:,0]*0
        c = pred_a_s1[:,0]*0
        #print(a.size())
        di = self.stage_num[0]//2
        dj = self.stage_num[1]//2
        dk = self.stage_num[2]//2
        
        
        for i in range(0,self.stage_num[0]):
            a = a+(i - di + local_s1)*pred_a_s1[:,:,i] # (n,3)
        
        a = a / (self.stage_num[0] * (1 + self.lambda_d * delta_s1))

        for j in range(0,self.stage_num[1]):
            b = b+(j - dj + local_s2)*pred_a_s2[:,:,j]
        b = b / (self.stage_num[0] * (1 + self.lambda_d * delta_s1)) / (
        self.stage_num[1] * (1 + self.lambda_d * delta_s2))

        for k in range(0,self.stage_num[2]):
            c = c+(k - dk + local_s3)*pred_a_s3[:,:,k]
        c = c / (self.stage_num[0] * (1 + self.lambda_d * delta_s1)) / (
        self.stage_num[1] * (1 + self.lambda_d * delta_s2)) / (
            self.stage_num[2] * (1 + self.lambda_d * delta_s3))

        
        V = 99.
   
        age = (a+b+c)*V
        return age