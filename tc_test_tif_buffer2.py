import pickle
from pylab import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(2020)
from torch import optim
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
import matplotlib as mpl
 
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
from torch.utils.data import Dataset
from torchvision import transforms
# from test_buffer_net import UNet
device = torch.device('cuda:0')
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import os
import segmentation_models_pytorch as smp

def spiltTrain_test1(imgs, labels,buffers):
    '''
    划分训练集和验证集,后面可以改随机划分
    train_img_list, train_label_list, train_buffer_list,test_img_list, test_label_list,test_buffer_list
    '''
    train_img_list = imgs[0:int(0.8*len(imgs))]
    train_label_list = labels[0:int(0.8*len(labels))]
    train_buffer_list = buffers[0:int(0.8*len(buffers))]

    test_img_list = imgs[int(0.8*len(imgs)):]
    test_label_list = labels[int(0.8*len(labels)):]
    test_buffer_list = buffers[int(0.8*len(buffers)):]

    return train_img_list, train_label_list, train_buffer_list,test_img_list, test_label_list,test_buffer_list

class MyDataset1(Dataset):
    def __init__(self, imgs, labels,buffers, train: False,isnorm='normon'):
        self.imgs = imgs
        self.labels = labels
        self.buffers = buffers
        self.train = train
        self.isnorm = isnorm

        #写一个数据增强的方法
        if self.train:
            self.trans = transforms.Compose([transforms.RandomHorizontalFlip(
                p=0.5),#依据概率p对PIL图片进行水平翻转，p默认0.5
                transforms.RandomAffine(degrees=(-90, 90), scale=(0.5, 0.5))])#随机仿射变化。degrees选择的度数范围。scale:比例因子区间，从范围a<=比例<=b中随机采样比例。

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        if self.isnorm == 'minmax':
            img = torch.from_numpy(self.imgs[idx]).float()#由from_numpy创建的Tensor
            img = (img - img.min())/(img.max()-img.min())#归一化
        if self.isnorm == 'meanstd':
            img = torch.from_numpy(self.imgs[idx]).float()#由from_numpy创建的Tensor
            img = (img-img.mean()) / img.std()#标准化
        if self.isnorm == 'mm':
            img = torch.from_numpy(self.imgs[idx]).float()#由from_numpy创建的Tensor
            img = (img - img.min())/(img.max()-img.min())#归一化# 先归一化再标准化
            img = (img-img.mean()) / img.std()#标准化
        if self.isnorm == 'normon':
            img = torch.from_numpy(self.imgs[idx]).float()
        # img = torch.from_numpy(self.imgs[idx]).float()
        label = torch.from_numpy(self.labels[idx]).unsqueeze(0).float()
        buffer = torch.from_numpy(self.buffers[idx]).unsqueeze(0).float()

        #数据增强（数据量多时可以不用）
        if self.train:
            img = self.trans(img)
            label = self.trans(label)
            buffer = self.trans(buffer)

        return img.cuda(), label.cuda(),buffer.cuda()

def metric_bi(pred: np.ndarray, label: np.ndarray):
    '''
    Metric for binary classification only.
    Input arrays can have only 0 and 1.
    return (ga, recall_n, precision_n, recall_p, precision_p, f1_p, f1_n)
    '''
    pp = np.sum(pred)  # predicted positive
    pn = label.size - pp  # predicted negative

    t_idx = pred == label  # True
    tp = np.sum(np.logical_and(t_idx, label == 1))  # TP
    tn = np.sum(np.logical_and(t_idx, label == 0))  # TN
    ga = np.sum(t_idx) / label.size  # global accuracy
    recall_n = tn / np.sum(label == 0)
    precision_n = tn / pn
    recall_p = tp / np.sum(label == 1)
    precision_p = tp / pp
    f1_p = 2 * ((precision_p * recall_p)/(precision_p + recall_p))
    f1_n = 2 * ((precision_n * recall_n)/(precision_n + recall_n))
    dice=2*(tp/(pp+np.sum(label == 1)))
    fp= pp - tp
    iou=tp/(fp+np.sum(label == 1))
    iou_n=tn/(fp+pn)
    miou = (iou+iou_n)/2
    return ga, recall_n, precision_n, recall_p, precision_p, f1_p, f1_n,dice,iou,miou

def compute_miou(pred, target):
    mini = 1

    # 计算公共区域
    intersection = pred * (pred == target)

    # 直方图
    area_inter, _ = np.histogram(intersection, bins=2, range=(mini, 2))
    area_pred, _ = np.histogram(pred, bins=2, range=(mini, 2))
    area_target, _ = np.histogram(target, bins=2, range=(mini, 2))
    area_union = area_pred + area_target - area_inter

    # 交集已经小于并集
    assert (area_inter <= area_union).all(
    ), "Intersection area should be smaller than Union area"

    rate = round(max(area_inter) / max(area_union), 4)
    return rate

def creat_pred(predictions, labels):
    # predictions = torch.from_numpy(predictions)
    # labels = torch.from_numpy(labels)

    pred_binary = torch.where(predictions == 1, torch.tensor(1), torch.tensor(0))
    # print('pred_binary',pred_binary)
    # print('pred_binary',pred_binary.shape)
    # p=np.where(pred_binary == 0)
    # print(p)  # 输出
    # pp=np.where(pred_binary == 1)
    # print(pp)  # 输出

    label_binary = torch.where(labels == 1, torch.tensor(1), torch.tensor(0))

    # print('label_binary',label_binary)
    # p=np.where(label_binary == 0)
    # print(p)  # 输出
    # pp=np.where(label_binary == 1)
    # print(pp)  # 输出
    
    # 创建距离为1个像素的缓冲区
    buffer = torch.nn.functional.max_pool2d(label_binary.float(), kernel_size=3, stride=1, padding=1)

    # print('buffer',buffer)
    # print('buffer',buffer.shape)
    # p=np.where(buffer == 0)
    # print(p)  # 输出
    # pp=np.where(buffer == 1)
    # print(pp)  # 输出

    correct = torch.logical_and(pred_binary, buffer)

    pred_array = correct

    return pred_array



def calc_coast_cross_entropy_loss(prediction, target, edge, edge_weight=16, pos_weight=torch.Tensor([1]).cuda()):
    
    bce = F.binary_cross_entropy_with_logits(prediction, target, pos_weight=pos_weight)
    # prediction, target = prediction.cpu(),target.cpu()
    if len(prediction[edge==1]) == 0:
        edge_bce = 0
    else:
        edge_bce = F.binary_cross_entropy_with_logits(prediction[edge==1], target[edge==1])

    loss = bce  + edge_weight * edge_bce

    return loss
def dice_loss(prediction, target):
    smooth = 1.0
    i_flat = prediction.view(-1)
    t_flat = target.view(-1)
    intersection = (i_flat * t_flat).sum()
    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))
def bceweight_dice_loss(prediction, target, bce_weight=0.5):#√
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss
class dice_bce_loss3(torch.nn.Module):#

    def __init__(self, postweight):
        super().__init__()
        self.postweight = postweight
 
    def forward(self, prediction, target):
        postweight=self.postweight
        dice = bceweight_dice_loss(prediction, target)
        bce= nn.BCEWithLogitsLoss(postweight)
        bceloss=bce(prediction, target)
        #不加log  
        dice_ce_loss =  dice + bceloss

        return dice_ce_loss
    
class CombinedLoss(torch.nn.Module):
    def __init__(self, dice_bce_weight, coast_cross_entropy_weight, postweight, edge_weight=2):
        super().__init__()
        self.dice_bce_loss = dice_bce_loss3(postweight)
        self.coast_cross_entropy_loss = calc_coast_cross_entropy_loss
        self.dice_bce_weight = dice_bce_weight
        self.coast_cross_entropy_weight = coast_cross_entropy_weight
        self.edge_weight = edge_weight

    def forward(self, prediction, target, edge):
        dice_bce = self.dice_bce_loss(prediction, target)
        coast_cross_entropy = self.coast_cross_entropy_loss(prediction, target, edge, edge_weight=self.edge_weight)
        return dice_bce*self.dice_bce_weight + coast_cross_entropy*self.coast_cross_entropy_weight
    
class CombinedLoss2(torch.nn.Module):
    def __init__(self, bce_weight, dice_weight, pos_weight, edge_weight=16):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight
        self.pos_weight = pos_weight

    def forward(self, prediction, target, edge):
        bce_dice = bceweight_dice_loss(prediction, target, self.bce_weight)
        coast_cross_entropy = calc_coast_cross_entropy_loss(prediction, target, edge, edge_weight=self.edge_weight, pos_weight=self.pos_weight)
        return bce_dice * self.dice_weight + coast_cross_entropy * (1 - self.dice_weight)

# #早停策略
# class EarlyStopping:
#     def __init__(self, patience=10, verbose=False, delta=0, save_path='my_directory'):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.save_path = save_path  

#     def __call__(self, val_loss, model):
#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_model(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_model(val_loss, model)
#             self.counter = 0

#     def save_model(self, val_loss, model):
#         full_path = os.path.join(self.save_path, 'mymodel.pkl')  # 在这里拼接路径和模型名称
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to {full_path} ...')
#         torch.save(model, full_path)
#         self.val_loss_min = val_loss


savepath1 = r'E:\LCQ\waterline2\20240603\result2\result'
savepath2 = r'ik_20240624_tif_buffer5m_128_zengjia49_FPNunet(DiceBREloss)_bitchsize32_Adam(lr0.0001)_CombinedLoss(a0.9_post0.1)_epoch200_rand_non_oldm'
savepath =  os.path.join(savepath1,savepath2)
print('savepath',savepath)

LOAD_MODEL = False #是否从断点开始，true:从断点开始，false:从epoch1开始
save_path = r'E:\LCQ\waterline2\20240603\result2\model\FPNunet(DiceBREloss)_bitchsize32_Adam(lr0.0001)_CombinedLoss(a0.9_post0.1)_epoch200_modelstate.pth'   # 断点保存的模型路径

from tensorboardX import SummaryWriter
writer = SummaryWriter(r'E:\LCQ\waterline2\20240603\result2\logs\FPNunet(DiceBREloss0.9)_200') 

# #-----------------------------pickle.load 读取保存的对象---------------------------#
# # 数据集
# with open(r'E:\LCQ\waterline\pca\test240504\pickle\20240504_wujiandu_newlinelabel_tif_buffer5m_multctwo_128-128_img_wubeijing_quchukongzhi_zengjia49_15J_1.pickle', 'rb') as f:
#     images1=pickle.load(f)
# with open(r'E:\LCQ\waterline\pca\test240504\pickle\20240504_wujiandu_newlinelabel_tif_buffer5m_multctwo_128-128_label_wubeijing_quchukongzhi_zengjia49_15J_1.pickle', 'rb') as f:
#     labels1=pickle.load(f)
# with open(r'E:\LCQ\waterline\pca\test240504\pickle\20240504_wujiandu_newlinelabel_tif_buffer5m_multctwo_128-128_buffer_wubeijing_quchukongzhi_zengjia49_15J_1.pickle', 'rb') as f:
#     buffers1=pickle.load(f)
# print('img_len', len(images1), len(labels1),len(buffers1))
# # 数据集
# with open(r'E:\LCQ\waterline\pca\test240504\pickle\20240504_wujiandu_newlinelabel_tif_buffer5m_multctwo_128-128_img_wubeijing_quchukongzhi_zengjia49_20J_1.pickle', 'rb') as f:
#     images2=pickle.load(f)
# with open(r'E:\LCQ\waterline\pca\test240504\pickle\20240504_wujiandu_newlinelabel_tif_buffer5m_multctwo_128-128_label_wubeijing_quchukongzhi_zengjia49_20J_1.pickle', 'rb') as f:
#     labels2=pickle.load(f)
# with open(r'E:\LCQ\waterline\pca\test240504\pickle\20240504_wujiandu_newlinelabel_tif_buffer5m_multctwo_128-128_buffer_wubeijing_quchukongzhi_zengjia49_20J_1.pickle', 'rb') as f:
#     buffers2=pickle.load(f)
# print('img_len', len(images2), len(labels2),len(buffers2))

# images=images1+images2
# labels=labels1+labels2
# buffers=buffers1+buffers2
# print('images_len', len(images), len(labels), len(buffers))

# # 数据集
with open(r'E:\LCQ\waterline2\20240524\pickle\20240524_tif_buffer5m_128_tvimg_zengjia49_35J_1.pickle', 'rb') as f:
    images=pickle.load(f)
with open(r'E:\LCQ\waterline2\20240524\pickle\20240524_tif_buffer5m_128_tvlabel_zengjia49_35J_1.pickle', 'rb') as f:
    labels=pickle.load(f)
with open(r'E:\LCQ\waterline2\20240524\pickle\20240524_tif_buffer5m_128_tvbuffer_zengjia49_35J_1.pickle', 'rb') as f:
    buffers=pickle.load(f)
print('img_len', len(images), len(labels),len(buffers))

# #--------------------------按列表顺序划分训练集与验证集------------------------------#

from sklearn.model_selection import train_test_split
train_img, test_img, train_label, test_label,train_buffer,test_buffer = train_test_split(images, labels, buffers, test_size=0.2, random_state=42)#X_train, X_test, y_train, y_test


# train_img = spiltTrain_test1(images, labels,buffers)[0]  # list,len(11)
# train_label = spiltTrain_test1(images, labels,buffers)[1]  # list,len(11)
# train_buffer = spiltTrain_test1(images, labels,buffers)[2]  # list,len(11)
# # print('train_buffer',np.array(train_buffer).shape)
# print('train_img', len(train_img), len(train_label),len(train_buffer))
# test_img = spiltTrain_test1(images, labels,buffers)[3]
# test_label = spiltTrain_test1(images, labels,buffers)[4]
# test_buffer = spiltTrain_test1(images, labels,buffers)[5]
# # print('test_buffer',np.array(test_buffer).shape)
# print('test_img',  len(test_img), len(test_label),len(test_buffer))
print('ok:allotdata')
# #----------------------------------------------------------------------------------#
#---------------------------------自定义数据集--------------------------------------#
train_set = MyDataset1(train_img, train_label, train_buffer,train=False,isnorm='normon')  # list
# train_loader = DataLoader(train_set, batch_size=32, shuffle=True,drop_last=True)#shuffle:每个epoch是否乱序;drop_last: 当样本数不能被batchsize整除时， 是否舍弃最后一批数据
test_set = MyDataset1(test_img, test_label,test_buffer, train=False,isnorm='normon')
# valid_loader = DataLoader(test_set, batch_size=32, shuffle=True,drop_last=True)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

#----------------------------------------------------------------------------------#
#----------------------------------------计算权重-----------------------------------#
labels = np.asarray(train_label)
pos_weight = np.sum(labels == 0) / np.sum(labels == 1)
pos_weight=pos_weight*0.1
pos_weight = torch.tensor(pos_weight).to(device)
print('pos_wight = ', pos_weight)
#----------------------------------------------------------------------------------#
#-----------------------自定义网络、损失函数、优化器、更新策略、激活函数----------------#
#定义网络
# num_classes=2
# net=UNet(in_channels=4, num_classes=num_classes-1).to(device) # 此处修改网络名称

# net = smp.Unet(encoder_name = "efficientnet-b0",
#         encoder_depth = 5,
#         encoder_weights = None,
#         decoder_use_batchnorm = True,
#         decoder_channels = (512, 256, 128, 64, 32), #(512, 256, 128, 64, 32),
#         decoder_attention_type = None,
#         in_channels= 4,
#         classes= 1,
#         activation= None,
#         aux_params= None).to(device)
# net=smp.Linknet(encoder_name = "resnet18",
#         encoder_depth = 5,
#         encoder_weights = None,
#         decoder_use_batchnorm = True,
#         in_channels = 4,
#         classes = 1,
#         activation = None,
#         aux_params= None,).to(device)
# net = smp.DeepLabV3(
#             encoder_name = "resnet34",
#             encoder_depth  = 4,
#             encoder_weights = None,
#             decoder_channels = 128,
#             in_channels = 4,
#             classes = 1,
#             activation = None,
#             upsampling = 8,
#             aux_params = None,).to(device)
# net = smp.PSPNet(
#             encoder_name = "resnet34",
#         encoder_weights = None,
#         encoder_depth = 4,
#         psp_out_channels = 512,
#         psp_use_batchnorm = True,
#         psp_dropout = 0,
#         in_channels = 4,
#         classes = 1,
#         activation = None,
#         upsampling  = 16,
#         aux_params  = None,).to(device)
# net = smp.FPN(
#     encoder_name="efficientnet-b0", 
#     encoder_weights = None,
#     in_channels = 4,
#     classes = 1,
#     activation = None,
# ).to(device)

from tc_net import U_Net,ASPP_Net,AttU_Net,CBAM_UNet,ASPP_Attion_UNet
# net=U_Net(in_ch=4).to(device)
# net=ASPP_Net(in_ch=4).to(device)
# net=AttU_Net(img_ch=4).to(device)
# net=CBAM_UNet(img_ch=4).to(device)
# net=ASPP_Attion_UNet(img_ch=4).to(device)

# from cbamunet_improve1 import CBAM_UNet_MultiScaleConv,CBAM_UNet_FPN
# net=CBAM_UNet_MultiScaleConv(img_ch=4).to(device)
# net=CBAM_UNet_FPN(img_ch=4).to(device)

# from mulscale_unet import multiscaleconv_UNet,multiscaleconv_UNet2
# net=multiscaleconv_UNet(in_ch=4).to(device)
# net=multiscaleconv_UNet2(in_ch=4).to(device)

# from cbam_fpn_mulconv import CBAM_UNet_multiscaleconv_FPN2
# net=CBAM_UNet_multiscaleconv_FPN2(img_ch=4).to(device)

# from segnet import SegNet
# net = SegNet(input_channels=4, output_channels=1).to(device)

from fpn_unet import FPN_UNet
net=FPN_UNet(in_ch=4).to(device)

sigmoid = nn.Sigmoid()

# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)#weight_decay防止过拟合
optimizer = optim.Adam(net.parameters(), lr=0.0001)


# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
edge_weight=2
criterion = CombinedLoss(dice_bce_weight=0.9, coast_cross_entropy_weight=0.1,  postweight=pos_weight.cuda(), edge_weight=edge_weight)

#学习率调整策略
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(1e10), eta_min=1e-5, verbose=False) # 参数MAX_STEP
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)#x
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) 
#----------------------------------------------------------------------------------#

#------------------------------------训练模型---------------------------------------#
#定义数组
epoch_list = []

train_loss_list = []
train_accuracy_list = []

train_rec_list = []
train_pre_list = []
train_f1_list = []
train_dice_list = []
train_iou_list = []
train_miou_list = []

train_rec_list2 = []
train_pre_list2 = []
train_f1_list2 = []
train_dice_list2 = []
train_iou_list2 = []
train_miou_list2 = []

eval_loss_list = []
eval_accuracy_list = []

eval_rec_list = []
eval_pre_list = []
eval_f1_list = []
eval_dice_list = []
eval_iou_list = []
eval_miou_list = []
eval_rec_list2 = []
eval_pre_list2 = []
eval_f1_list2 = []
eval_dice_list2 = []
eval_iou_list2 = []
eval_miou_list2 = []

max_rec = 0.0
max_loss =1.0

#循环训练
from datetime import datetime
start_epoch = 0
epochs = 200

#读取断点
if LOAD_MODEL and os.path.isfile(save_path):  
    print("Loading saved model and continue training...")
    checkpoint = torch.load(save_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    start_epoch = 1

 # 记录训练开始的时间
start_time = datetime.now() 

# # 创建一个 EarlyStopping 对象
# early_stopping = EarlyStopping(patience=10, verbose=True, save_path=savepath)


for epoch in range(start_epoch, epochs+1):#range(1, n+1)循环n次
    print('Epoch {}/{}'.format(epoch, epochs))
    print('-' * 10)#打印10个“-”

    epoch_start_time = datetime.now() # 记录这个 epoch 开始的时间

    ipe = 0  # iteration per epoch
    train_loss = 0.0
    train_accuracy = 0.0

    train_recall_1 = 0.0
    train_precision_1 = 0.0
    train_f1_1 = 0.0
    train_dice_1 = 0.0
    train_iou_1 = 0.0
    train_miou = 0.0

    count = 0
    eval_loss = 0.0
    eval_accuracy = 0.0

    eval_recall_1 = 0.0
    eval_precision_1 = 0.0
    eval_f1_1 = 0.0
    eval_dice_1 = 0.0
    eval_iou_1 = 0.0
    eval_miou = 0.0

    #开始训练网络
    net.train()#训练集
    for imgT, labelT, bufferT in train_loader:#一张一张的计算
        ipe += 1

        optimizer.zero_grad()#清除优化器
        outT = net(imgT)#网络输出结果
        # print(out.shape)
        #--------------通过一些措施优化模型参数--------------#
        lossT = criterion(outT,labelT, bufferT)#计算损失值。loss相当于误差
        # lossT = calc_coast_cross_entropy_loss(outT, labelT, bufferT, edge_weight=edge_weight, pos_weight=pos_weight.cuda()) # 重要参数edge_weigh
        
        lossT.backward()#损失值反向传播
        optimizer.step()#优化器参数更新
        train_loss += lossT.item() 
        #---------------------------------------------------#
        outT = sigmoid(outT).round()  # torch 8*1*16*16

        #原来的训练代码
        predT = outT.cpu().detach().numpy()
        labelT = labelT.cpu().detach().numpy()

        #新建指标后的代码
        ##——————————————————————————————————##
        # predT = outT.cpu()
        # labelT = labelT.cpu()
        # predt = creat_pred(predT,labelT)
        # predT = predt.cpu().detach().numpy()
        # labelT = labelT.cpu().detach().numpy()
        ##————————————————————————————————————————##

        #评价指标#把预测值和你的标签值做一个计算：

        metric_score = metric_bi(predT, labelT)
        train_accuracy += metric_score[0]
        train_precision_1 = metric_score[4]
        train_recall_1 = metric_score[3]
        train_f1_1 = metric_score[5]
        train_dice_1=metric_score[7]
        train_iou_1=metric_score[8]
        train_miou=metric_score[9]
        # train_miou = compute_miou(predT, labelT)

        
        train_pre_list.append(train_precision_1)
        train_rec_list.append(train_recall_1)
        train_f1_list.append(train_f1_1)
        train_dice_list.append(train_dice_1)
        train_iou_list.append(train_iou_1)
        train_miou_list.append(train_miou)

    scheduler.step()      
    net.eval()
    # torch.no_grad() #to increase the validation process uses less memory
    with torch.no_grad():
        for imgV, labelV, bufferV in valid_loader:#一张一张的计算
            count += 1
            outV = net(imgV)#网络输出结果
            lossV = criterion(outV, labelV, bufferV)#计算损失值。loss相当于误差
            # lossV = calc_coast_cross_entropy_loss(outV, labelV, bufferV, edge_weight=edge_weight, pos_weight=pos_weight.cuda()) # 重要参数edge_weigh

            #eval不需要loss.backward
            eval_loss += lossV.item()#总的loss

            outV = sigmoid(outV).round()

            predV = outV.cpu().detach().numpy()
            labelV = labelV.cpu().detach().numpy()

                        #新建指标后的代码
            ##——————————————————————————————————##
            # predV = outV.cpu()
            # labelV = labelV.cpu()
            # predv = creat_pred(predV,labelV)
            # predV = predv.cpu().detach().numpy()
            # labelV = labelV.cpu().detach().numpy()
            ##——————————————————————————————————##

            metric_score1 = metric_bi(predV, labelV)
            eval_accuracy += metric_score1[0]
            eval_precision_1 = metric_score1[4]
            eval_recall_1 = metric_score1[3]
            eval_f1_1 = metric_score1[5]
            eval_dice_1 = metric_score1[7]
            eval_iou_1 = metric_score1[8]
            eval_miou = metric_score1[9]
            # eval_miou = compute_miou(predV, labelV)
            
            eval_pre_list.append(eval_precision_1)
            eval_rec_list.append(eval_recall_1)
            eval_f1_list.append(eval_f1_1)
            eval_dice_list.append(eval_dice_1)
            eval_iou_list.append(eval_iou_1)
            eval_miou_list.append(eval_miou)

    #重复轮次
    epoch_list.append(epoch)

    #计算各项指标的累加平均情况（Accumulate average，缩写accave）
    train_loss /= ipe
    train_loss_list.append(train_loss)
    train_accuracy /= ipe
    train_accuracy_list.append(train_accuracy)

    train_rec = np.array(train_rec_list)
    train_rec = np.nanmean(train_rec)
    train_rec_list2.append(train_rec)
    train_pre = np.array(train_pre_list)
    train_pre = np.nanmean(train_pre)
    train_pre_list2.append(train_pre)
    train_f1 = 2*(train_pre*train_rec)/(train_rec+train_pre)
    train_f1_list2.append(train_f1)
    train_dice = np.array(train_dice_list)
    train_dice = np.nanmean(train_dice)
    train_dice_list2.append(train_dice)
    train_iou = np.array(train_iou_list)
    train_iou = np.nanmean(train_iou)
    train_iou_list2.append(train_iou)
    train_miou = np.array(train_miou_list)
    train_miou = np.nanmean(train_miou)
    train_miou_list2.append(train_miou)


    eval_loss /= count
    eval_loss_list.append(eval_loss)
    eval_accuracy /= count
    eval_accuracy_list.append(eval_accuracy)

    eval_rec = np.array(eval_rec_list)
    eval_rec = np.nanmean(eval_rec)
    eval_rec_list2.append(eval_rec)
    eval_pre = np.array(eval_pre_list)
    eval_pre = np.nanmean(eval_pre)
    eval_pre_list2.append(eval_pre)
    eval_f1 = 2*(eval_pre*eval_rec)/(eval_rec+eval_pre)
    eval_f1_list2.append(eval_f1)
    eval_dice = np.array(eval_dice_list)
    eval_dice = np.nanmean(eval_dice)
    eval_dice_list2.append(eval_dice)
    eval_iou = np.array(eval_iou_list)
    eval_iou = np.nanmean(eval_iou)
    eval_iou_list2.append(eval_iou)
    eval_miou = np.array(eval_miou_list)
    eval_miou = np.nanmean(eval_miou)
    eval_miou_list2.append(eval_miou)


    # 在每个epoch结束时记录结束时间并计算训练时间
    epoch_end_time = datetime.now()
    time_per_epoch = epoch_end_time - epoch_start_time
    expected_time = time_per_epoch * (epochs - epoch)

    # scheduler.step()
    msg = '[Epoch {}][traLoss {:.4f}][Acc {:.2%}][Pre1 {:.2%}][Rec1 {:.2%}][f11 {:.2%}][dice1 {:.2%}][iou1 {:.2%}][miou {:.2%}]'
    print(msg.format(epoch, train_loss, train_accuracy, train_pre, train_rec,  train_f1, train_dice, train_iou, train_miou))
    msg = '[Epoch {}][valLoss {:.4f}][Acc {:.2%}][Pre1 {:.2%}][Rec1 {:.2%}][f11 {:.2%}][dice1 {:.2%}][iou1 {:.2%}][miou {:.2%}]'
    print(msg.format(epoch, eval_loss, eval_accuracy, eval_pre, eval_rec,  eval_f1, eval_dice, eval_iou, eval_miou))

    # 打印训练结果和预计结束时间
    print("Epoch {} finished, took {}. 预计结束时间 {}.".format(epoch, time_per_epoch, datetime.now() + expected_time))

    writer.add_scalar('traLoss', train_loss, epoch)
    writer.add_scalar('train_pre', train_pre, epoch)
    writer.add_scalar('train_rec', train_rec, epoch)
    writer.add_scalar('train_f1', train_f1, epoch)
    writer.add_scalar('train_miou', train_miou, epoch)
    writer.add_scalar('eval_loss', eval_loss, epoch)
    writer.add_scalar('eval_pre', eval_pre, epoch)
    writer.add_scalar('eval_rec', eval_rec, epoch)
    writer.add_scalar('eval_f1', eval_f1, epoch)
    writer.add_scalar('eval_miou', eval_miou, epoch)

    # Save the model after each epoch
    torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)

    # 保存在测试集上的最佳模型
    if eval_rec > max_rec:
        max_rec = eval_rec
        torch.save(
            net, savepath + '_betterrec.pkl')
        torch.save(
            net.state_dict(), savepath + '_betterrec_state_dict.pkl')
        print('权重已更新rec')

    # 保存在测试集上的最佳模型
    if eval_loss < max_loss:
        max_loss = eval_loss
        torch.save(
            net, savepath + '_betterloss.pkl')
        torch.save(
            net.state_dict(), savepath + '_betterloss_state_dict.pkl')
        print('权重已更新loss')

    # early_stopping(eval_loss, net)
    # # 判断是否需要提前结束训练
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     break

# 所有训练结束后计算总的训练时间
training_time = datetime.now() - start_time
print('Total training time: {}'.format(training_time))

#保存数据
torch.save(net,  savepath +'.pkl')

plt_data = pd.DataFrame({'epoch_list': epoch_list,
                         't_loss': train_loss_list,
                         't_acc': train_accuracy_list,
                         't_pre': train_pre_list2,
                         't_rec': train_rec_list2,
                         't_f1': train_f1_list2,
                         't_dice': train_dice_list2,
                         't_iou': train_iou_list2,
                         't_miou': train_miou_list2})
plt_data.to_csv( savepath +'_trainmetric.csv',
                index=None, encoding='utf8')

plt_data = pd.DataFrame({'epoch_list': epoch_list,
                        'v_loss': eval_loss_list,
                         'v_acc': eval_accuracy_list,
                         'v_pre': eval_pre_list2,
                         'v_rec': eval_rec_list2,
                         'v_f1': eval_f1_list2,
                         'v_dice': eval_dice_list2,
                         'v_iou': eval_iou_list2,
                         'v_miou': eval_miou_list2})
plt_data.to_csv( savepath +'_evalmetric.csv',
                index=None, encoding='utf8')

# print('ok:训练完成')
#----------------------------------------------------------------------------------#
plt.figure(figsize=(16, 12))#长、宽
plt.subplot(2, 2, 1)
plt.plot(epoch_list, train_loss_list, 'o-')
plt.legend(('loss'), loc='upper right')
plt.xlabel('epoch次数')
plt.ylabel('loss值')
plt.title('train_loss')

plt.subplot(2, 2, 2)
plt.plot(epoch_list, eval_loss_list, 'x-')
plt.legend(('loss'), loc='upper right')
plt.xlabel('epoch次数')
plt.ylabel('loss值')
plt.title('eval_loss')

plt.subplot(2, 2, 3)
plt.plot(epoch_list, train_accuracy_list,
         epoch_list, train_pre_list2,
         epoch_list, train_rec_list2,
         epoch_list, train_f1_list2,
         epoch_list, train_dice_list2,
         epoch_list, train_iou_list2,
         epoch_list, train_miou_list2)
plt.legend(('accuracy','precision','recall','f1','dice','iou','miou'),prop={'family':'SimHei'}, loc='upper right')
plt.xlabel('epoch次数')
plt.title('train_metric')


plt.subplot(2, 2, 4)
plt.plot(epoch_list, eval_accuracy_list,
         epoch_list, eval_pre_list2,
         epoch_list, eval_rec_list2,
         epoch_list, eval_f1_list2,
         epoch_list, eval_dice_list2,
         epoch_list, eval_iou_list2,
         epoch_list, eval_miou_list2)
plt.legend(('accuracy','precision','recall','f1','dice','iou','miou'),prop={'family':'SimHei'}, loc='upper right')
plt.xlabel('epoch次数')
plt.title('val_metric')

plt.savefig( savepath +'.png')
# plt.show()
print('ok:plt')
