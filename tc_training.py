from skimage import io  # 使用IO库读取tif图片
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import skimage.io
import pandas as pd
from torch.nn.functional import pad
import matplotlib.pyplot as plt
from numpy import result_type
import torch
from torch import optim
from torch.utils.data import DataLoader
from time import time
from scipy.ndimage.filters import median_filter
import pickle
import cv2
import numpy as np
import torch.nn as nn
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 必须在gdal前面
import os
from osgeo import gdal_array as ga
from PIL import Image
from osgeo import gdal


def readTif(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
    '''
    读取tif,返回为data,chw
    '''
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  波段数
    bands = dataset.RasterCount
    #  获取数据
    if (data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    arrXY = []  # 用于存储每个像素的(x,y)坐标
    for i in range(height):
        row = []
        for j in range(width):
            xx = geotrans[0] + i * geotrans[1] + j * geotrans[2]
            yy = geotrans[3] + i * geotrans[4] + j * geotrans[5]
            col = [xx, yy]
            row.append(col)
        arrXY.append(row)

    return width, height, bands, data, geotrans, proj, arrXY


def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    global dataset
    dataset = driver.Create(path, int(im_width), int(
        im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset


def to_tensor(patch):  # patch就是一小块图片，就是补丁
    patch = torch.from_numpy(patch)
    # Variable(torch.unsqueeze(patch_tensor,dim=0).float(),requires_grad = False)
    patch = patch.unsqueeze(dim=0)
    return patch.cuda()


def to_tensor2(patch):  # patch就是一小块图片，就是补丁
    patch = patch.astype(np.float32)
    patch = torch.from_numpy(patch)
    # Variable(torch.unsqueeze(patch_tensor,dim=0).float(),requires_grad = False)
    patch = patch.unsqueeze(dim=0)
    return patch.cuda()


def to_numpy(out):
    out = out[0, 0, :, :].detach().cpu().numpy()
    return out


def Linearstrech(gray_array, truncated_value=2, maxout=255, minout=0):
    '''百分比截断代码，2%的拉伸,如果是5%修改为5即可'''
    truncate_down1 = np.percentile(gray_array, truncated_value)
    truncate_up1 = np.percentile(gray_array, 100-truncated_value)
    gray_array = ((maxout-minout)/(truncate_up1 - truncate_down1))*gray_array
    gray_array[gray_array < minout] = minout
    gray_array[gray_array > maxout] = maxout
    return gray_array


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
    dice = 2*(tp/(pp+np.sum(label == 1)))
    fp = pp - tp
    iou = tp/(fp+np.sum(label == 1))
    iou_n = tn/(fp+pn)
    miou = (iou+iou_n)/2
    return ga, recall_n, precision_n, recall_p, precision_p, f1_p, f1_n, dice, iou, miou


def creat_pred(predictions, labels):
    pred_binary = torch.where(predictions == 1, torch.tensor(1).to(
        predictions.device), torch.tensor(0).to(predictions.device))

    label_binary = torch.where(labels == 1, torch.tensor(1).to(
        labels.device), torch.tensor(0).to(labels.device))

    # 创建距离为1个像素的缓冲区
    buffer = torch.nn.functional.max_pool2d(label_binary.float(
    ), kernel_size=3, stride=1, padding=1).to(predictions.device)

    correct = torch.logical_and(pred_binary, buffer)

    pred_array = correct

    return pred_array


sigmoid = nn.Sigmoid()


# 定义创建缓冲区的函数
def creat_pred(predictions, labels):
    pred_binary = torch.where(
        predictions == 1, torch.tensor(1), torch.tensor(0))

    label_binary = torch.where(labels == 1, torch.tensor(1), torch.tensor(0))

    # 创建距离为1个像素的缓冲区
    buffer = torch.nn.functional.max_pool2d(
        label_binary.float(), kernel_size=3, stride=1, padding=1)

    correct = torch.logical_and(pred_binary, buffer)

    pred_array = correct

    return pred_array


with open(r'E:\LCQ\waterline2\20240524\pickle\20240524_tif_buffer5m_128_testimg_zengjia49_35J_1.pickle', 'rb') as f:
    images = pickle.load(f)
with open(r'E:\LCQ\waterline2\20240524\pickle\20240524_tif_buffer5m_128_testlabel_zengjia49_35J_1.pickle', 'rb') as f:
    labels = pickle.load(f)


# ################################################
img_dir = r'E:\LCQ\waterline2\20240603\result2\pre\0ikonos\ik_img_pickle'
label_dir = r'E:\LCQ\waterline2\20240603\result2\pre\0ikonos\ik_label_pickle'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)  # 如果路径不存在，则创建该路径
if not os.path.exists(label_dir):
    os.makedirs(label_dir)  # 如果路径不存在，则创建该路径


# 遍历图像数组，保存每张图片到指定路径
for i, (img, label) in enumerate(zip(images, labels)):
    # 只取前三个通道，对应RGB
    img_rgb = img[[2, 1, 0], :, :]
    # Normalize for visualization
    img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())
    # Change from (C,H,W) to (H,W,C)
    img_rgb = np.transpose(img_rgb, (1, 2, 0))
    # 保存为png
    plt.imsave(os.path.join(img_dir, f'image_{i}.png'), img_rgb)

    # Normalize label to [0, 1], and scale it to [0, 255]
    label_norm = (label - label.min()) / (label.max() - label.min()) * 255
    # Convert normalized label to uint8
    label_norm = label_norm.astype(np.uint8)
    # Write out as a binary image
    cv2.imwrite(os.path.join(label_dir, f'label_{i}.png'), label_norm)
##########################################################

raise

# 设定文件夹路径
output_dir = r'E:\LCQ\waterline2\20240603\result2\pre\0ikonos\unet2'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # 如果路径不存在，则创建该路径


# 指定你想要的目录
output_dir2 = r'E:\LCQ\waterline2\20240603\result2\pre\0ikonos'

# 使用 os.path.join() 创建文件名和路径
output_file2 = os.path.join(output_dir2, 'unet2_ikonos.xlsx')


# 载入预训练模型
net = torch.load(r'E:\LCQ\waterline2\20240603\result2\result\20240621_tif_buffer5m_128_zengjia49_35J_unet(2)_bitchsize32_Adam(lr0.0001)_CombinedLoss(a0.8_post0.1)_epoch200_rand_non_oldm_betterloss.pkl')
net.eval()  # 将模型转为eval模式

# 评估指标的存储列表
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
miou_list = []


for i, (pre_img, pre_label) in enumerate(zip(images, labels)):

    pre_label[pre_label > 0] = 1

    # gf2
    # patch = to_tensor(pre_img)  # 将图片转换成tensor
    # ikons
    patch = to_tensor2(pre_img)

    patch = patch.float()
    out = net(patch)  # 网络输出结果
    out = sigmoid(out).round()

#############################################
    # out = out.cpu()
    # # out = out.cpu().detach().numpy()
    # # out = out.squeeze()
    # # 加入缓冲区判断步骤
    # out_buff = creat_pred(torch.round(torch.sigmoid(out)), torch.from_numpy(pre_label).unsqueeze(0).unsqueeze(0))
    # result = out_buff.cpu().detach().numpy()
    # result = result.squeeze()
##############################################

    result = out.cpu().detach().numpy()
    # 去除长度为1的轴
    result = result.squeeze()

    # 保存预测结果为PNG
    output_file = os.path.join(output_dir, f'pred_{i}.png')
    skimage.io.imsave(output_file, result)

    # # 计算各项指标并添加到对应的列表中
    # accuracy = np.mean(result.flatten() == pre_label.flatten())
    # accuracy_list.append(accuracy)

    precision = precision_score(pre_label.flatten(), result.flatten())
    # precision_list.append(precision)

    # recall = recall_score(pre_label.flatten(), result.flatten())
    # recall_list.append(recall)

    f1 = f1_score(pre_label.flatten(), result.flatten())
    # f1_list.append(f1)

    # miou = jaccard_score(pre_label.flatten(), result.flatten())
    # miou_list.append(miou)

    metric_score = metric_bi(result.flatten(), pre_label.flatten())
    accuracy = metric_score[0]
    # precision = metric_score[4]
    recall = metric_score[3]
    # f1 = metric_score[5]
    miou = metric_score[9]
    # train_miou = compute_miou(predT, labelT)

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    miou_list.append(miou)

# 输出各项分数的平均值
print('Average Accuracy:', sum(accuracy_list) / len(accuracy_list))
print('Average Precision:', sum(precision_list) / len(precision_list))
print('Average Recall:', sum(recall_list) / len(recall_list))
print('Average F1 Score:', sum(f1_list) / len(f1_list))
print('Average MIoU:', sum(miou_list) / len(miou_list))


# 定义字典，其中包含了我们要保存的数据
data = {
    'Average Accuracy': [sum(accuracy_list) / len(accuracy_list)],
    'Average Precision': [sum(precision_list) / len(precision_list)],
    'Average Recall': [sum(recall_list) / len(recall_list)],
    'Average F1 Score': [sum(f1_list) / len(f1_list)],
    'Average MIoU': [sum(miou_list) / len(miou_list)]
}

# 创建一个 DataFrame
df = pd.DataFrame(data)

# 将 DataFrame 保存为 Excel 文件
df.to_excel(output_file2, index=False)

print('完成测试')
