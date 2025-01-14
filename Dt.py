# -*- coding: utf-8 -*-
# @Time : 2024/6/21 14:33
# @Author : 王梦潇
# @File : Dt.py.py
# Function: 加载数据集
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import scipy.signal
from sklearn.model_selection import KFold
import pandas as pd
from datetime import datetime, timedelta
from torch.utils.data import Dataset
from utils.utils import *
def load_data(data_dir='C:/mancy/experiment/cross_modal_re/data/', task='II_Pleth', batch_size=128, model_name = 'DDPM_Unet',debug=False,resample={'switch':False,'num_samples':1024}):
    # 数据读取
    train_path = data_dir + task+ '/' + task + '_train_data.npy'
    val_path = data_dir + task + '/' + task + '_val_data.npy'
    test_path = data_dir + task + '/' + task + '_test_data.npy'

    X_Train_Ori = np.load(train_path, allow_pickle=True)
    X_Val_Ori = np.load(val_path, allow_pickle=True)
    X_Test_Ori = np.load(test_path, allow_pickle=True)



    # 是否重采样到512
    if resample['switch']:
        num_samples = resample['num_samples']
        X_Train_Ori = np.array([scipy.signal.resample(x, num_samples, axis=-1) for x in X_Train_Ori])
        X_Val_Ori = np.array([scipy.signal.resample(x, num_samples, axis=-1) for x in X_Val_Ori])
        X_Test_Ori = np.array([scipy.signal.resample(x, num_samples, axis=-1) for x in X_Test_Ori])


    if model_name == 'DDPM_Unet':
        X_Train_Data = X_Train_Ori[:, :, :]  # 这个选择待重建的部分
        Y_Train_Data = X_Train_Ori[:, [1], :]  # 这个选择p_sample_loop
        X_Val_Data = X_Val_Ori[:, :, :]
        Y_Val_Data = X_Val_Ori[:, [1], :]
        X_Test_Data = X_Test_Ori[:, :, :]
        Y_Test_Data = X_Test_Ori[:, [1], :]
    # elif model_name == 'DualVAE':
        # X_Train_Data = X_Train_Ori[:, 0, :]
        # Y_Train_Data = X_Train_Ori[:, 1, :]
        # X_Val_Data = X_Val_Ori[:, 0, :]
        # Y_Val_Data = X_Val_Ori[:, 1, :]
        # X_Test_Data = X_Test_Ori[:, 0, :]
        # Y_Test_Data = X_Test_Ori[:, 1, :]
    else:
        # 适用于卷积架构，需要多一个维度
        X_Train_Data = X_Train_Ori[:, [0], :]
        Y_Train_Data = X_Train_Ori[:, [1], :]
        X_Val_Data = X_Val_Ori[:, [0], :]
        Y_Val_Data = X_Val_Ori[:, [1], :]
        X_Test_Data = X_Test_Ori[:, [0], :]
        Y_Test_Data = X_Test_Ori[:, [1], :]

    if debug:
        debug_batch = 1000
        X_Train_Data = X_Train_Data[:debug_batch]
        Y_Train_Data = Y_Train_Data[:debug_batch]
        X_Val_Data = X_Val_Data[:debug_batch]
        Y_Val_Data = Y_Val_Data[:debug_batch]
        X_Test_Data = X_Test_Data[:debug_batch]
        Y_Test_Data = Y_Test_Data[:debug_batch]
    X_Train_Data = torch.FloatTensor(X_Train_Data)
    Y_Train_Data = torch.FloatTensor(Y_Train_Data)
    X_Val_Data = torch.FloatTensor(X_Val_Data)
    Y_Val_Data = torch.FloatTensor(Y_Val_Data)
    X_Test_Data = torch.FloatTensor(X_Test_Data)
    Y_Test_Data = torch.FloatTensor(Y_Test_Data)

    del X_Train_Ori, X_Val_Ori, X_Test_Ori

    train_set = TensorDataset(X_Train_Data, Y_Train_Data)
    val_set = TensorDataset(X_Val_Data, Y_Val_Data)
    test_set = TensorDataset(X_Test_Data, Y_Test_Data)
    print('train_set',len(train_set))
    print('val_set',len(val_set))
    print('test_set',len(test_set))

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size,  num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    X_shape = X_Test_Data.shape
    Y_shape = Y_Test_Data.shape

    return train_loader, val_loader, test_loader, X_shape, Y_shape

def load_data_fold(data_dir='C:/mancy/experiment/cross_modal_re/data/', task='II_Pleth',downtask='heart_failure', batch_size=128,debug=False,resample={'switch':False,'num_samples':1024},target_fold=0,input_modal='all',split_method='inter',train_sample=False,filtered=False):
    '''
    读取包含下游任务标签的数据集
    :param data_dir:
    :param task:
    :param downtask:
    :param batch_size:
    :param debug:
    :param resample:
    :return:
    '''

    merged_idx = np.load(data_dir + task + '/merged_idx.npy')
    merged_data = np.load(data_dir + task + '/merged_data.npy')
    if task == 'II_Pleth_ABP':
        merged_data[:,2,:] = merged_data[:,2,:]/100


    merged_label = np.load(data_dir + task + '/merged_label.npy')

    # 滤波
    if filtered:
        fs = merged_data.shape[-1]/10
        ecg_data = [filter_ecg_channel(x[0, :],fs) for x in merged_data]
        ppg_data = [filter_ppg_channel(x[1, :],fs) for x in merged_data]
        abp_data = [filter_abp_channel(x[2, :],fs) for x in merged_data]

        merged_data = np.stack([ecg_data,ppg_data,abp_data],axis=1)
        del ecg_data,ppg_data,abp_data

    if split_method == 'inter':
        X_Train_Data, y_Train_Data,idx_Tarin_Data, X_Val_Data, y_Val_Data, idx_Val_Data, X_Test_Data, y_Test_Data,test_idx = k_fold_split(merged_idx, merged_data, merged_label, target_fold=target_fold)
    else:
        X_Train_Data, y_Train_Data,idx_Tarin_Data, X_Val_Data, y_Val_Data, idx_Val_Data, X_Test_Data, y_Test_Data,test_idx = random_split(merged_idx, merged_data, merged_label)

    downtask_map= {'heart_failure':0,'hypertensive':1,'diabetes':2,'kidney_failure':3}
    if downtask == 'all':
        y_train = y_Train_Data
        y_val = y_Val_Data
        y_test = y_Test_Data
    else:
        y_train = y_Train_Data[:,[downtask_map[downtask]]]
        y_val =y_Val_Data[:,[downtask_map[downtask]]]
        y_test = y_Test_Data[:,[downtask_map[downtask]]]

    # 把label中所有的nan变为0
    Y_Train_Data = np.nan_to_num(y_train)
    Y_Val_Data = np.nan_to_num(y_val)
    Y_Test_Data = np.nan_to_num(y_test)

    # 是否重采样到512
    if resample['switch']:
        num_samples = resample['num_samples']
        X_Train_Data = np.array([scipy.signal.resample(x, num_samples, axis=-1) for x in X_Train_Data])
        X_Val_Data = np.array([scipy.signal.resample(x, num_samples, axis=-1) for x in X_Val_Data])
        X_Test_Data = np.array([scipy.signal.resample(x, num_samples, axis=-1) for x in X_Test_Data])

    if debug:
        debug_batch = 1000
        X_Train_Data = X_Train_Data[:debug_batch]
        Y_Train_Data = Y_Train_Data[:debug_batch]
        X_Val_Data = X_Val_Data[:debug_batch]
        Y_Val_Data = Y_Val_Data[:debug_batch]
        X_Test_Data = X_Test_Data[:debug_batch]
        Y_Test_Data = Y_Test_Data[:debug_batch]
    X_Train_Data = torch.FloatTensor(X_Train_Data)
    Y_Train_Data = torch.FloatTensor(Y_Train_Data)
    X_Val_Data = torch.FloatTensor(X_Val_Data)
    Y_Val_Data = torch.FloatTensor(Y_Val_Data)
    X_Test_Data = torch.FloatTensor(X_Test_Data)
    Y_Test_Data = torch.FloatTensor(Y_Test_Data)

    if input_modal== 'ECG':
        X_Train_Data = X_Train_Data[:, [0], :]
        X_Val_Data = X_Val_Data[:, [0], :]
        X_Test_Data = X_Test_Data[:, [0], :]
    elif input_modal == 'PPG':
        X_Train_Data = X_Train_Data[:, [1], :]
        X_Val_Data = X_Val_Data[:, [1], :]
        X_Test_Data = X_Test_Data[:, [1], :]
    elif input_modal == 'ABP':
        X_Train_Data = X_Train_Data[:, [2], :]
        X_Val_Data = X_Val_Data[:, [2], :]
        X_Test_Data = X_Test_Data[:, [2], :]

    train_set = TensorDataset(X_Train_Data, Y_Train_Data)
    val_set = TensorDataset(X_Val_Data, Y_Val_Data)
    test_set = TensorDataset(X_Test_Data, Y_Test_Data)
    print('train_set',len(train_set))
    print('val_set',len(val_set))
    print('test_set',len(test_set))

    if train_sample:
        set_seed(42)

        # 计算 25% 的数据量
        subset_size = int(len(train_set) * 0.25)
        # 随机生成数据索引
        indices = np.random.permutation(len(train_set))
        subset_indices = indices[:subset_size]
        # 使用 SubsetRandomSampler 采样
        sampler = SubsetRandomSampler(subset_indices)

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  num_workers=0,sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size,  num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    X_shape = X_Test_Data.shape
    Y_shape = Y_Test_Data.shape

    print('查看train中Y分布',np.unique(Y_Train_Data, return_counts=True))
    print('查看val中Y分布',np.unique(Y_Val_Data, return_counts=True))
    print('查看test中Y分布',np.unique(Y_Test_Data, return_counts=True))

    return train_loader, val_loader, test_loader, X_shape, Y_shape

class CustomDataset(Dataset):
    '''适合于args.preload==true时，保存数据集'''
    def __init__(self, datas, labels,metas):
        """
        tensor_dataset: 一个 TensorDataset 实例，包含了 ecgs, locations, ys
        record_names: 记录标识的列表
        """
        self.tensor_dataset = TensorDataset(datas, labels)
        self.record_names = metas

    def __getitem__(self, index):
        data,  y = self.tensor_dataset[index]
        meta = self.record_names[index]
        return data,y,meta

    def __len__(self):
        return len(self.tensor_dataset)

def standardize_meta(meta):
    '''标准化meta数据'''
    # 批量处理的步骤：
    # 1. 时间（hour）的周期性编码
    hours = meta[:, 0]  # 提取 hour 列 (n_samples,)
    time_sin = np.sin(2 * np.pi * hours / 24)  # 正弦周期性编码
    time_cos = np.cos(2 * np.pi * hours / 24)  # 余弦周期性编码
    # 2. 年龄（age）的 Min-Max 标准化
    ages = meta[:, 1]  # 提取 age 列 (n_samples,)
    min_age = np.min(ages)  # 计算最小值
    max_age = np.max(ages)  # 计算最大值
    print('min_age',min_age,'max_age',max_age)
    age_normalized = (ages - min_age) / (max_age - min_age)  # Min-Max 标准化
    # 3. 性别（gender）不需要编码，直接使用
    genders = meta[:, 2]  # 提取 gender 列 (n_samples,)
    # 4. 拼接结果：将时间编码、标准化后的年龄、性别拼接成新的特征向量
    meta_processed = np.stack([time_sin, time_cos, age_normalized, genders], axis=1)

    return meta_processed

def time_embedding_plot():
    '''画图看一看适用于hour的正余弦编码'''

    # 创建数据
    data = pd.DataFrame({'hour': list(range(24))})
    # data.plot()

    # 生成sin/cos特征
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 23.0)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 23.0)

    # sin/cos绘图
    plt.figure(figsize=(3, 2))
    plt.plot(data['hour'], data['hour_sin'], label='sin',linewidth=3)
    plt.plot(data['hour'], data['hour_cos'], label='cos',linewidth=2)
    # data['hour_sin'].plot()
    # data['hour_cos'].plot()

    # (sin,cos)组合绘图
    # plt.figure(figsize=(3, 2))
    # plt.scatter(data['hour_sin'], data['hour_cos'])
    # plt.xlabel('hour_sin')
    # plt.ylabel('hour_cos')
    # 正方形
    # plt.gca().set_aspect('equal', adjustable='box')
    # data.plot.scatter('hour_sin', 'hour_cos').set_aspect('equal')
    plt.show()


def load_data_fold_meta(data_dir='C:/mancy/experiment/cross_modal_re/data/', task='II_Pleth_new',downtask='all', batch_size=128,debug=False,resample={'switch':False,'num_samples':1024},target_fold=0,input_modal='all',split_method='inter',train_sample=False,filtered=False,save_test = False):
    '''
    读取包含下游任务标签的数据集
    :param data_dir:
    :param task:
    :param downtask:
    :param batch_size:
    :param debug:
    :param resample:
    :return:
    '''

    merged_idx = np.load(data_dir + task + '/merged_idx.npy')
    merged_data = np.load(data_dir + task + '/merged_data.npy')
    merged_label = np.load(data_dir + task + '/merged_label.npy')
    merged_time = np.load(data_dir + task + '/merged_time_stamps.npy')

    if task == 'II_Pleth_ABP' and input_modal == 'all':
        merged_data[:,2,:] = merged_data[:,2,:]/100

    # plt.figure(figsize=(10, 6))
    # plt.plot(merged_data[0, 0, :])

    # 滤波
    if filtered:
        fs = merged_data.shape[-1]/10
        ecg_data = [filter_ecg_channel(x[0, :],fs) for x in merged_data]
        ppg_data = [filter_ppg_channel(x[1, :],fs) for x in merged_data]
        abp_data = [filter_abp_channel(x[2, :],fs) for x in merged_data]

        merged_data = np.stack([ecg_data,ppg_data,abp_data],axis=1)
        del ecg_data,ppg_data,abp_data
    #
    #     plt.plot(merged_data[0,0,:])
    #
    # plt.show()

    idx_meta = idx2meta_map(task)
    mapping_idx2seg_time = idx2seg_time_map(task)



    merged_meta = []
    for i, id in enumerate(merged_idx):
        # 将meta和time都放进去
        meta = {}

        seg_time = mapping_idx2seg_time[id]
        seg_time = datetime.strptime(seg_time, "%Y-%m-%d %H:%M:%S")
        time_stamp = seg_time + timedelta(seconds=merged_time[i])
        # 提取time_stamp的hour
        time_stamp_hour = time_stamp.hour

        meta['time'] = time_stamp_hour

        meta['age'] = idx_meta[id]['age_imputed']
        meta['gender'] = idx_meta[id]['gender_imputed']
        merged_meta.append(meta)

    # meta: hour, age, sex
    merged_meta_np =  np.array([[float(item['time']), float(item['age']), float(item['gender'])] for item in merged_meta])
    # meta:time_sin, time_cos, age, gender(M:1, F:0)
    merged_meta_np = standardize_meta(merged_meta_np)
    print('final merged_meta',merged_meta_np.shape)


    if split_method == 'inter':
        X_Train_Data, y_Train_Data,idx_Tarin_Data, X_Val_Data, y_Val_Data, idx_Val_Data, X_Test_Data, y_Test_Data ,idx_Test_Data,train_meta,val_meta,test_meta = k_fold_split(merged_idx, merged_data, merged_label,merged_meta=merged_meta_np, target_fold=target_fold)
    else:
        X_Train_Data, y_Train_Data,idx_Tarin_Data, X_Val_Data, y_Val_Data, idx_Val_Data, X_Test_Data, y_Test_Data,idx_Test_Data,train_meta,val_meta,test_meta = random_split(merged_idx, merged_data, merged_label,merged_meta=merged_meta_np)

    downtask_map= {'heart_failure':0,'hypertensive':1,'diabetes':2,'kidney_failure':3}
    if downtask == 'all':
        y_train = y_Train_Data
        y_val = y_Val_Data
        y_test = y_Test_Data
    else:
        y_train = y_Train_Data[:,[downtask_map[downtask]]]
        y_val =y_Val_Data[:,[downtask_map[downtask]]]
        y_test = y_Test_Data[:,[downtask_map[downtask]]]

    # 把label中所有的nan变为0
    Y_Train_Data = np.nan_to_num(y_train)
    Y_Val_Data = np.nan_to_num(y_val)
    Y_Test_Data = np.nan_to_num(y_test)

    # 是否重采样到512
    if resample['switch']:
        num_samples = resample['num_samples']
        X_Train_Data = np.array([scipy.signal.resample(x, num_samples, axis=-1) for x in X_Train_Data])
        X_Val_Data = np.array([scipy.signal.resample(x, num_samples, axis=-1) for x in X_Val_Data])
        X_Test_Data = np.array([scipy.signal.resample(x, num_samples, axis=-1) for x in X_Test_Data])

    if debug:
        debug_batch = 1000
        X_Train_Data = X_Train_Data[:debug_batch]
        Y_Train_Data = Y_Train_Data[:debug_batch]
        X_Val_Data = X_Val_Data[:debug_batch]
        Y_Val_Data = Y_Val_Data[:debug_batch]
        X_Test_Data = X_Test_Data[:debug_batch]
        Y_Test_Data = Y_Test_Data[:debug_batch]
    X_Train_Data = torch.FloatTensor(X_Train_Data)
    Y_Train_Data = torch.FloatTensor(Y_Train_Data)
    X_Val_Data = torch.FloatTensor(X_Val_Data)
    Y_Val_Data = torch.FloatTensor(Y_Val_Data)
    X_Test_Data = torch.FloatTensor(X_Test_Data)
    Y_Test_Data = torch.FloatTensor(Y_Test_Data)

    if input_modal== 'ECG':
        X_Train_Data = X_Train_Data[:, [0], :]
        X_Val_Data = X_Val_Data[:, [0], :]
        X_Test_Data = X_Test_Data[:, [0], :]
    elif input_modal == 'PPG':
        X_Train_Data = X_Train_Data[:, [1], :]
        X_Val_Data = X_Val_Data[:, [1], :]
        X_Test_Data = X_Test_Data[:, [1], :]
    elif input_modal == 'ABP':
        X_Train_Data = X_Train_Data[:, [2], :]
        X_Val_Data = X_Val_Data[:, [2], :]
        X_Test_Data = X_Test_Data[:, [2], :]

    # train_set = TensorDataset(X_Train_Data, Y_Train_Data)
    # val_set = TensorDataset(X_Val_Data, Y_Val_Data)
    # test_set = TensorDataset(X_Test_Data, Y_Test_Data)
    train_set = CustomDataset(X_Train_Data, Y_Train_Data,train_meta)
    val_set = CustomDataset(X_Val_Data, Y_Val_Data,val_meta)
    test_set = CustomDataset(X_Test_Data, Y_Test_Data,test_meta)
    print('train_set',len(train_set))
    print('val_set',len(val_set))
    print('test_set',len(test_set))

    if save_test:
        # 保存Y_Test_Data为npy文件
        np.save(data_dir + task + '/Test_label.npy', Y_Test_Data.numpy())
        np.save(data_dir + task + '/Test_meta.npy', test_meta.numpy())

    if train_sample:
        set_seed(42)

        # 计算 25% 的数据量
        subset_size = int(len(train_set) * 0.25)
        # 随机生成数据索引
        indices = np.random.permutation(len(train_set))
        subset_indices = indices[:subset_size]
        # 使用 SubsetRandomSampler 采样
        sampler = SubsetRandomSampler(subset_indices)

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  num_workers=0,sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size,  num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    X_shape = X_Test_Data.shape
    Y_shape = Y_Test_Data.shape

    print('查看train中Y分布',np.unique(Y_Train_Data, return_counts=True))
    print('查看val中Y分布',np.unique(Y_Val_Data, return_counts=True))
    print('查看test中Y分布',np.unique(Y_Test_Data, return_counts=True))
    return train_loader, val_loader, test_loader, X_shape, Y_shape,train_meta
def k_fold_split(merged_idx, merged_data, merged_label,merged_meta=None, target_fold=0):
    '''病人间的划分'''
    unique_idx = np.unique(merged_idx)
    # 生成10折交叉验证的划分
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # 保存每一折的索引
    folds = list(kf.split(unique_idx))
    # 获取训练集和测试集索引
    train_idx, test_idx = folds[target_fold]
    # 提取训练集和测试集的唯一索引
    train_unique_idx = unique_idx[train_idx]
    test_unique_idx = unique_idx[test_idx]
    print('test_unique_idx',test_unique_idx)

    # 提取训练集和测试集数据
    train_data = merged_data[np.isin(merged_idx, train_unique_idx)]
    test_data = merged_data[np.isin(merged_idx, test_unique_idx)]

    train_label = merged_label[np.isin(merged_idx, train_unique_idx)]
    test_label = merged_label[np.isin(merged_idx, test_unique_idx)]
    test_idx = merged_idx[np.isin(merged_idx, test_unique_idx)]

    # 如果需要验证集，可以从训练集中进一步划分
    # 例如按 8:1:1 比例分割训练集和验证集
    proportion = [0.8, 0.1, 0.1]
    val_split = int((proportion[1]/(proportion[0]+proportion[1])) * len(train_unique_idx))

    val_idx = train_unique_idx[:val_split]
    final_train_idx = train_unique_idx[val_split:]

    val_data = merged_data[np.isin(merged_idx, val_idx)]
    val_label = merged_label[np.isin(merged_idx, val_idx)]
    val_idx = merged_idx[np.isin(merged_idx, val_idx)]

    final_train_data = merged_data[np.isin(merged_idx, final_train_idx)]
    final_train_label = merged_label[np.isin(merged_idx, final_train_idx)]
    final_train_index = merged_idx[np.isin(merged_idx, final_train_idx)]

    if merged_meta is not None:
        merged_meta_np = merged_meta
        merged_meta_torch = torch.FloatTensor(merged_meta_np)
        val_meta = merged_meta_torch[np.isin(merged_idx, val_idx)]
        final_train_meta = merged_meta_torch[np.isin(merged_idx, final_train_idx)]
        test_meta = merged_meta_torch[np.isin(merged_idx, test_unique_idx)]
        return final_train_data, final_train_label,final_train_index, val_data, val_label,val_idx, test_data, test_label, test_idx,final_train_meta,val_meta,test_meta
    else:
        return final_train_data, final_train_label,final_train_index, val_data, val_label,val_idx, test_data, test_label, test_idx

def random_split(merged_idx, merged_data, merged_label,merged_meta=None,):
    '''适用于intra（病人内）的随机划分'''
    num_available_segments = len(merged_idx)
    set_seed(42)
    indices = np.random.permutation(num_available_segments)
    # 划分比例
    proportion = [0.8, 0.1, 0.1]
    # 按照相同的索引顺序打乱数据
    shuffled_data = merged_data[indices]
    shuffled_idx = merged_idx[indices]
    shuffled_label = merged_label[indices]

    # train
    train_end = int(proportion[0] * num_available_segments)
    val_end = int((proportion[0] + proportion[1]) * num_available_segments)
    train_data = shuffled_data[:train_end]
    train_idx = shuffled_idx[:train_end]
    train_label = shuffled_label[:train_end]
    # val
    val_data = shuffled_data[train_end:val_end]
    val_idx = shuffled_idx[train_end:val_end]
    val_label = shuffled_label[train_end:val_end]
    # test
    test_data = shuffled_data[val_end:]
    test_idx = shuffled_idx[val_end:]
    test_label = shuffled_label[val_end:]

    plt.plot(test_data[237,0,:])
    plt.show()
    print('idx',test_idx[237])
    print('label',test_label[237])

    if merged_meta is not None:
        merged_meta_np = merged_meta
        merged_meta_torch = torch.FloatTensor(merged_meta_np)
        train_meta = merged_meta_torch[indices[:train_end]]
        val_meta = merged_meta_torch[indices[train_end:val_end]]
        test_meta = merged_meta_torch[indices[val_end:]]

        return train_data, train_label, train_idx, val_data, val_label, val_idx, test_data, test_label, test_idx,train_meta, val_meta, test_meta
    else:
        return train_data, train_label, train_idx, val_data, val_label, val_idx, test_data, test_label, test_idx


def load_three_modalities(data_dir='C:/mancy/experiment/cross_modal_re/data/', task='ppg_abp_ecg_mimic_ii', batch_size=128,debug=False,resample={'switch':False,'num_samples':1024}):
    '''仅适用于ppg_abp_ecg_mimic_ii数据集'''
    train_path = data_dir + task+ '/' + task + '_train_data.npy'
    val_path = data_dir + task + '/' + task + '_val_data.npy'
    test_path = data_dir + task + '/' + task + '_test_data.npy'

    X_Train_Ori = np.load(train_path, allow_pickle=True)
    X_Val_Ori = np.load(val_path, allow_pickle=True)
    X_Test_Ori = np.load(test_path, allow_pickle=True)

    # 是否重采样到512
    if resample['switch']:
        num_samples = resample['num_samples']
        X_Train_Ori = np.array([scipy.signal.resample(x, num_samples, axis=-1) for x in X_Train_Ori])
        X_Val_Ori = np.array([scipy.signal.resample(x, num_samples, axis=-1) for x in X_Val_Ori])
        X_Test_Ori = np.array([scipy.signal.resample(x, num_samples, axis=-1) for x in X_Test_Ori])

    if debug:
        debug_batch = 1000
        X_Train_Ori = X_Train_Ori[:debug_batch]
        X_Val_Ori = X_Val_Ori[:debug_batch]
        X_Test_Ori = X_Test_Ori[:debug_batch]
    X_Train_Data = torch.FloatTensor(X_Train_Ori)
    X_Val_Data = torch.FloatTensor(X_Val_Ori)
    X_Test_Data = torch.FloatTensor(X_Test_Ori)

    del X_Train_Ori, X_Val_Ori, X_Test_Ori

    # ppg, abp ,ecg 变换顺序
    X_Train_Data = X_Train_Data.permute(1,2,0)
    X_Val_Data = X_Val_Data.permute(1,2,0)
    X_Test_Data = X_Test_Data.permute(1,2,0)

    train_set = TensorDataset(X_Train_Data)
    val_set = TensorDataset(X_Val_Data)
    test_set = TensorDataset(X_Test_Data)
    print('train_set',len(train_set))
    print('val_set',len(val_set))
    print('test_set',len(test_set))

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    print(X_Test_Data.shape)

    return train_loader, val_loader, test_loader, X_Test_Data.shape


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def idx2meta_map(task):
    '''根据索引寻找filename，根据filename寻找metadata'''
    df_map = pd.read_csv(f'data/matching_records_{task}.csv')
    idx = df_map.index
    filename = df_map['seg_name'].str.split('_').str[0]
    df_meta_map = pd.read_csv('data/waveform_match_diagnosis_metadata_complete.csv')
    filename_meta = df_meta_map['filename'].str.split('.').str[0]
    meta_dict = df_meta_map[['age_imputed','gender_imputed']].set_index(filename_meta).to_dict(orient='index')
    # print(meta_dict)

    # 创建索引到元数据的映射
    idx2meta = {}
    for i, f in zip(idx, filename):
        if f in meta_dict:
            idx2meta[i] = meta_dict[f]
        else:
            idx2meta[i] = None
    # print(idx2meta)
    return idx2meta

def idx2seg_time_map(task):
    df_map = pd.read_csv(f'data/matching_records_{task}.csv')
    idx = df_map.index
    seg_time = df_map['seg_time']

    # 生成index的映射idx-->seg_time
    mapping = {i: time for i, time in zip(idx, seg_time)}
    return mapping


if __name__ == '__main__':
    # train_loader, val_loader, test_loader, X_shape, Y_shape = load_data_fold(task='II_Pleth_ABP',target_fold=2,input_modal='ECG',downtask='heart_failure',split_method='intra')
    # train_loader, val_loader, test_loader, X_shape, Y_shape = load_downstream_data(target_fold=2,input_modal='all',down_task_split_method='inter')
    train_loader, val_loader, test_loader, X_shape, Y_shape,train_meta = load_data_fold_meta(task='II_Pleth_ABP', target_fold=2,
                                                                             input_modal='ECG',
                                                                             downtask='all',
                                                                             split_method='inter',
                                                                             filtered=False,
                                                                             save_test=True)
    # print(train_meta.shape)
    # time_embedding_plot()






