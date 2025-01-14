# -*- coding: utf-8 -*-
# @Time : 2024/7/18 18:40
# @Author : 王梦潇
# @File : utils.py
# Function:
import math
from matplotlib.ticker import MaxNLocator
from biosppy.signals.tools import filter_signal
from biosppy.signals.abp import abp
from biosppy.signals.ppg import ppg
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import pad
from scipy.io import savemat
import torch
# from rich import print
# from rich.progress import track
# import os
# from scipy.signal import butter, filtfilt, iirnotch, welch

'''画图'''
def ECGplot_three_signals(sigg1, sigg2,sigg3, new_sig1=None, new_sig2=None,new_sig3=None,fs=100.0,epoch=0,name=['ECG','PPG','ABP']):
    '''
    :param sigg1: shape(batch,signal_length) eg.(1,1000)
    :param sigg2:
    :param sigg3:
    :param new_sig1:
    :param new_sig2:
    :param new_sig3:
    :param fs:
    :param epoch:
    :param name:
    :return:
    '''

    fig, [ax1,ax2,ax3] = plt.subplots(3,1,figsize=(8, 8), dpi=150)

    # SIGNAL 1
    #     ax1.set_xticks(np.arange(0, 12, 0.5))
    # ax1.set_yticks(np.arange(-1.0, +3.0, 0.5))
    # ax1.minorticks_on()
    # ax1.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # ax1.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))
    t = np.arange(0, sigg1.shape[-1] * 1 / fs, 1 / fs)

    ax1.plot(t, sigg1[0], label=f"{name[0]} signal", color='k')
    # ymin = -1.5
    # ymax = 2.5
    if new_sig1 is not None:
        ax1.plot(t, new_sig1[0] + 1, label=f"new {name[0]} signal", color='g')
        # ymax = ymax + 1
    ax1.legend(loc='upper right', bbox_to_anchor=(.96, 1.0))
    # ax1.set(xlabel='time (s)')
    #     ax.set(ylabel='Voltage (mV)')
    ax1.autoscale(tight=True)
    ax1.set_xlim(left=0, right=10)

    # ax1.set_ylim(top=ymax, bottom=ymin)
    # title
    ax1.set_title(f'{name[0]} Signal in epoch{epoch}')

    # SIGNAL 2
    #     ax2.set_xticks(np.arange(0, 12, 0.5))
    # ax2.set_yticks(np.arange(-1.0, +3.0, 0.5))
    # ax2.minorticks_on()
    # ax2.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # ax2.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))

    ax2.plot(t, sigg2[0], label=f"{name[1]} signal", color='k')
    # ymin = -1.5
    # ymax = 2.5
    if new_sig2 is not None:
        ax2.plot(t, new_sig2[0] + 1, label=f"new {name[1]} signal", color='g')
        # ymax = ymax + 1
    ax2.legend(loc='upper right', bbox_to_anchor=(.96, 1.0))
    # ax2.set(xlabel='time (s)')
    #     ax.set(ylabel='Voltage (mV)')
    ax2.autoscale(tight=True)
    ax2.set_xlim(left=0, right=10)

    # ax2.set_ylim(top=ymax, bottom=ymin)
    # title
    ax2.set_title(f'{name[1]} Signal in epoch{epoch}')

    # SIGNAL 3
    # ax3.set_yticks(np.arange(-1.0, +3.0, 0.5))
    # ax3.minorticks_on()
    # ax3.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # ax3.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))

    ax3.plot(t, sigg3[0], label=f"{name[2]} signal", color='k')
    # ymin = -1.5
    # ymax = 2.5
    if new_sig3 is not None:
        ax3.plot(t, new_sig3[0] + 1, label=f"new {name[2]} signal", color='g')
        # ymax = ymax + 1
    ax3.legend(loc='upper right', bbox_to_anchor=(.96, 1.0))
    ax3.set(xlabel='time (s)')
    #     ax.set(ylabel='Voltage (mV)')
    ax3.autoscale(tight=True)
    ax3.set_xlim(left=0, right=10)

    # ax3.set_ylim(top=ymax, bottom=ymin)
    # title
    ax3.set_title(f'{name[2]} Signal in epoch{epoch}')

    # 设置共享的 x 轴刻度限制
    ax3.xaxis.set_major_locator(MaxNLocator(prune='lower'))


    ## 以下是只包含y坐标，方便画示意图的内容，一般隐掉
    # for ax in [ax1, ax2, ax3]:
    #     # 去掉所有边框 (spines)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['bottom'].set_visible(False)
    #     # ax.spines['left'].set_visible(False)
    #
    #     # 去掉刻度
    #     ax.set_xticks([])
    #     # ax.set_yticks([])
    #     ax.set(xlabel='', ylabel='')
    #
    #     # 清除标题
    #     ax.set_title('')
    #
    #     ax.get_legend().remove()
    #
    #     ax.set_xlim(left=0, right=5)

    return plt

def ECGplot_two_signals(sigg1, sigg2, new_sig1=None, new_sig2=None, save=False, i_path=None,fs=100.0,epoch=0,name=['ECG','PPG']):
    # print(sigg1.shape)
    # print(new_sig1.shape)
    # print(sigg2.shape)
    # print(new_sig2.shape)

    fig, [ax1,ax2] = plt.subplots(2,1,figsize=(8, 8), dpi=150)

    # SIGNAL 1
    #     ax1.set_xticks(np.arange(0, 12, 0.5))
    # ax1.set_yticks(np.arange(-1.0, +3.0, 0.5))
    # ax1.minorticks_on()
    # ax1.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # ax1.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))
    t = np.arange(0, sigg1.shape[-1] * 1 / fs, 1 / fs)

    ax1.plot(t, sigg1[0], label=f"{name[0]} signal", color='k')
    # ymin = -1.5
    # ymax = 2.5
    if new_sig1 is not None:
        ax1.plot(t, new_sig1[0] + 1, label=f"new {name[0]} signal", color='g')
        # ymax = ymax + 1
    ax1.legend(loc='upper right', bbox_to_anchor=(.96, 1.0))
    # ax1.set(xlabel='time (s)')
    #     ax.set(ylabel='Voltage (mV)')
    ax1.autoscale(tight=True)
    ax1.set_xlim(left=0, right=10)

    # ax1.set_ylim(top=ymax, bottom=ymin)
    # title
    ax1.set_title(f"new {name[0]}Signal in epoch{epoch}")

    # SIGNAL 2
    #     ax2.set_xticks(np.arange(0, 12, 0.5))
    # ax2.set_yticks(np.arange(-1.0, +3.0, 0.5))
    # ax2.minorticks_on()
    # ax2.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # ax2.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))
    t = np.arange(0, sigg2.shape[-1] * 1 / fs, 1 / fs)

    ax2.plot(t, sigg2[0], label=f"new {name[1]} signal", color='k')
    # ymin = -1.5
    # ymax = 2.5
    if new_sig2 is not None:
        ax2.plot(t, new_sig2[0] + 1, label=f"new {name[1]} signal", color='g')
        # ymax = ymax + 1
    ax2.legend(loc='upper right', bbox_to_anchor=(.96, 1.0))
    ax2.set(xlabel='time (s)')
    #     ax.set(ylabel='Voltage (mV)')
    ax2.autoscale(tight=True)
    ax2.set_xlim(left=0, right=10)

    # ax2.set_ylim(top=ymax, bottom=ymin)
    # title
    ax2.set_title(f"new {name[1]} Signal in epoch{epoch}")
    return plt

def ECGplot_one_signal(sigg1, new_sig1=None, save=False, i_path=None,fs=100.0,epoch=0, name='ECG',grid=False,xlabel=False,ylabel=False,legend=False,title=False,grid_color='grey',line_color='k',ax1=None,y_axis_format='%.2f'):
    print(sigg1.shape)
    if ax1 is None:
        # 如果没有传入 ax，创建一个新的 figure 和子图
        fig, ax1 = plt.subplots(figsize=(8, 8), dpi=150)
    if grid_color == 'red':
        color_major = 'red'
        color_minor = (1, 0.7, 0.7)
    t = np.arange(0, sigg1.shape[-1] * 1 / fs, 1 / fs)

    ax1.plot(t, sigg1[0], label=f"{name} signal", color=line_color)
    # ymin = -0.5
    # ymax = 0.8
    if new_sig1 is not None:
        ax1.plot(t, new_sig1[0] + 1, label=f"new {name} signal", color='g')
        # ymax = ymax + 1
    # SIGNAL 1
    # ax1.set_yticks(np.arange(-1.0, +3.0, 0.5))

    else:
        color_major = '#E0E0E0'
        color_minor = '#F0F0F0'
    if grid:
        ax1.minorticks_on()
        ax1.set_xticks(np.arange(0, 10, 0.2))
        # ax1.set_yticks(np.arange(ymin, ymax, 0.5))

        # 红色
        # ax1.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        # ax1.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))
        # 灰色
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color=color_major)
        ax1.grid(which='minor', linestyle='-', linewidth='0.5', color=color_minor)



    if legend:
        ax1.legend(loc='upper right', bbox_to_anchor=(.96, 1.0))

    if xlabel:
        ax1.set(xlabel='time (s)')

    if ylabel:
        ax1.set_ylabel(name,rotation=0,labelpad=40)
    #     ax.set(ylabel='Voltage (mV)')
    ax1.autoscale(tight=True)
    ax1.set_xlim(left=0, right=10)

    # ax1.set_ylim(top=ymax, bottom=ymin)
    # title

    if title:
        ax1.set_title(f'{name} Signal in epoch{epoch}')

    # 不显示横纵坐标刻度
    ax1.set_xticklabels([])

    # 设置纵轴刻度为保留两位小数
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{y_axis_format}' % x))
    # ax1.set_yticklabels([])

    return plt
'''学习率调整策略'''

def adjust_learning_rate(optimizer, epoch, warmup_epochs,initial_lr, min_lr, epochs):
    """Decay the learning rate with half-cycle cosine after warmup
    Reference: Continual Self-supervised Learning: Towards Universal Multi-modal Medical Data Representation Learning
    """
    if epoch < warmup_epochs:
        lr = initial_lr * epoch / warmup_epochs
    else:
        lr = min_lr + (initial_lr - min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

'''滤波'''
def apply_filter_bandpass(signal,fs,older_ratio=0.3):
    '''滤波'''
    order = int(older_ratio * fs)
    # Filter signal
    signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                    order=order, frequency=[2,30],
                                    sampling_rate=fs)
    return signal

def apply_filter_highpass(signal, fs, older_ratio=0.3):
    '''滤波'''
    order = int(older_ratio * fs)
    # Filter signal
    signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='highpass',
                                    order=order, frequency=1,
                                    sampling_rate=fs)
    return signal

def apply_filter_lowpass(signal, fs,older_ratio=0.3):
    '''滤波'''
    order = int(older_ratio * fs)
    # Filter signal
    signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='lowpass',
                                    order=order, frequency=0.67,
                                    sampling_rate=fs)
    return signal

def median_filter(signal,fs):
    """
    一维中值矫正滤波函数
    :param signal: 输入信号，为一维数组
    :param kernel_size: 滤波器大小
    :return: 滤波后的信号
    """
    kernel_size = int(fs*0.8)
    pad_width = int((kernel_size - 1) // 2)
    padded_signal = np.pad(signal, pad_width, mode='edge')
    filtered_signal = np.zeros_like(signal)
    for i in range(len(signal)):

        filtered_signal[i] = signal[i] - np.median(padded_signal[i:i + kernel_size])
    return filtered_signal

def median_filter_batch(signal, fs):
    """
    A batch median 'high-pass' filter implementation for PyTorch.
    :param signal: Input signal tensor, shape (batch_size, 12, 5000)
    :param fs: Sampling frequency
    :return: High-pass filtered signal
    """
    # Calculate kernel size for the filter based on the sampling frequency
    kernel_size = int(fs * 0.8)+1  # 401
    # Calculate padding width
    pad_width = (kernel_size-1) // 2
    # Pad the signal on both sides using replicate mode
    padded_signal = pad(signal, (pad_width, pad_width), mode='replicate')

    # Use unfold to create sliding windows of size kernel_size
    windows = padded_signal.unfold(-1, kernel_size, 1)
    # Compute the median across the window size dimension
    medians = windows.median(dim=-1).values

    # Subtract the medians from the original signal to highlight rapid changes
    high_pass_filtered_signal = signal - medians

    return high_pass_filtered_signal

def filter_ecg_channel(data,fs):
    signal = apply_filter_highpass(data, fs=fs, older_ratio=0.3)
    return signal
def filter_ppg_channel(data,fs):
    filtered = apply_filter_lowpass(data, fs=fs, older_ratio=0.1)

    return filtered

def filter_abp_channel(data,fs):
    # POWERLINE_FREQ = 2
    # b_notch, a_notch = notch_filter(POWERLINE_FREQ, 30, fs)
    # tempfilt = filtfilt(b_notch, a_notch, data)
    # b2, a2 = butter_lowpass(16.0, fs)
    # tempfilt = filtfilt(b2, a2, tempfilt)
    # return tempfilt
    filtered = apply_filter_lowpass(data, fs,older_ratio=0.08)
    return filtered

'''读取数据'''
def read_wfdb_data(filename):
    '''读取wfdb数据'''
    record = wfdb.rdrecord(filename)
    # signals, fields = wfdb.rdsamp(filename)
    # annotation = wfdb.rdann(filename, 'atr')
    # print(annotation.__dict__)
    return record
def wfdb2mat(path,output_path):
    '''将wfdb格式的信号转换为mat格式'''
    record = wfdb.rdrecord(path)
    data = record.p_signal
    mat_dict = {'signal': data, 'fs': record.fs, 'units': record.units, 'sig_name': record.sig_name}
    savemat(output_path, mat_dict)

'''标准化'''
# def compute_mean_std():
#     samples, _, _ = torch.load("data/out/ecg_ppg_abp/train.pt")
#     samples = samples[:, :, 72500:75000]
#     mu = []
#     sigma = []
#     # for each signal
#     for i in range(samples.shape[1]):
#         vals = []
#         for x in range(len(samples)):
#             sample = samples[x, i]
#             # if current signal is available
#             if sample.sum() != 0.0:
#                 vals.append(sample)
#
#         vals = torch.cat(vals)
#         # 将 NaN 替换为 0
#         vals = torch.nan_to_num(vals, nan=0.0)
#         mu.append(vals.mean())
#         sigma.append(vals.std())
#     print('mu:', mu)
#     print('sigma:', sigma)
#     return mu, sigma
#
# def create_splits(mu, sigma):
#     """
#     Use the population mean and standard deviation to normalize each sample
#     Saves the output to out/population-norm/{split}.pt
#
#     Args:
#         mu: list of population means for each channel
#         sigma: list of population standard deviations for each channel
#
#     Returns:
#         None
#     """
#     output_path = "data/out/ecg_ppg_abp"
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     for split in ["train", "val", "test"]:
#         samples, ys, names = torch.load(f"data/out/ecg_ppg_abp/{split}.pt")
#         num_channels = samples.shape[1]
#         for i in range(num_channels):
#             mu_i = mu[i]
#             sigma_i = sigma[i]
#             for x in track(
#                 range(len(samples)), description="Normalizing...", transient=True
#             ):
#                 if samples[x, i].sum() != 0.0:
#                     samples[x, i] = (samples[x, i] - mu_i) / sigma_i
#
#         samples = samples.float()
#         torch.save((samples, ys, names), f"data/out/ecg_ppg_abp/{split}_po_norm.pt")

# def create_individual_splits():
#     """
#     Normalize each sample individually
#     Saves the output to data/out/sample-norm/{split}.pt
#     """
#     output_path = "data/out/ecg_ppg_abp"
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     for split in ["train", "val", "test"]:
#         samples, ys, names = torch.load(f"data/out/ecg_ppg_abp/{split}.pt")
#         num_channels = samples.shape[1]
#         for i in range(num_channels):
#             for x in track(
#                 range(len(samples)), description="Normalizing...", transient=True
#             ):
#                 if samples[x, i].sum() != 0.0:
#                     mu_i = samples[x, i, 72500:75000].mean()
#                     sigma_i = samples[x, i, 72500:75000].std()
#                     samples[x, i] = (samples[x, i] - mu_i) / sigma_i
#
#         samples = samples.float()
#         torch.save((samples, ys, names), output_path+f"/{split}_norm.pt")

if __name__ == '__main__':
    # mimic_ii_ecg_ppg = 'C:\mancy\experiment\cross_modal_re\data\ecg_ppg_mimic_ii\ecg_ppg_mimic_ii_train_data.npy'
    # mimic_iv_ecg_ppg = 'C:\mancy\experiment\cross_modal_re\data\II_Pleth\merged_data.npy'
    mimic_iv_ecg_ppg_abp = 'C:\mancy\experiment\cross_modal_re\data\II_Pleth_ABP\merged_data.npy'
    data= np.load(mimic_iv_ecg_ppg_abp)
    print(data.shape)
    id = 600
    length = 624
    sig1 = data[id,[0],:length]
    sig2 = data[id,[1],:length]
    sig3 = data[id,[2],:length]

    # # 画图
    # plt = ECGplot_three_signals(sig1,sig2,sig3,fs=62.4)
    # plt.show()
    #
    # # 检查数据中是否有nan的
    # print(np.isnan(ecg_ppg).any())

    # # 滤波
    fs = 62.4
    sig_ecg = filter_ecg_channel(sig1,fs=fs)
    sig_ppg = filter_ppg_channel(sig2,fs=fs)
    sig_abp = filter_abp_channel(sig3,fs=fs)
    plt1 = ECGplot_three_signals(sig1,sig2,sig3,fs=fs)
    plt1.show()
    plt2 = ECGplot_three_signals(sig_ecg,sig_ppg,sig_abp,fs=fs)
    plt2.show()

    # FFT频谱





