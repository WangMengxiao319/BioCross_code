"""推理所用波形评估指标。"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def MSE(y, y_pred):
    '''
    计算信号的均方误差 (Mean Squared Error, MSE)
    shape: (batch, signal_length)
    '''
    return np.mean(np.square(y - y_pred),axis=1)

def RMSE(y, y_pred):
    '''
    计算信号的均方根误差 (Root Mean Squared Error, RMSE)
    shape: (batch, signal_length)
    '''
    return np.sqrt(MSE(y, y_pred))

def MAD(y, y_pred):
    """
    计算信号的最大绝对偏差 (Maximum Absolute Deviation, MAD)
    shape: (batch, signal_length)
    解释:
    MAD值越低越好，表示预测信号与原始信号的最大偏差越小。
    """
    return np.max(np.abs(y - y_pred), axis=1)

def PRD(y, y_pred):
    """
    计算信号的百分均方根差 (Percentage Root Mean Square Difference, PRD)
    shape: (batch, signal_length)
    解释:
    PRD值越低越好，表示预测信号与原始信号的相对差异越小。
    """
    N = np.sum(np.square(y_pred - y), axis=1)
    D = np.sum(np.square(y_pred - np.mean(y)), axis=1)

    PRD = np.sqrt(N / D) * 100

    return PRD

def COS_SIM(y, y_pred):
    """
    计算信号的余弦相似度 (Cosine Similarity)
    shape: (batch, signal_length)
    解释:
    余弦相似度值越高越好，表示预测信号与原始信号的方向越一致。
    """
    cos_sim = []
    # y = np.squeeze(y, axis=-1)
    # y_pred = np.squeeze(y_pred, axis=-1)
    for idx in range(len(y)):
        kl_temp = cosine_similarity(y[idx].reshape(1, -1), y_pred[idx].reshape(1, -1))
        cos_sim.append(kl_temp)

    cos_sim = np.array(cos_sim)
    return cos_sim

def R_suqared(y, y_pred):
    """
    计算信号的R^2值 (Coefficient of Determination, R^2)
    y: (batch, signal_length)
    解释:
    R^2值越接近1越好，表示预测信号与原始信号的相关性越高。
    """
    # 计算每个样本的均值
    y_mean = np.mean(y, axis=1, keepdims=True)

    # 计算 SS_res 和 SS_tot
    ss_res = np.sum((y - y_pred) ** 2, axis=1)
    ss_tot = np.sum((y - y_mean) ** 2, axis=1)

    # 计算 R² 值
    r2_scores = 1 - (ss_res / ss_tot)
    return r2_scores

def SNR(y1, y2):
    """
    计算信号的信噪比 (Signal-to-Noise Ratio, SNR)
    参数:
    y1: 原始信号
    y2: 带噪声的信号
    shape: (batch, signal_length)
    返回:
    每个信号的SNR值
    解释:
    SNR值越高越好，表示信号中的噪声成分越少。
    """
    N = np.sum(np.square(y1), axis=1)
    D = np.sum(np.square(y2 - y1), axis=1)

    SNR = 10 * np.log10(N / D)

    return SNR
