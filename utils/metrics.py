import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastdtw import fastdtw
import pywt
from tqdm import tqdm

def SSD(y, y_pred):
    '''
    计算信号的平方和误差 (Sum of Squared Differences, SSD)

    解释:
    SSD值越低越好，表示预测信号与原始信号的差异越小。
    shape: (batch, signal_length)
    '''

    return np.sum(np.square(y - y_pred), axis=1)  # axis 1 is the signal dimension

def MSE(y, y_pred):
    '''
    计算信号的均方误差 (Mean Squared Error, MSE)
    shape: (batch, signal_length)
    '''
    return np.mean(np.square(y - y_pred),axis=1)  # axis 1 is the signal dimension


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
    return np.max(np.abs(y - y_pred), axis=1)  # axis 1 is the signal dimension


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

def PRD_formal(y, y_pred):
    """
    计算信号的百分均方根差 (Percentage Root Mean Square Difference, PRD)
    shape: (batch, signal_length)
    解释:
    PRD值越低越好，表示预测信号与原始信号的相对差异越小。
    """
    # 判断y的维度
    if len(y.shape) == 1:
        y = y.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)
    N = np.sum(np.square(y_pred - y), axis=1)
    D = np.sum(np.square(y), axis=1)

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


def SNR_improvement(y_in, y_out, y_clean):
    """
    计算信号的信噪比改善 (SNR Improvement)

    参数:
    y_in: 输入信号
    y_out: 输出信号
    y_clean: 干净信号
    shape: (batch, signal_length)
    返回:
    信噪比改善值

    解释:
    SNR改善值越高越好，表示信号处理后的信噪比提升越大。
    """
    return SNR(y_clean, y_out) - SNR(y_clean, y_in)



def DTW(y, y_pred):
    """
    计算信号的动态时间规整 (Dynamic Time Warping, DTW)

    参数:
    y: 原始信号
    y_pred: 预测信号
    shape: (batch, signal_length)
    返回:
    每个信号的DTW值

    解释:
    DTW值越低越好，表示预测信号与原始信号的相似度越高。
    """

    dtw_scores = []
    for i in range(len(y)):
        distance, _ = fastdtw(y[i], y_pred[i])
        dtw_scores.append(distance)

    return np.array(dtw_scores)
def MSEWPRD_weight(y_true):
    # DWT decomposition
    wavelet = 'db8'
    # 执行单级离散小波变换
    coeffs = pywt.wavedec(y_true, wavelet, level=5)  # level+1, 其中0是近似系数，1-5是细节系数

    # multiscale entropy information measure
    # 1. The mean wavelet subband energy
    E = np.zeros(len(coeffs))
    for k in range(len(coeffs)):
        E[k] = np.mean(np.square(coeffs[k]))
    E_sum = np.sum(E)
    E_normalized = E / E_sum
    # 2.multiscale entropy information measure
    H = np.zeros(len(coeffs))
    for k in range(len(coeffs)):
        H[k] = -E_normalized[k] * np.log(E_normalized[k])
    return H
def MSEWPRD(y,y_pred):
    '''Multiscale Entropy-Based Weighted PRD Measure
        shape: (batch, signal_length)
    '''
    MSEWPRD_scores = []
    for i in tqdm(range(len(y))):
        # y_true = np.squeeze(y[i], axis=0)
        # y_hat = np.squeeze(y_pred[i], axis=0)
        y_true = y[i]
        y_hat = y_pred[i]
        # the weight for the th approximation band
        H = MSEWPRD_weight(y_true)
        # DWT coefficients
        coeffs_true = pywt.wavedec(y_true, 'db8', level=5)
        coeffs_hat = pywt.wavedec(y_hat, 'db8', level=5)

        # PRD
        MSEWPRD_score = np.sum([H[k] * PRD_formal(coeffs_true[k], coeffs_hat[k]) for k in range(6)])
        MSEWPRD_scores.append(MSEWPRD_score)

    return np.array(MSEWPRD_scores)


def mertrics_all(y,y_pred):
    """
    计算所有的指标
    """
    metrics = {}
    metrics_mean = {}
    metrics['SSD'] = SSD(y, y_pred)
    metrics['MSE'] = MSE(y, y_pred)
    metrics['RMSE'] = RMSE(y, y_pred)
    metrics['MAD'] = MAD(y, y_pred)
    metrics['PRD'] = PRD(y, y_pred)
    metrics['COS_SIM'] = COS_SIM(y, y_pred)
    metrics['R_suqared'] = R_suqared(y, y_pred)
    metrics['SNR'] = SNR(y, y_pred)
    metrics['DTW'] = DTW(y, y_pred)
    metrics['MSEWPRD'] = MSEWPRD(y, y_pred)

    # 每个指标都求平均值
    for key in metrics.keys():
        metrics_mean[key] = np.mean(metrics[key])
    return metrics_mean




if __name__ == '__main__':
    y_true = np.load('../data/II_Pleth_new/all_original_modal1.npy')
    y_pred = np.load('../output/II_Pleth_new/DualVAE0_dropout_contrastive/all_reconstructed_modal1.npy')
    #
    score = MSEWPRD(y_true,y_pred)
    print(score)