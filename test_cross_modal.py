'''this is inference(test) code for BioCross-FFT-cross (float 32)'''

import yaml
import os
from model.BioCross import TripleVAE_cmeta3_fft
from torch.utils.data import DataLoader, TensorDataset,Dataset
import numpy as np
import torch
import utils.metrics as metrics
import matplotlib.pyplot as plt

def meta_embedding(Q_z, meta_data):
    '''
    :param input: Q_z: latent space (numpy)
           meta_data: list-->object e.g. {'time': 1, 'age': 51.0, 'gender': 0.0}
    :return:
    '''
    # Convert meta_data tensor to numpy
    meta_data_np = np.array(meta_data)
    X = meta_data_np

    # metadata-specific matrix P
    P = np.linalg.inv(X.T @ X) @ X.T @ Q_z

    # Convert to tensor
    P = torch.tensor(P).float()

    return P

class CustomDataset(Dataset):
    '''Suitable for args.preload == true, used to store the dataset'''
    def __init__(self, datas, labels,metas):
        """
        tensor_dataset: a TensorDataset instance containing ecgs, locations, ys
        record_names: a list of record identifiers
        """
        self.tensor_dataset = TensorDataset(datas, labels)
        self.record_names = metas

    def __getitem__(self, index):
        data,  y = self.tensor_dataset[index]
        meta = self.record_names[index]
        return data,y,meta

    def __len__(self):
        return len(self.tensor_dataset)
def main():
    # Raw paths for data and labels
    data_dir = 'data/'

    with open(f"config/TriplemetaVAE3.yaml", "r") as f:
        config = yaml.safe_load(f)
    task = config['task']
    latent_dim_conv = config['model']['latent_dim_conv']
    latent_dim_lstm = config['model']['latent_dim_lstm']
    z_dim = config['model']['z_dim']
    merge = config['model']['merge']
    fixed = config['model']['fixed']


    foldername = f'output/{task}'
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    # Use GPU if available, otherwise use CPU.
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', DEVICE)
    torch.manual_seed(42)

    X_Test_Data = np.load(os.path.join(data_dir,task,'Test_data.npy'))
    Y_Test_Data = np.load(os.path.join(data_dir,task,'Test_label.npy'))
    test_meta = np.load(os.path.join(data_dir,task,'Test_meta.npy'))

    X_Test_Data = torch.FloatTensor(X_Test_Data)
    Y_Test_Data = torch.FloatTensor(Y_Test_Data)
    test_meta = torch.FloatTensor(test_meta)
    test_set = CustomDataset(X_Test_Data, Y_Test_Data, test_meta)

    print('test_set', len(test_set))
    batch_size = config['train']['batch_size']
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    X_shape = X_Test_Data.shape

    # Feature dimension of two modalities
    input_dim1 = X_shape[-1] # ECG 624/512


    model = TripleVAE_cmeta3_fft(
        device=DEVICE,
        z_dim =z_dim,
        latent_dim_conv=latent_dim_conv,
        latent_dim_lstm=latent_dim_lstm,
        in_channels=1,
        out_channels=1,
        out_features=input_dim1,
        merge=merge,
        fixed=fixed,
    ).to(DEVICE)

    print('eval best')
    model.eval()

    output_path = f'checkpoints/{task}/model.pth'
    model.load_state_dict(torch.load(output_path,map_location=DEVICE))



    # Reconstruction experiment
    mertics_recons_modal_ecg,ori_modal_ecg,re_modal_ecg = infer_other_modality(task,model, test_loader,DEVICE, known_modality='PPG',foldername=foldername)


    # Save reconstructed data
    print(re_modal_ecg.shape)
    np.save(foldername + '/all_reconstructed_modal_ecg.npy', re_modal_ecg)




def infer_other_modality(task, model, data_loader, device, known_modality='PPG',
                         foldername='', noise_set=None, draw=True):
    """批量重建；返回逐样本平均指标、原始波形和重建波形。"""
    if known_modality not in ('PPG', 'ECG_PPG'):
        raise ValueError("known_modality must be 'PPG' or 'ECG_PPG'.")
    channels = [1] if known_modality == 'PPG' else [0, 1]
    target_idx = 0 if known_modality == 'PPG' else 2
    construct_name = 'ECG' if target_idx == 0 else 'ABP'
    originals, reconstructions = [], []
    model.eval()

    with torch.inference_mode():
        for i, (x, _, meta) in enumerate(data_loader):
            inputs = x[:, channels, :].to(device)
            embeddings = model.get_embeddings(inputs, known_modality, meta.to(device))
            reconstructed = model.decode(embeddings)[target_idx].cpu().numpy()
            original = x[:, [target_idx], :].cpu().numpy()
            if task == 'II_Pleth_ABP' and construct_name == 'ABP':
                reconstructed *= 100  # 训练时 ABP / 100，还原到测试文件中的幅值。
            if original.shape != reconstructed.shape:
                raise ValueError(f'真值 {original.shape} 与重建结果 {reconstructed.shape} 形状不一致。')
            originals.append(original)
            reconstructions.append(reconstructed)

            if draw and i == 0:
                image_dir = os.path.join(foldername, 'rec_img')
                os.makedirs(image_dir, exist_ok=True)
                time = np.arange(original.shape[-1]) * 10 / original.shape[-1]
                for idx in (0, 10, 20):
                    if idx >= len(original):
                        continue
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(time, original[idx, 0], label='Original', color='black')
                    ax.plot(time, reconstructed[idx, 0], label='Reconstructed', color='tab:orange')
                    ax.set(xlabel='Time (s)', ylabel=construct_name)
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(os.path.join(image_dir, f'test_rec_{construct_name}_{idx}.png'), dpi=150)
                    plt.close(fig)

    if not originals:
        raise ValueError('测试集为空，无法进行批量推理。')
    all_original_modal = np.concatenate(originals, axis=0)
    all_reconstructed_modal = np.concatenate(reconstructions, axis=0)
    original = all_original_modal[:, 0, :]
    reconstructed = all_reconstructed_modal[:, 0, :]
    # 拼接后逐样本计算再平均，避免最后一个小 batch 被赋予过大权重。
    metric_functions = {
        'mse': metrics.MSE, 'rmse': metrics.RMSE, 'mad': metrics.MAD,
        'prd': metrics.PRD_formal, 'cos_sim': metrics.COS_SIM,
        'r_square': metrics.R_suqared, 'snr_noise': metrics.SNR,
    }
    metrics_output = {
        name: float(np.mean(function(original, reconstructed)))
        for name, function in metric_functions.items()
    }
    print(metrics_output)
    return metrics_output, all_original_modal, all_reconstructed_modal

def infer_get_embeddings(task,model, data_loader, device,known_modality='ECG',foldername='',P_ecg=None,P_ppg=None,P_abp=None):
    '''
    get embeddings of input modalities
    '''
    model.eval()  # Set the model to evaluation mode
    # Store all embeddings
    z_all = []
    channel = {'ECG': 0, 'PPG': 1, 'ABP': 2}[known_modality]
    with torch.inference_mode():
        for x, _, meta in data_loader:
            inputs = x[:, [channel], :].to(device)
            if task == 'II_Pleth_ABP' and known_modality == 'ABP':
                inputs = inputs / 100  # 与训练时的 ABP 输入尺度一致。
            z = model.get_embeddings(inputs, known_modality, meta.to(device))
            z_all.append(z.cpu().detach().numpy())
    if not z_all:
        raise ValueError('测试集为空，无法提取 embeddings。')
    z_all = np.concatenate(z_all, axis=0)
    print(f'{known_modality} embeddings done!')
    print(z_all.shape)

    np.save(os.path.join(foldername, f'embeddings_{known_modality}.npy'), z_all)
    return z_all


if __name__=='__main__':
    main()
