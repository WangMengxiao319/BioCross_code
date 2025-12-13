'''this is inference(test) code for BioCross-FFT-cross (float 32)'''

import yaml
import os
from model.BioCross import *
from torch.utils.data import DataLoader, TensorDataset,Dataset
import numpy as np
import torch
import utils.metrics as metrics

from utils.utils import ECGplot_one_signal

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
    split_method = config['train']['split_method']
    latent_dim_conv = config['model']['latent_dim_conv']
    latent_dim_lstm = config['model']['latent_dim_lstm']
    z_dim = config['model']['z_dim']
    merge = config['model']['merge']
    encoder_pretrain_path = config['model']['encoder_pretrain_path']
    fixed = config['model']['fixed']
    input_modal = config['input_modal']


    foldername = f'output/{task}'
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    # Check whether NPU/GPU is available; use GPU if available, otherwise use CPU
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', DEVICE)
    # torch.backends.cudnn.benchmark = True


    # Load dataset
    # if input_modal == 'ECG':
    #     X_Test_Data_ecg = np.load(os.path.join(data_dir,task,'all_original_modal_ecg.npy'))
    # elif input_modal == 'PPG':
    #     X_Test_Data_ppg = np.load(os.path.join(data_dir,task,'all_original_modal_ppg.npy'))
    # elif input_modal == 'ABP':
    #     X_Test_Data_abp = np.load(os.path.join(data_dir,task,'all_original_modal_abp.npy'))



    X_Test_Data = np.load(os.path.join(data_dir,task,'Test_data.npy'))
    Y_Test_Data = np.load(os.path.join(data_dir,task,'Test_label.npy'))
    test_meta = np.load(os.path.join(data_dir,task,'Test_meta.npy'))

    X_Test_Data = torch.FloatTensor(X_Test_Data)
    Y_Test_Data = torch.FloatTensor(Y_Test_Data)
    test_meta = torch.FloatTensor(test_meta)
    test_set = CustomDataset(X_Test_Data, Y_Test_Data, test_meta)

    print('test_set', len(test_set))
    batch_size = 128
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    X_shape = X_Test_Data.shape
    Y_shape = Y_Test_Data.shape

    print('Check Y distribution in test set', np.unique(Y_Test_Data, return_counts=True))

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

    model.to(DEVICE)

    # print('eval final')
    # evaluate(task,model, val_loader, 1, DEVICE, foldername=foldername)

    print('eval best')
    model.eval()

    output_path = foldername + "/model.pth"
    model.load_state_dict(torch.load(output_path,map_location=DEVICE))



    # Reconstruction experiment
    mertics_recons_modal_ecg,ori_modal_ecg,re_modal_ecg = infer_other_modality(task,model, test_loader,DEVICE, known_modality='PPG',foldername=foldername)


    # Save reconstructed data
    print(re_modal_ecg.shape)
    np.save(foldername + '/all_reconstructed_modal_ecg.npy', re_modal_ecg)




# Define inference function
def infer_other_modality(task,model, data_loader, device,known_modality='ECG',foldername='',noise_set = None,draw=True):
    """
    Use one modality to infer the other modality.

    :param dualvae_model: Trained DualVAE model.
    :param data_loader: DataLoader providing input batches.
    :param known_modality: Specify which modality is known ('input1' or 'input2').
    :return: List of reconstructed data for the other modality.
    """
    model.eval()  # Set the model to evaluation mode
    reconstructed_modality = []

    mse_total1 = 0
    rmse_total1 = 0
    mad_total1 = 0
    prd_total1 = 0
    cos_sim_total1 = 0
    snr_noise = 0
    r_square_total1 = 0
    mse_total2 = 0
    rmse_total2 = 0
    mad_total2 = 0
    prd_total2 = 0
    cos_sim_total2 = 0
    r_square_total2 = 0

    ssd_total = 0
    mse_total = 0
    rmse_total = 0
    mad_total = 0
    prd_total = 0
    cos_sim_total = 0
    snr_noise = 0
    r_square_total = 0

    snr_ecg = 0
    snr_ppg = 0
    snr_abp = 0

    print('len_data_loader:',len(data_loader))

    with torch.no_grad():  # Disable gradient computation
        all_original_modal = np.empty((0, 1, 512))
        all_reconstructed_modal = np.empty((0, 1, 512))

        for (i,data) in enumerate(data_loader):
            if task == 'II_Pleth_ABP':
                x, y ,meta= data
            elif task == 'ppg_abp_ecg_mimic_ii':
                x = data

            data1 = x[:, [0], :]
            data2 = x[:, [1], :]
            data3 = x[:, [2], :]
            input1, input2, input3 = data1.to(device), data2.to(device), data3.to(device)
            meta = meta.to(device)

            if not os.path.exists(foldername + '/rec_img/'):
                os.makedirs(foldername + '/rec_img/')


            if known_modality == 'PPG':
                construct_name = 'ECG'
                input2 = input2.to(device)
                embeddings = model.get_embeddings(input2,known_modality,meta)
                output_ecg,_,_ = model.decode(embeddings)

                ori_modal1 = input1.cpu().detach().numpy()
                re_modal1 = output_ecg.cpu().detach().numpy()

                # Per batch, plot figures
                if i == 0:
                    for idx in [0, 10, 20]:
                        ori_ecg = ori_modal1[idx]
                        re_ecg = re_modal1[idx]
                        plt_rec = ECGplot_one_signal(ori_ecg, re_ecg, save=False, fs=ori_ecg.shape[-1]/10, epoch=i,name=['ECG'])
                        plt_rec.savefig(foldername + f'/rec_img/test_rec_{construct_name}_{idx}.png')
                        # r_squre = metrics.R_suqared(ori_signal, re_signal)
                        # print(f"R^2: {r_squre}")

            elif known_modality == 'ECG_PPG':
                construct_name = 'ABP'
                if draw:
                    if i == 0:
                        raw_ecg = input1.cpu().detach().numpy()
                        raw_ppg = input2.cpu().detach().numpy()
                        raw_abp = input3.cpu().detach().numpy()
                input1 = input1.to(device)
                input2 = input2.to(device)

                input = torch.cat((input1,input2),dim=1)
                embeddings = model.get_embeddings(input,known_modality,meta)
                _,_,output_abp = model.decode(embeddings)
                ori_modal1 = input3.cpu().detach().numpy()
                re_modal1 = output_abp.cpu().detach().numpy()
                # Per batch, plot figures
                if i == 0:
                    for idx in [0, 10, 20]:
                        ori_abp = ori_modal1[idx]
                        re_abp = re_modal1[idx]
                        plt_rec = ECGplot_one_signal(ori_abp,re_abp, save=False, fs=ori_modal1.shape[-1]/10, epoch=i,name=['ABP'])
                        plt_rec.savefig(foldername + f'/rec_img/test_rec_{construct_name}_{idx}.png')
                        # r_squre = metrics.R_suqared(ori_signal, re_signal)
                        # print(f"R^2: {r_squre}")



            else:
                raise ValueError("known_modality must be 'input1' or 'input2'.")

            ### Compute proxy task metrics
            # Dimension 0 of original_modal is batch size, dimension 1 is signal length
            all_original_modal = np.concatenate((all_original_modal, ori_modal1), axis=0)
            all_reconstructed_modal = np.concatenate((all_reconstructed_modal, re_modal1), axis=0)
            origin_modal = np.squeeze(ori_modal1, axis=1)  # (batchsize, signal_length)
            reconstructed_modal_squeeze = np.squeeze(re_modal1, axis=1)  # (batchsize, signal_length)

            mse_total += np.mean(metrics.MSE(origin_modal, reconstructed_modal_squeeze))
            rmse_total += np.mean(metrics.RMSE(origin_modal, reconstructed_modal_squeeze))
            mad_total += np.mean(metrics.MAD(origin_modal, reconstructed_modal_squeeze))
            prd_total += np.mean(metrics.PRD(origin_modal, reconstructed_modal_squeeze))
            cos_sim_total += np.mean(metrics.COS_SIM(origin_modal, reconstructed_modal_squeeze))
            r_square_total += np.mean(metrics.R_suqared(origin_modal, reconstructed_modal_squeeze))
            snr_noise += np.mean(metrics.SNR(origin_modal, reconstructed_modal_squeeze))

        # Return metrics per batch, per sample, per point
        metrics_output = {
            "mse": mse_total / len(data_loader),
            "rmse": rmse_total / len(data_loader),
            "mad": mad_total / len(data_loader),
            "prd": prd_total / len(data_loader),
            "cos_sim": cos_sim_total / len(data_loader),
            "r_square": r_square_total / len(data_loader),
            "snr_noise": snr_noise / len(data_loader),
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
    with torch.no_grad():  # Disable gradient computation

        for (i,data) in enumerate(data_loader):
            if task == 'II_Pleth_ABP':
                x, y, meta = data
            elif task == 'ppg_abp_ecg_mimic_ii':
                x = data
            # # Old dataset
            # input1, input2 = x.to(device), y.to(device)
            # TODO: new dataset
            data1 = x[:, [0], :]
            data2 = x[:, [1], :]
            data3 = x[:, [2], :]
            input1, input2,input3 = data1.to(device), data2.to(device),data3.to(device)
            meta = meta.to(device)
            if known_modality == 'ECG':
                z = model.get_embeddings(input1,known_modality,meta)
            elif known_modality == 'PPG':
                z = model.get_embeddings(input2,known_modality,meta)
            elif known_modality == 'ABP':
                z = model.get_embeddings(input3,known_modality,meta)
            z_all.append(z.cpu().detach().numpy())
    z_all = np.concatenate(z_all, axis=0)
    print(f'{known_modality} embeddings done!')
    print(z_all.shape)

    np.save(foldername + f'/embeddings_{known_modality}.npy', z_all)
    return z_all


if __name__=='__main__':
    main()
