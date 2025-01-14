'''this is BioCross-FFT-cross'''

import yaml
import os
from Dt import load_data,load_data_fold,load_three_modalities,load_data_fold_meta
from model.BioCross import *
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
import numpy as np
import torch
from torch.optim import Adam,AdamW
from tqdm import tqdm
import pickle
import utils.metrics as metrics
import matplotlib.pyplot as plt
import wandb
from utils.utils import ECGplot_one_signal,ECGplot_two_signals,ECGplot_three_signals,adjust_learning_rate

def meta_embedding(Q_z, meta_data):
    '''
    :param input: Q_z: latent space (numpy)
           meta_data: list-->object eg.{'time': 1, 'age': 51.0, 'gender': 0.0}
    :return:
    '''
    # meta_data的tensor转换为numpy
    meta_data_np = np.array(meta_data)
    X = meta_data_np

    # metadata-specific matrix P
    P = np.linalg.inv(X.T @ X) @ X.T @ Q_z

    # 转换成tensor
    P = torch.tensor(P).float()

    #
    # # Pm: metadata-specific latent space
    # Pm = X @ P
    # meta_embedding = Q_z - Pm

    return P
def main():
    # 数据、标签原始路径
    data_dir = 'data/'

    model_name = 'TriplemetaVAE3_fft_multihead2'

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


    model_name = f'{model_name}_{split_method}'
    if encoder_pretrain_path == '':
        model_name = f'{model_name}_no_pretrain'
    if config['filtered']:
        model_name = f'{model_name}_filtered'

    foldername = f'output/{task}/{model_name}'
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    # 检测CUDA是否可用，如果可用，使用GPU，否则使用CPU
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', DEVICE)
    torch.backends.cudnn.benchmark = True
    # 数据读取
    debug = config['debug']
    resample = config['resample']

    if task == 'II_Pleth_ABP':
        train_loader, val_loader, test_loader, X_shape, Y_shape,train_meta = load_data_fold_meta(data_dir=data_dir, task=task,
                                                                            batch_size=config['train']['batch_size'],
                                                                            debug=debug,
                                                                            resample=resample,
                                                                            target_fold=2,
                                                                            train_sample=config['train']['train_sample'],
                                                                            split_method=config['train']['split_method'],
                                                                            filtered=config['filtered']
                                                                            )
    elif task == 'ppg_abp_ecg_mimic_ii':
        train_loader, val_loader, test_loader, X_shape= load_three_modalities(data_dir=data_dir,
                                                                                        task=task,
                                                                                        batch_size=config['train']['batch_size'],
                                                                                        debug=debug,
                                                                                        resample=resample)


    # wandb.save("/home/lzy/mancy/cross_modal_re/model/DualVAE0.py")

    # 两个模态的特征维度
    input_dim1 = X_shape[-1] # ECG 624/512



    model = TripleVAE_cmeta3_fft(
        device=DEVICE,
        pretrain_model_path=encoder_pretrain_path,
        z_dim =z_dim,
        latent_dim_conv=latent_dim_conv,
        latent_dim_lstm=latent_dim_lstm,
        in_channels=1,
        out_channels=1,
        out_features=input_dim1,
        merge=merge,
        fixed=fixed
    ).to(DEVICE)

    model.to(DEVICE)
    model.eval()

    ### 计算meta matrix

    # ecg_embedding = np.empty((0, z_dim))
    # ppg_embedding = np.empty((0, z_dim))
    # abp_embedding = np.empty((0, z_dim))
    #
    # for (i, data) in tqdm(enumerate(train_loader), total=len(train_loader)):
    #     signals, label, meta = data
    #     signals = signals.to(DEVICE)
    #     ecg = signals[:, [0], :]
    #     ppg = signals[:, [1], :]
    #     abp = signals[:, [2], :]
    #
    #     z1 = model.pretrain_model.get_embeddings(ecg, known_modality='ECG')
    #     z2 = model.pretrain_model.get_embeddings(ppg, known_modality='PPG')
    #     z3 = model.pretrain_model.get_embeddings(abp, known_modality='ABP')
    #
    #     ecg_embedding = np.concatenate((ecg_embedding, z1.cpu().detach().numpy()), axis=0)
    #     ppg_embedding = np.concatenate((ppg_embedding, z2.cpu().detach().numpy()), axis=0)
    #     abp_embedding = np.concatenate((abp_embedding, z3.cpu().detach().numpy()), axis=0)
    #
    # P_ecg = meta_embedding(ecg_embedding, train_meta).to(DEVICE)
    # P_ppg = meta_embedding(ppg_embedding, train_meta).to(DEVICE)
    # P_abp = meta_embedding(abp_embedding, train_meta).to(DEVICE)
    #
    # print('P_ecg:', P_ecg.shape)
    # print('P_ppg:', P_ppg.shape)
    # print('P_abp:', P_abp.shape)

    if debug:
        notes = 'debug'
    elif config['phase'] == 'test':
        notes = 'test'
    else:
        notes=''
    experiment = wandb.init(project='cross_modal_re', resume='allow', anonymous='must', config=config,notes=notes)
    experiment.config.update({'model_name': model_name})

    if config['phase'] == 'train':
        train(task,model, config['train'], train_loader, DEVICE,
              valid_loader=val_loader, valid_epoch_interval=1, foldername=foldername)

    # print('eval final')
    # evaluate(task,model, val_loader, 1, DEVICE, foldername=foldername)

    print('eval best')
    output_path = foldername + "/model.pth"
    model.load_state_dict(torch.load(output_path,map_location=DEVICE))

    mertics_recons_modal_ecg,ori_modal_ecg,re_modal_ecg = infer_other_modality(task,model, test_loader,DEVICE, known_modality='PPG',foldername=foldername)
    mertics_recons_modal_abp,ori_modal1_abp,re_modal_abp = infer_other_modality(task,model, test_loader,DEVICE, known_modality='ECG_PPG',foldername=foldername)

    wandb.log({'mertics_recons_modal_ecg':mertics_recons_modal_ecg})
    wandb.log({'mertics_recons_modal_abp':mertics_recons_modal_abp})

    # 保存原始数据
    print(ori_modal_ecg.shape)
    np.save(foldername + '/all_original_modal_ecg.npy', ori_modal_ecg)
    np.save(foldername + '/all_original_modal_abp.npy', ori_modal1_abp)

    # # 保存重建的数据
    print(re_modal_ecg.shape)
    np.save(foldername + '/all_reconstructed_modal_ecg.npy', re_modal_ecg)
    np.save(foldername + '/all_reconstructed_modal_abp.npy', re_modal_abp)

    infer_get_embeddings(task,model, test_loader, DEVICE, known_modality='ECG', foldername=foldername)
    infer_get_embeddings(task,model, test_loader, DEVICE, known_modality='PPG', foldername=foldername)
    infer_get_embeddings(task,model, test_loader, DEVICE, known_modality='ABP', foldername=foldername)


    # don't use before final model is determined
    # print('eval final')
    # output_path = foldername + "/final.pth"
    # model.load_state_dict(torch.load(output_path))
    # evaluate(model, test_loader, 1, DEVICE, foldername=foldername)

def train(task,model, config, train_loader, device, valid_loader=None, valid_epoch_interval=5, foldername=""):
    if config['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=config["lr"])  #TODO: AdamW
    elif config['optimizer'] == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=config["lr"])

    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"


    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=.5, verbose=True
    )

    best_valid_loss = 1e10
    best_valid_all_loss = 1e10
    best_valid_re_loss = 1e10
    best_valid_kl_loss = 1e10

    # 训练
    for epoch_no in range(config["epochs"]):
        adjust_learning_rate(optimizer, epoch_no, config["warmup_epochs"],config["lr"],config["min_lr"],config["epochs"])
        model.train()
        train_loss = 0
        re_loss = 0
        kl_loss = 0
        con_loss = 0
        with tqdm(train_loader) as trn_dataloader:

            for batch_idx, data in enumerate(trn_dataloader, start=1):
                if task == 'II_Pleth_ABP':
                    x, y, meta = data
                elif task == 'ppg_abp_ecg_mimic_ii':
                    x = data
                # TODO:新的数据集
                data1 = x[:,[0],:]
                data2 = x[:,[1],:]
                data3 = x[:,[2],:]
                data1, data2, data3 = data1.to(device), data2.to(device), data3.to(device)
                meta = meta.to(device)

                optimizer.zero_grad()
                output1,output2,output3,input1,input2,input3,mu1, log_var1, mu2, log_var2,mu3, logvar3 = model(data1, data2, data3,meta)

                batch_size = data1.shape[0]
                loss = model.loss_function(output1,output2,output3,input1,input2,input3,mu1, log_var1, mu2, log_var2,mu3, logvar3,batch_size=batch_size)
                loss['loss'].backward()
                train_loss += loss['loss'].item()
                re_loss += loss["Reconstruction_Loss"].item()
                kl_loss += loss["KLD"].item()
                # con_loss += loss["Contrastive_Loss"].item()
                optimizer.step()

                ordered_dict = {
                        "train_avg_epoch_loss": train_loss / batch_idx/data1.shape[0]/data1.shape[-1] ,
                        "train_re_loss": re_loss / batch_idx/data1.shape[0]/data1.shape[-1],
                        "train_kl_loss": kl_loss / batch_idx/data1.shape[0]/data1.shape[-1],
                        # "train_con_loss": con_loss / batch_idx/data1.shape[0]/data1.shape[-1],
                        "epoch": epoch_no,
                    }
                trn_dataloader.set_postfix(
                    ordered_dict=ordered_dict,
                    refresh=True,
                )
                wandb.log(ordered_dict)
        # lr_scheduler.step()


        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            re_loss_valid = 0
            kl_loss_valid = 0
            con_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader) as val_dataloader:
                    for batch_no, data in enumerate(val_dataloader, start=1):
                        if task == 'II_Pleth_ABP':
                            x, y, meta = data
                        elif task == 'ppg_abp_ecg_mimic_ii':
                            x = data

                        data1 = x[:, [0], :]
                        data2 = x[:, [1], :]
                        data3 = x[:, [2], :]
                        data1, data2, data3 = data1.to(device), data2.to(device), data3.to(device)
                        meta = meta.to(device)
                        batch_size = data1.shape[0]
                        output1, output2, output3, input1, input2, input3,mu1, log_var1, mu2, log_var2,mu3, logvar3 = model(
                            data1, data2, data3,meta)
                        loss = model.loss_function(output1, output2, output3, input1, input2, input3, mu1, log_var1, mu2, log_var2,mu3, logvar3,batch_size=batch_size)

                        avg_loss_valid += loss["loss"].item()
                        re_loss_valid += loss["Reconstruction_Loss"].item()
                        kl_loss_valid += loss["KLD"].item()
                        # con_loss_valid += loss["Contrastive_Loss"].item()

                        ordered_dict = {
                            "valid_avg_epoch_loss": avg_loss_valid / batch_no/data1.shape[0]/data1.shape[-1],
                            "val_re_loss": re_loss_valid / batch_no/data1.shape[0]/data1.shape[-1],
                            "val_kl_loss": kl_loss_valid / batch_no/data1.shape[0]/data1.shape[-1],
                            # "val_con_loss": con_loss_valid / batch_no/data1.shape[0]/data1.shape[-1],
                            "epoch": epoch_no,
                        }
                        val_dataloader.set_postfix(
                            ordered_dict=ordered_dict,
                            refresh=True,
                        )
            wandb.log(ordered_dict)

            judge_loss = re_loss_valid / batch_no/data1.shape[0]/data1.shape[-1]
            if best_valid_loss > judge_loss:
                best_valid_loss = judge_loss
                best_valid_all_loss = avg_loss_valid / batch_no / data1.shape[0] / data1.shape[-1]
                best_valid_re_loss = re_loss_valid / batch_no / data1.shape[0] / data1.shape[-1]
                # best_valid_kl_loss = kl_loss_valid / batch_no / data1.shape[0] / data1.shape[-1]
                print("\n best loss is updated to ", best_valid_loss, "at", epoch_no, )

                if foldername != "":
                    torch.save(model.state_dict(), output_path)

                # 绘图
                output_ecg = output1.cpu().detach().numpy()
                output_ppg = output2.cpu().detach().numpy()
                output_abp = output3.cpu().detach().numpy()
                data_ecg = data1.cpu().detach().numpy()
                data_ppg = data2.cpu().detach().numpy()
                data_abp = data3.cpu().detach().numpy()
                # print("在这停顿！")
                for (i,List) in enumerate([0,10,20]):
                    re_ecg = output_ecg[List]
                    re_ppg = output_ppg[List]
                    re_abp = output_abp[List]
                    ori_ecg = data_ecg[List]
                    ori_ppg = data_ppg[List]
                    ori_abp = data_abp[List]

                    plt1 = ECGplot_three_signals(ori_ecg, ori_ppg,ori_abp, re_ecg, re_ppg,re_abp, fs=ori_ecg.shape[-1]/10, epoch=epoch_no,name=['ECG','PPG','ABP'])
                    plt1.savefig(foldername + f'/val_plt{i}.png')
                    plt1.close()
                    wandb.log({f'val_plt{i}': wandb.Image(foldername + f'/val_plt{i}.png')})


            wandb.log({"best_valid_loss": best_valid_loss})
            wandb.log({"best_valid_all_loss": best_valid_all_loss})
            wandb.log({"best_valid_re_loss": best_valid_re_loss})
            wandb.log({"best_valid_kl_loss": best_valid_kl_loss})
    torch.save(model.state_dict(), final_path)

def evaluate(task,model, test_loader, shots, device, foldername=""):
    ssd_total = 0
    mad_total = 0
    prd_total = 0
    cos_sim_total = 0
    snr_noise = 0
    snr_recon = 0
    snr_improvement = 0
    eval_points = 0

    restored_sig = []
    with tqdm(test_loader) as it:
        for batch_no, data in enumerate(it, start=1):
            if task == 'II_Pleth_ABP':
                x, y = data
            elif task == 'ppg_abp_ecg_mimic_ii':
                x = data
            data1 = x[:, [0], :]
            data2 = x[:, [1], :]
            data1, data2 = data1.to(device), data2.to(device)
            output1,output2,input1,input2, mu1, log_var1, mu2, log_var2  = model(data1, data2)
            output_ecg = output1.cpu().detach().numpy()
            output_ppg = output2.cpu().detach().numpy()
            data_ecg = data1.cpu().detach().numpy()
            data_ppg = data2.cpu().detach().numpy()
            # 画一下效果
            if batch_no == 1:
                # print("在这停顿！")
                List = 0
                re_ecg = output_ecg[List]
                re_ppg = output_ppg[List]
                ori_ecg = data_ecg[List]
                ori_ppg = data_ppg[List]

                plt_test = ECGplot_two_signals(ori_ecg, ori_ppg, re_ecg, re_ppg, save=False, fs=62.4)
                plt_test.savefig(foldername + '/test_plt0.png')
                wandb.log({"test_plt0": wandb.Image(foldername + '/test_plt0.png')})

                List = 10
                re_ecg = output_ecg[List]
                re_ppg = output_ppg[List]
                ori_ecg = data_ecg[List]
                ori_ppg = data_ppg[List]

                plt_test = ECGplot_two_signals(ori_ecg, ori_ppg, re_ecg, re_ppg, save=False, fs=62.4)
                plt_test.savefig(foldername + '/test_plt1.png')
                wandb.log({"test_plt1": wandb.Image(foldername + '/test_plt1.png')})

                List = 20
                re_ecg = output_ecg[List]
                re_ppg = output_ppg[List]
                ori_ecg = data_ecg[List]
                ori_ppg = data_ppg[List]

                plt_test = ECGplot_two_signals(ori_ecg, ori_ppg, re_ecg, re_ppg, save=False, fs=62.4)
                plt_test.savefig(foldername + '/test_plt2.png')
                wandb.log({"test_plt2": wandb.Image(foldername + '/test_plt2.png')})

            eval_points += len(output_ecg) # Batchsize
            ssd_total += np.sum(metrics.SSD(data_ecg, output_ecg)) + np.sum(metrics.SSD(output_ppg, data_ppg))
            mad_total += np.sum(metrics.MAD(data_ecg, output_ecg)) + np.sum(metrics.MAD(output_ppg, data_ppg))
            prd_total += np.sum(metrics.PRD(data_ecg, output_ecg)) + np.sum(metrics.PRD(output_ppg, data_ppg))
            # cos_sim_total += np.sum(metrics.COS_SIM(clean_numpy, out_numpy))
            # snr_noise += np.sum(metrics.SNR(clean_numpy, noisy_numpy))
            snr_recon += np.sum(metrics.SNR(data_ecg, output_ecg)) + np.sum(metrics.SNR(output_ppg, data_ppg))
            # snr_improvement += np.sum(metrics.SNR_improvement(noisy_numpy, out_numpy, clean_numpy))

            ordered_dict = {
                "ssd_total": ssd_total / eval_points,
                "mad_total": mad_total / eval_points,
                "prd_total": prd_total / eval_points,
                # "cos_sim_total": cos_sim_total / eval_points,
                # "snr_in": snr_noise / eval_points,
                "snr_out": snr_recon / eval_points,
                # "snr_improve": snr_improvement / eval_points,
            }
            it.set_postfix(
                ordered_dict=ordered_dict,
                refresh=True,
            )

        # np.save(foldername + '/denoised.npy', restored_sig)

        print("ssd_total: ", ssd_total / eval_points)
        print("mad_total: ", mad_total / eval_points, )
        print("prd_total: ", prd_total / eval_points, )
        # print("cos_sim_total: ", cos_sim_total / eval_points, )
        # print("snr_in: ", snr_noise / eval_points, )
        print("snr_out: ", snr_recon / eval_points, )
        # print("snr_improve: ", snr_improvement / eval_points, )

        wandb.log(ordered_dict)


# 定义推断函数
def infer_other_modality(task,model, data_loader, device,known_modality='ECG',foldername='',P_ecg=None,P_ppg=None,P_abp=None):
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
                embeddings = model.get_embeddings(input2,known_modality,meta)
                output_ecg,_,_ = model.decode(embeddings)

                ori_modal1 = input1.cpu().detach().numpy()
                re_modal1 = output_ecg.cpu().detach().numpy()

                # 每个batch、画图
                if i == 0:
                    for idx in [0, 10, 20]:
                        ori_ecg = ori_modal1[idx]
                        re_ecg = re_modal1[idx]
                        plt_rec = ECGplot_one_signal(ori_ecg, re_ecg, save=False, fs=ori_ecg.shape[-1]/10, epoch=i,name=['ECG'])
                        plt_rec.savefig(foldername + f'/rec_img/test_rec_{construct_name}_{idx}.png')
                        # r_squre = metrics.R_suqared(ori_signal, re_signal)
                        # print(f"R^2: {r_squre}")
                        wandb.log({f'test_rec_{construct_name}_{idx}': wandb.Image(foldername + f'/rec_img/test_rec_{construct_name}_{idx}.png')})
            elif known_modality == 'ECG_PPG':
                construct_name = 'ABP'
                input = torch.cat((input1,input2),dim=1)
                embeddings = model.get_embeddings(input,known_modality,meta)
                _,_,output_abp = model.decode(embeddings)
                ori_modal1 = input3.cpu().detach().numpy()
                re_modal1 = output_abp.cpu().detach().numpy()
                # 每个batch、画图
                if i == 0:
                    for idx in [0, 10, 20]:
                        ori_abp = ori_modal1[idx]
                        re_abp = re_modal1[idx]
                        plt_rec = ECGplot_one_signal(ori_abp,re_abp, save=False, fs=ori_modal1.shape[-1]/10, epoch=i,name=['ABP'])
                        plt_rec.savefig(foldername + f'/rec_img/test_rec_{construct_name}_{idx}.png')
                        # r_squre = metrics.R_suqared(ori_signal, re_signal)
                        # print(f"R^2: {r_squre}")
                        wandb.log({f'test_rec_{construct_name}_{idx}': wandb.Image(foldername + f'/rec_img/test_rec_{construct_name}_{idx}.png')})
            else:
                raise ValueError("known_modality must be 'input1' or 'input2'.")

            ### 计算代理任务的指标
            # original_modal 维度0是batchsize，维度1是信号长度
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
            # snr_noise += np.mean(metrics.SNR(origin_modal, reconstructed_modal))

        # 返回per batch, per sample, per point的指标
        metrics_output = {
            "mse": mse_total / len(data_loader),
            "rmse": rmse_total / len(data_loader),
            "mad": mad_total / len(data_loader),
            "prd": prd_total / len(data_loader),
            "cos_sim": cos_sim_total / len(data_loader),
            "r_square": r_square_total / len(data_loader),
            # "snr_noise": snr_noise / len(data_loader),
        }

        print(metrics_output)

        return metrics_output, all_original_modal, all_reconstructed_modal

def infer_get_embeddings(task,model, data_loader, device,known_modality='ECG',foldername='',P_ecg=None,P_ppg=None,P_abp=None):
    '''
    get embeddings of input modalities
    '''
    model.eval()  # Set the model to evaluation mode
    # 存储所有的embedding
    z_all = []
    with torch.no_grad():  # Disable gradient computation

        for (i,data) in enumerate(data_loader):
            if task == 'II_Pleth_ABP':
                x, y, meta = data
            elif task == 'ppg_abp_ecg_mimic_ii':
                x = data
            # # 旧的数据集
            # input1, input2 = x.to(device), y.to(device)
            # TODO:新的数据集
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