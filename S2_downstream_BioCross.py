# -*- coding: utf-8 -*-
# @Time : 2024/8/1 21:19
# @Author : 王梦潇
# @File : S2_downstream_TripleVAE.py
# Function:  S1阶段训练好的TripleVAE模型，进行下游任务的训练
#目前暂时选择四个任务：
'''
heart_failure
hypertensive
diabetes
kidney_failure
'''
import yaml
import os
from Dt import load_data_fold, load_data_fold_meta
from model.Triple_downstream import TripleVAE_downstream
import numpy as np
import torch
from torch.optim import Adam,AdamW
from tqdm import tqdm
import pickle
import utils.metrics as metrics
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import roc_auc_score
import random
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def main(fold=0):
    seed_torch(seed=666)
    # 数据、标签原始路径
    data_dir = 'data/'


    with open(f"config/BioCross_downstream.yaml", "r") as f:
        config = yaml.safe_load(f)
    task = config['task']
    down_task = config['down_task']
    model_name = config['train']['encoder']
    split_method = config['train']['split_method']  # 这个是数据集的划分
    foldername = f'output/{task}/{split_method}/{down_task}/{model_name}/{config["input_modal"]}/{config["model"]["fixed"]}'
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    # 检测CUDA是否可用，如果可用，使用GPU，否则使用CPU
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', DEVICE)
    print('input_modal:', config['input_modal'])
    torch.backends.cudnn.benchmark = True
    # 数据读取
    debug = config['debug']
    resample = config['resample']
    train_loader, val_loader, test_loader, X_shape, Y_shape,train_meta = load_data_fold_meta(data_dir=data_dir, task=task, downtask=down_task,
                                                                        batch_size=config['train']['batch_size'],
                                                                        debug=debug,
                                                                        resample=resample,
                                                                        target_fold=fold,
                                                                        input_modal=config['input_modal'],
                                                                        split_method=config['train']['split_method'])

    # print("！分隔符01！")
    if debug:
        notes = 'debug'
    elif config['phase'] == 'test':
        notes = 'test'
    else:
        notes=''

    input_modal = config['input_modal']
    latent_dim_conv = config['model']['latent_dim_conv']
    latent_dim_lstm = config['model']['latent_dim_lstm']
    z_dim = config['model']['z_dim']
    merge = config['model']['merge']
    dropout = config['model']['dropout']
    fixed = config['model']['fixed']
    input_dim1 = X_shape[-1]  # ECG 624
    downstream_cls_num = Y_shape[1]
    experiment = wandb.init(project='cross_modal_re', resume='allow', anonymous='must', config=config,notes=notes)
    model = TripleVAE_downstream(encoder=config['train']['encoder'], downstream_cls_num=downstream_cls_num, known_modality=input_modal,
                               device=DEVICE, z_dim=z_dim, latent_dim_conv=latent_dim_conv, latent_dim_lstm=latent_dim_lstm, in_channels=1, out_channels=1, out_features=input_dim1, merge=merge, dropout=dropout, fixed=fixed)
    model = model.to(DEVICE)
    experiment.config.update({'model_name': model_name})
    # 训练
    if config['phase'] == 'train':
        wandb.log({"fold": fold})
        train(model, config['train'], train_loader, DEVICE,downstream_cls_num,
              valid_loader=val_loader, valid_epoch_interval=1, foldername=foldername)

    # 评估
    print('eval best')
    if config['eval_model'] == 'best':
        output_path = foldername + "/model.pth"
    else:
        output_path = foldername + "/final.pth"
    model.load_state_dict(torch.load(output_path, map_location=DEVICE))

    evaluate(model, config['train'], test_loader,  DEVICE, downstream_cls_num, foldername=foldername)



def train(model, config, train_loader, device, num_classes,valid_loader=None, valid_epoch_interval=5, foldername=""):
    if config['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=config["lr"])  #TODO: AdamW
    elif config['optimizer'] == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=config["lr"])

    if config['criterion'] == 'BCEWithLogitsLoss':
        criterion = torch.nn.BCEWithLogitsLoss()

    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"


    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=.5, verbose=True
    )

    best_valid_score = 0


    # 训练
    for epoch_no in range(config["epochs"]):
        model.train()
        train_loss = 0
        with tqdm(train_loader) as trn_dataloader:
            for batch_idx, data in enumerate(trn_dataloader, start=1):
                optimizer.zero_grad()
                x, y, meta = data
                x, y, meta = x.to(device), y.to(device), meta.to(device)
                if 'meta' in config['encoder']:
                    output = model(x, meta)
                else:
                    output = model(x)

                loss = criterion(output, y)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

                ordered_dict = {
                        "train_avg_epoch_loss": train_loss / batch_idx ,
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
            with torch.no_grad():
                output_all = np.empty((0, num_classes))
                target_all = np.empty((0, num_classes))
                with tqdm(valid_loader) as val_dataloader:
                    for batch_no, data in enumerate(val_dataloader, start=1):
                        x, y, meta = data
                        x, y, meta = x.to(device), y.to(device), meta.to(device)
                        if 'meta' in config['encoder']:
                            output = model(x, meta)
                        else:
                            output = model(x)
                        loss = criterion(output, y)
                        avg_loss_valid += loss.item()

                        pred = torch.sigmoid(output)
                        # 连接到所有的输出
                        output_all = np.concatenate((output_all, pred.cpu().numpy()))
                        target_all = np.concatenate((target_all, y.cpu().numpy()))

                        ordered_dict = {
                            "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                            "epoch": epoch_no,
                        }
                        val_dataloader.set_postfix(
                            ordered_dict=ordered_dict,
                            refresh=True,
                        )
            wandb.log(ordered_dict)
            val_auroc = roc_auc_score(target_all, output_all)


            judge_score = val_auroc
            if best_valid_score < judge_score:
                best_valid_score = judge_score
                if foldername != "":
                    torch.save(model.state_dict(), output_path)
                    print(f"Save model to {output_path}")
                wandb.log({"best_val_AUROC": best_valid_score})
            # 计算auroc
            print(f"Epoch {epoch_no} val_AUROC: {val_auroc}")
            wandb.log({"val_AUROC": val_auroc})

    torch.save(model.state_dict(), final_path)

def evaluate(model, config,test_loader, device, num_classes, foldername=""):
    model.eval()

    avg_loss_test= 0
    with torch.no_grad():
        output_all = np.empty((0, num_classes))
        target_all = np.empty((0, num_classes))
        embeddings_all = np.empty((0, 128)) # z_dim=128
        with tqdm(test_loader) as tst_dataloader:
            for batch_no, data in enumerate(tst_dataloader, start=1):
                x, y, meta = data
                x, y, meta = x.to(device), y.to(device), meta.to(device)
                if 'meta' in config['encoder']:
                    output = model(x, meta)
                    embeddings = model.get_embeddings(x, meta)
                else:
                    output = model(x)
                    embeddings = model.get_embeddings(x)
                pred = torch.sigmoid(output)
                # 连接到所有的输出
                output_all = np.concatenate((output_all, pred.cpu().numpy()))
                target_all = np.concatenate((target_all, y.cpu().numpy()))
                embeddings_all = np.concatenate((embeddings_all, embeddings.cpu().numpy()))


        # 计算auroc
        test_auroc = roc_auc_score(target_all, output_all)
        print(f"Test_AUROC: {test_auroc}")
        wandb.log({"test_AUROC": test_auroc})

        # 向txt中写入结果
        with open(foldername + "/result_AUROC.txt", "a") as f:
            f.write(f"{test_auroc}\n")

        # 保存结果
        np.save(foldername + "/output_all.npy", output_all)
        np.save(foldername + "/target_all.npy", target_all)
        np.save(foldername + "/embeddings_all.npy", embeddings_all)

if __name__=='__main__':
    # for fold in range(10):
    #     print(f"fold:{fold}")
    #     main(fold=fold)
    main(fold=2)