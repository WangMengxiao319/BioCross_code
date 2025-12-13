'''this is inference(test) code for BioCross-FFT-cross--downstreamS'''

import yaml
import os
from Dt import load_data_fold, load_data_fold_meta
from model.Triple_downstream import TripleVAE_downstream
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset,Dataset
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import random
from utils.metrics_cls import evaluate_experiment
import pandas as pd
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # Disable hash randomization to make experiments reproducible
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
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
def main(fold=0):
    seed_torch(seed=666)
    # Raw paths for data and labels
    data_dir = 'data/'


    with open(f"config/BioCross_downstream.yaml", "r") as f:
        config = yaml.safe_load(f)
    task = config['task']
    down_task = config['down_task']
    input_modal = config['input_modal']
    foldername = f'output/{task}/{down_task}'
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    # Check whether GPU is available; use GPU if available, otherwise use CPU
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', DEVICE)
    print('input_modal:', config['input_modal'])
    torch.backends.cudnn.benchmark = True

    X_Test_Data = np.load(os.path.join(data_dir, task, 'Test_data.npy'))
    Y_Test_Data = np.load(os.path.join(data_dir, task, 'Test_label.npy'))

    if input_modal== 'ECG':
        X_Test_Data = X_Test_Data[:, [0], :]
    elif input_modal == 'PPG':
        X_Test_Data = X_Test_Data[:, [1], :]
    elif input_modal == 'ABP':
        X_Test_Data = X_Test_Data[:, [2], :]


    downtask_map= {'heart_failure':0,'hypertensive':1,'diabetes':2,'kidney_failure':3}
    if down_task == 'all':
        Y_Test_Data = Y_Test_Data
    else:
        Y_Test_Data = Y_Test_Data[:,[downtask_map[down_task]]]

    test_meta = np.load(os.path.join(data_dir, task, 'Test_meta.npy'))

    X_Test_Data = torch.FloatTensor(X_Test_Data)
    Y_Test_Data = torch.FloatTensor(Y_Test_Data)
    test_meta = torch.FloatTensor(test_meta)
    test_set = CustomDataset(X_Test_Data, Y_Test_Data, test_meta)

    print('test_set', len(test_set))
    batch_size = 128
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    input_modal = config['input_modal']
    latent_dim_conv = config['model']['latent_dim_conv']
    latent_dim_lstm = config['model']['latent_dim_lstm']
    z_dim = config['model']['z_dim']
    merge = config['model']['merge']
    dropout = config['model']['dropout']
    fixed = config['model']['fixed']

    X_shape = X_Test_Data.shape
    Y_shape = Y_Test_Data.shape
    input_dim1 = X_shape[-1]  # ECG 624
    downstream_cls_num = Y_shape[1]
    # experiment = wandb.init(project='cross_modal_re', resume='allow', anonymous='must', config=config,notes=notes)
    model = TripleVAE_downstream(encoder=config['train']['encoder'], downstream_cls_num=downstream_cls_num, known_modality=input_modal,
                               device=DEVICE, z_dim=z_dim, latent_dim_conv=latent_dim_conv, latent_dim_lstm=latent_dim_lstm, in_channels=1, out_channels=1, out_features=input_dim1, merge=merge, dropout=dropout, fixed=fixed)
    model = model.to(DEVICE)
    # experiment.config.update({'model_name': model_name})
    model.eval()
    

    # Evaluation
    print('eval best')
    output_path = foldername + "/model.pth"
    model.load_state_dict(torch.load(output_path, map_location=DEVICE))
    
    # Save as a 16-bit .pth file
    # torch.save(model.state_dict(), foldername+"/model_fp16.pth")  # Save weights

    evaluate(model, config['train'], test_loader,  DEVICE, downstream_cls_num, foldername=foldername)




def evaluate(model, config,test_loader, device, num_classes, foldername=""):

    avg_loss_test= 0
    with torch.no_grad():
        output_all = np.empty((0, num_classes))
        target_all = np.empty((0, num_classes))
        embeddings_all = np.empty((0, 128)) # z_dim=128
        for batch_no, data in enumerate(test_loader, start=1):
            x, y, meta = data
            x, y, meta = x.to(device), y.to(device), meta.to(device)
            if 'meta' in config['encoder']:
                output = model(x, meta)
                embeddings = model.get_embeddings(x, meta)
            else:
                output = model(x)
                embeddings = model.get_embeddings(x)
            pred = torch.sigmoid(output)
            # Concatenate to all outputs
            output_all = np.concatenate((output_all, pred.cpu().numpy()))
            target_all = np.concatenate((target_all, y.cpu().numpy()))
            embeddings_all = np.concatenate((embeddings_all, embeddings.cpu().numpy()))


        # Compute AUROC
        test_auroc = roc_auc_score(target_all, output_all)
        print(f"Test_AUROC: {test_auroc}")
        # wandb.log({"test_AUROC": test_auroc})

        # Write results to a txt file
        with open(foldername + "/result_AUROC.txt", "a") as f:
            f.write(f"{test_auroc}\n")

        
        threshold = 0.5
        y_pred= np.where(output_all>threshold,1,0)   # y_pred: values greater than threshold are set to 1, otherwise 0
        # print(y_pred.shape)
        
        df_result = evaluate_experiment(target_all, output_all,y_pred, metrics='all')

        df_result.to_csv(f'{foldername}/metrics.csv')
        
        

if __name__=='__main__':
    # for fold in range(10):
    #     print(f"fold:{fold}")
    #     main(fold=fold)
    main(fold=2)
