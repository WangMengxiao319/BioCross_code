# -*- coding: utf-8 -*-
# @Time : 2024/12/10 18:57
# @Author : 王梦潇
# @File : Triple_downstream.py
# Function:

import torch
import torch.nn as nn
from torchinfo import summary
from model.BioCross_base import TripleVAE
from model.BioCross import *
# from model.basic_conv1d import AdaptiveConcatPool1d,Flatten

def dfs_freeze(model):
    '''冻结模型参数( "Depth-First Search"（深度优先搜索）)'''
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)

def un_dfs_freeze(model):
    '''解冻模型参数'''
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        un_dfs_freeze(child)

class TripleVAE_downstream(nn.Module):
    def __init__(self,
                 encoder,
                 downstream_cls_num,
                 known_modality,
                 device,
                 z_dim:int,
                 latent_dim_conv: int,
                 latent_dim_lstm:int,
                 in_channels:int,
                 out_channels:int,
                 out_features:int,
                 merge = 'dropout',
                 dropout=0.2,
                 fixed = False
                 ) -> None:
        super(TripleVAE_downstream, self).__init__()
        if encoder == 'TriplemetaVAE_FFT':
            self.pretrain_model_path = './output/II_Pleth_ABP/model.pth'
            self.cross_encoders = TripleVAE_cmeta3_fft(device=device,z_dim =z_dim,latent_dim_conv=latent_dim_conv,latent_dim_lstm=latent_dim_lstm,
                                                       in_channels=in_channels,out_channels=out_channels,out_features=out_features,
                                                       merge=merge,fixed=fixed)
            self.z_dim = z_dim

        self.encoder = encoder
        self.known_modality = known_modality
        self.merge = merge
        self.dropout = dropout
        if self.pretrain_model_path:
            print('load pretrain model')
            model_pretrain_dict = torch.load(self.pretrain_model_path, map_location=device)
            self.cross_encoders.load_state_dict(model_pretrain_dict)
        self.downstream_predictor = nn.Sequential(
            # AdaptiveConcatPool1d(),
            # Flatten(),
            nn.Linear(self.z_dim, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(64, downstream_cls_num)
        )
        # self.downstream_predictor = nn.Sequential(
        #     # AdaptiveConcatPool1d(),
        #     # Flatten(),
        #     nn.Linear(self.z_dim, downstream_cls_num)
        # )

        if fixed:
            dfs_freeze(self.cross_encoders)
    def forward(self, input, meta=None):
        if 'meta' in self.encoder:
            z = self.cross_encoders.get_embeddings(input,self.known_modality, meta)
        elif 'multiVAE' in self.encoder:
            input_x = []
            for channel in range(input.shape[1]):
                input_x.append(input[:, [channel], :])
            z = self.cross_encoders.get_embeddings(input_x, self.known_modality)
        elif 'resnet1d' in self.encoder:
            z = self.cross_encoders(input)

        else:
            z = self.cross_encoders.get_embeddings(input, self.known_modality)
        # Predict the downstream task
        output = self.downstream_predictor(z)
        return output
    def get_embeddings(self, input, meta=None):
        if 'meta' in self.encoder:
            z = self.cross_encoders.get_embeddings(input,self.known_modality, meta)
        elif 'multiVAE' in self.encoder:
            input_x = []
            for channel in range(input.shape[1]):
                input_x.append(input[:, [channel], :])
            z = self.cross_encoders.get_embeddings(input_x, self.known_modality)
        elif 'resnet1d' in self.encoder:
            z = self.cross_encoders(input)
        else:
            z = self.cross_encoders.get_embeddings(input, self.known_modality)
        return z

if __name__ == '__main__':
    model = TripleVAE_downstream(encoder='resnet1d_wang', downstream_cls_num=1, known_modality='ABP', device="cuda", z_dim=128, latent_dim_conv=256, latent_dim_lstm=512, in_channels=1, out_channels=1, out_features=512, merge='dropout', dropout=0.2, fixed=False)
    print(model)
    summary(model, input_size=(1,1, 512), device='cuda')
