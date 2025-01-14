# -*- coding: utf-8 -*-
# @Time : 2024/6/21 12:14
# @Author : 王梦潇
# @File : VAE.py

import torch
import torch.nn as nn
from torchinfo import summary
from torch.nn import functional as F
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

# 定义编码器
def conv_1d(in_planes, out_planes, stride=1, size=3):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=size, stride=stride, padding=(size - 1) // 2, bias=False)


class BasicBlock1d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=(1,)):
        super(BasicBlock1d, self).__init__()
        latent_dim = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(inplanes, latent_dim, kernel_size=(1,), stride=stride, bias=False),
            nn.BatchNorm1d(latent_dim))

        self.conv2 = nn.Sequential(
            nn.Conv1d(inplanes, latent_dim, kernel_size=(3,), stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(latent_dim))

        self.conv3 = nn.Sequential(
            nn.Conv1d(inplanes, latent_dim, kernel_size=(7,), stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(latent_dim))

        self.conv4 = nn.Sequential(
            nn.Conv1d(inplanes, latent_dim, kernel_size=(11,), stride=stride, padding=5, bias=False),
            nn.BatchNorm1d(latent_dim))
        self.featureFusion = nn.Sequential(
            nn.Conv1d(latent_dim * 4, planes, kernel_size=(9,), stride=(1, ), padding=4, bias=False),
            nn.BatchNorm1d(planes))
        self.relu = nn.LeakyReLU()

        self.stride = stride
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=(1,), padding=0, stride=stride,
                          bias=False),
                nn.BatchNorm1d(planes),
            )


    def forward(self, x):
        residual = x
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = self.relu(out)
        out = self.featureFusion(out)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class VaeEncoder(nn.Module):

    def __init__(self, in_channels=12, latent_dim_conv=256, latent_dim_lstm=512):
        super(VaeEncoder, self).__init__()
        # First layers
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=(13,), bias=False,
                               stride=(2,), padding=(6,))
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=(13,), bias=False,
                               stride=(1,), padding=(6,))
        self.layers1 = nn.Sequential()

        self.inplanes = 32
        self.layers1.add_module('layer_1',
                                BasicBlock1d(32, latent_dim_conv, stride=2))
        self.layers1.add_module('layer_2',
                                BasicBlock1d(latent_dim_conv, latent_dim_conv, stride=2))
        # self.inplanes *= 8
        self.layers1.add_module('layer_3',
                                BasicBlock1d(latent_dim_conv, latent_dim_conv, stride=4))
        self.layers1.add_module('layer_4',
                                BasicBlock1d(latent_dim_conv, latent_dim_conv, stride=4))


        self.lstm1 = nn.LSTM(input_size=256, hidden_size=latent_dim_lstm, batch_first=True, bidirectional=True)
        self.latent_dim = latent_dim_lstm*2
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.conv2(x0)

        x = self.layers1(x0)
        x = x.permute(0, 2, 1)

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm1(x)
        latentFeature = self.avgpool(lstm_output.permute(0, 2, 1))
        latentFeature = latentFeature.view(latentFeature.size(0), -1)
        return latentFeature


# 设计ResDecoder
# 用上采样加卷积代替了反卷积
class ResizeConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size // 2))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockDec(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, scale_factor=1):
        super(BasicBlockDec, self).__init__()
        latent_dim = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(inplanes, latent_dim, kernel_size=(1,), stride=(1,), bias=False),
            nn.BatchNorm1d(latent_dim))

        self.conv2 = nn.Sequential(
            nn.Conv1d(inplanes, latent_dim, kernel_size=(3,), stride=(1,), padding=1, bias=False),
            nn.BatchNorm1d(latent_dim))

        self.conv3 = nn.Sequential(
            nn.Conv1d(inplanes, latent_dim, kernel_size=(7,), stride=(1,), padding=3, bias=False),
            nn.BatchNorm1d(latent_dim))

        self.conv4 = nn.Sequential(
            nn.Conv1d(inplanes, latent_dim, kernel_size=(11,), stride=(1,), padding=5, bias=False),
            nn.BatchNorm1d(latent_dim))
        self.featureFusion = nn.Sequential(
            ResizeConv1d(latent_dim * 4, planes, kernel_size=9, scale_factor=scale_factor),
            nn.BatchNorm1d(planes))
        self.relu = nn.LeakyReLU()

        if scale_factor != 1 or self.inplanes != planes:
            self.downsample = nn.Sequential(
                ResizeConv1d(self.inplanes, planes, kernel_size=1, scale_factor=scale_factor),
                nn.BatchNorm1d(planes),
            )
    def forward(self, x):
        residual = x
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = self.relu(out)
        out = self.featureFusion(out)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out

class VaeDecoder(nn.Module):

    def __init__(self, input_dim=512,out_channels=12):
        super().__init__()
        self.input_dim = input_dim

        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=(3,), bias=False,
                               stride=(1,), padding=(1,))
        self.bn1 = nn.BatchNorm1d(32)

        self.layers1 = nn.Sequential()

        self.inplanes = 32
        self.layers1.add_module('layer_1',
                                BasicBlock1d(32, 256, stride=1))
        self.layers1.add_module('layer_2',
                                BasicBlock1d(256, 256, stride=1))
        # self.inplanes *= 8
        self.layers1.add_module('layer_3',
                                BasicBlock1d(256, 256, stride=1))
        self.layers1.add_module('layer_4',
                                BasicBlock1d(256, 256, stride=1))

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 32, kernel_size=(1,), stride=(1,), bias=False),
            nn.BatchNorm1d(32))

        self.conv3 = nn.Conv1d(32, out_channels, kernel_size=(1,), stride=(1,), bias=False)

    def forward(self, x0):
        x0 = x0.unsqueeze(1)
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        # x0 = self.relu(x0)
        x0 = self.layers1(x0)

        out = self.conv2(x0)
        out = self.relu(out)
        out = self.conv3(out)
        return out

def product_of_experts(mu_set_, log_var_set_):
    '''参考：hinton-受限玻尔兹曼机'''
    tmp = 0
    for i in range(len(mu_set_)):
        tmp += torch.div(1, torch.exp(log_var_set_[i]))

    poe_var = torch.div(1., tmp)
    poe_log_var = torch.log(poe_var)

    tmp = 0.
    for i in range(len(mu_set_)):
        tmp += torch.div(1., torch.exp(log_var_set_[i])) * mu_set_[i]
    poe_mu = poe_var * tmp
    return poe_mu, poe_log_var

class ContrastiveLoss(nn.Module):
    def __init__(self, initial_temperature: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        # 定义可训练的温度参数
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

    def forward(self, left: torch.Tensor, right: torch.Tensor, third: torch.Tensor, batch_size: int):
        # 归一化
        left_normed = F.normalize(left, p=2, dim=-1)
        right_normed = F.normalize(right, p=2, dim=-1)
        third_normed = F.normalize(third, p=2, dim=-1)

        # 计算对比损失的logits
        logits_left_right = torch.matmul(left_normed, right_normed.T) * torch.exp(self.temperature)
        logits_left_third = torch.matmul(left_normed, third_normed.T) * torch.exp(self.temperature)
        logits_right_third = torch.matmul(right_normed, third_normed.T) * torch.exp(self.temperature)

        # 计算概率
        prob_left_right = F.softmax(logits_left_right, dim=-1)
        prob_left_third = F.softmax(logits_left_third, dim=-1)
        prob_right_third = F.softmax(logits_right_third, dim=-1)

        # 构造标签矩阵
        labels = torch.eye(batch_size, dtype=torch.float32, device=left.device)

        # 计算交叉熵损失
        loss_left_right = F.cross_entropy(prob_left_right, labels, reduction='sum')
        loss_left_third = F.cross_entropy(prob_left_third, labels, reduction='sum')
        loss_right_third = F.cross_entropy(prob_right_third, labels, reduction='sum')

        # 平均损失
        loss = (loss_left_right + loss_left_third + loss_right_third) / 3

        return loss


# 定义 VAE
class TripleVAE(nn.Module):

    def __init__(self,
                 z_dim:int,
                 latent_dim_conv: int,
                 latent_dim_lstm:int,
                 in_channels:int,
                 out_channels:int,
                 out_features:int,
                 merge = 'concat',
                 initial_temperature: float = 1.0
                 ) -> None:
        super(TripleVAE, self).__init__()

        # merge方式
        self.merge = merge
        # self.latent_dim = latent_dim

        self.encoder1 = VaeEncoder(in_channels=in_channels,latent_dim_conv=latent_dim_conv,latent_dim_lstm=latent_dim_lstm)
        self.encoder2 = VaeEncoder(in_channels=in_channels,latent_dim_conv=latent_dim_conv,latent_dim_lstm=latent_dim_lstm)
        self.encoder3 = VaeEncoder(in_channels=in_channels,latent_dim_conv=latent_dim_conv,latent_dim_lstm=latent_dim_lstm)
        self.fc_mu1 = nn.Linear(self.encoder1.latent_dim, z_dim)
        self.fc_var1 = nn.Linear(self.encoder1.latent_dim, z_dim)
        self.fc_mu2 = nn.Linear(self.encoder2.latent_dim, z_dim)
        self.fc_var2 = nn.Linear(self.encoder2.latent_dim, z_dim)
        self.fc_mu3 = nn.Linear(self.encoder3.latent_dim, z_dim)
        self.fc_var3 = nn.Linear(self.encoder3.latent_dim, z_dim)

        # Build Decoder
        if self.merge == 'concat':
            z_dim = z_dim * 3

        self.decoder_input1 = nn.Linear(z_dim, out_features)
        self.decoder_input2 = nn.Linear(z_dim, out_features)
        self.decoder_input3 = nn.Linear(z_dim, out_features)

        self.decoder1 = VaeDecoder(out_channels=out_channels)
        self.decoder2 = VaeDecoder(out_channels=out_channels)
        self.decoder3 = VaeDecoder(out_channels=out_channels)

        self.contrastive_loss = ContrastiveLoss(initial_temperature=initial_temperature)

    def encode(self, input1, input2, input3):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        ##### modality 1
        result1 = self.encoder1(input1)
        result1 = torch.flatten(result1, start_dim=1)
        mu1 = self.fc_mu1(result1)         # Split the result into mu and var components of the latent Gaussian distribution
        log_var1 = self.fc_var1(result1)
        ###### modality 2
        result2 = self.encoder2(input2)
        result2 = torch.flatten(result2, start_dim=1)
        mu2 = self.fc_mu2(result2)
        log_var2 = self.fc_var2(result2)
        ### modality3
        result3 = self.encoder3(input3)
        result3 = torch.flatten(result3, start_dim=1)

        mu3 = self.fc_mu3(result3)
        log_var3 = self.fc_var3(result3)

        return [mu1, log_var1, mu2, log_var2, mu3, log_var3]

    def encode_feature(self, input1, input2, input3):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        ##### modality 1
        result1 = self.encoder1(input1)
        result1 = torch.flatten(result1, start_dim=1)

        ###### modality 2
        result2 = self.encoder2(input2)
        result2 = torch.flatten(result2, start_dim=1)

        ### modality3
        result3 = self.encoder3(input3)
        result3 = torch.flatten(result3, start_dim=1)

        return [result1, result2, result3]

    def encode_reparameterize(self, result1, result2, result3):
        mu1 = self.fc_mu1(result1)         # Split the result into mu and var components of the latent Gaussian distribution
        log_var1 = self.fc_var1(result1)
        mu2 = self.fc_mu2(result2)
        log_var2 = self.fc_var2(result2)
        mu3 = self.fc_mu3(result3)
        logvar3 = self.fc_var3(result3)
        return mu1, log_var1, mu2, log_var2, mu3, logvar3

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result1 = self.decoder_input1(z)
        result1 = self.decoder1(result1)

        result2 = self.decoder_input2(z)
        result2 = self.decoder2(result2)

        result3 = self.decoder_input3(z)
        result3 = self.decoder3(result3)
        return result1, result2, result3

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input1,input2,input3):

        multi_num=3
        # encoder
        mu1, log_var1, mu2, log_var2, mu3, logvar3 = self.encode(input1,input2,input3)
        z1 = self.reparameterize(mu1, log_var1)   # (batch,z_dim)
        z2 = self.reparameterize(mu2, log_var2)   # # (batch,z_dim)
        z3 = self.reparameterize(mu3, logvar3)

        # 融合
        if self.merge == 'concat':
            z = torch.cat((z1, z2, z3), dim=1)
        elif self.merge == 'mean':
            z = torch.mean(torch.stack([z1, z2, z3]), dim=0)
        elif self.merge == 'dropout':
            random_index = torch.randint(0, multi_num, (z1.shape[-1],), device=z1.device)
            index_expanded = random_index.unsqueeze(0).unsqueeze(1).expand(-1, z1.shape[0], -1)  # (1, batch_size, z_dim)
            z_all = torch.stack((z1,z2,z3), dim=0)
            output = torch.gather(z_all, 0, index_expanded)
            z = output.squeeze(0)  #(batch_size, z_dim)



        # decoder
        output1,output2,output3 = self.decode(z)

        return output1,output2,output3,input1,input2,input3, mu1, log_var1, mu2, log_var2,mu3, logvar3

    def loss_function(self,
                      output1,output2,output3,input1,input2,input3, mu1, log_var1, mu2, log_var2,mu3, logvar3, batch_size,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        z1 = self.reparameterize(mu1, log_var1)  # (batch,z_dim)
        z2 = self.reparameterize(mu2, log_var2)  # # (batch,z_dim)
        z3 = self.reparameterize(mu3, logvar3)

        kld_weight = 1  # Account for the minibatch samples from the dataset
        recons_loss_1 = F.mse_loss(output1, input1,reduction="sum")  # TODO:
        recons_loss_2 = F.mse_loss(output2, input2,reduction="sum")  # TODO:
        recons_loss_3 = F.mse_loss(output3, input3,reduction="sum")  # TODO:
        recons_loss = recons_loss_1 + recons_loss_2 + recons_loss_3

        kld_loss_1 = torch.mean(-0.5 * torch.sum(1 + log_var1 - mu1 ** 2 - log_var1.exp(), dim=1), dim=0)
        kld_loss_2 = torch.mean(-0.5 * torch.sum(1 + log_var2 - mu2 ** 2 - log_var2.exp(), dim=1), dim=0)
        kld_loss_3 = torch.mean(-0.5 * torch.sum(1 + logvar3 - mu3 ** 2 - logvar3.exp(), dim=1), dim=0)
        kld_loss = kld_loss_1 + kld_loss_2 + kld_loss_3

        # 计算对比损失
        contrastive_loss = self.contrastive_loss(z1, z2,z3, batch_size)

        loss = recons_loss + kld_weight * kld_loss + contrastive_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': kld_loss.detach(),'Contrastive_Loss': contrastive_loss.detach()}

    def get_embeddings(self, input,known_modality):
        if known_modality == 'ECG':
            # Encode the known modality to get latent variables
            mu1, log_var1, _, _ ,_,_= self.encode(input, torch.zeros_like(input),torch.zeros_like(input))
            z1 = self.reparameterize(mu1, log_var1)
            if self.merge == 'concat':
                z = torch.cat((z1, z1,z1), dim=1)  # Concatenate the same latent vector
            elif self.merge == 'mean':
                z = z1  # For mean, it's just the same latent vector
            elif self.merge == 'dropout':
                z = z1  # In inference, use the single known latent vector
        elif known_modality == 'PPG':
            _, _, mu2, log_var2,_,_ = self.encode(torch.zeros_like(input), input,torch.zeros_like(input))
            z2 = self.reparameterize(mu2, log_var2)
            if self.merge == 'concat':
                z = torch.cat((z2, z2,z2), dim=1)
            elif self.merge == 'mean':
                z = z2
            elif self.merge == 'dropout':
                z = z2

        elif known_modality == 'ABP':
            _, _, _, _, mu3, log_var3 = self.encode(torch.zeros_like(input), torch.zeros_like(input),input)
            z3 = self.reparameterize(mu3, log_var3)
            if self.merge == 'concat':
                z = torch.cat((z3, z3,z3), dim=1)
            elif self.merge == 'mean':
                z = z3
            elif self.merge == 'dropout':
                z = z3
        elif known_modality == 'ECG_PPG':
            input1 = input[:, [0], :]
            input2 = input[:, [1], :]
            mu1, log_var1, mu2, log_var2,_,_ = self.encode(input1, input2,torch.zeros_like(input1))
            z1 = self.reparameterize(mu1, log_var1)
            z2 = self.reparameterize(mu2, log_var2)
            mu_set = []
            log_var_set = []
            mu_set.append(mu1)
            mu_set.append(mu2)
            log_var_set.append(log_var1)
            log_var_set.append(log_var2)
            poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
            z = self.reparameterize(poe_mu, poe_log_var)
            if self.merge == 'concat':
                z = torch.cat((z,z,z), dim=1)
            elif self.merge == 'mean':
                z = (z1 + z2)/2
            elif self.merge == 'dropout':
                z = z
        elif known_modality == 'all':
            input1 = input[:, [0], :]
            input2 = input[:, [1], :]
            input3 = input[:, [2], :]
            mu1, log_var1, mu2, log_var2,mu3, log_var3 = self.encode(input1, input2, input3)
            mu_set = []
            log_var_set = []
            mu_set.append(mu1)
            mu_set.append(mu2)
            mu_set.append(mu3)
            log_var_set.append(log_var1)
            log_var_set.append(log_var2)
            log_var_set.append(log_var3)
            poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
            z = self.reparameterize(poe_mu, poe_log_var)
        return z
    def get_specific_embeddings(self,input1, input2, input3):
        mu1, log_var1, mu2, log_var2, mu3, logvar3 = self.encode(input1,input2,input3)
        z1 = self.reparameterize(mu1, log_var1)   # (batch,z_dim)
        z2 = self.reparameterize(mu2, log_var2)   # # (batch,z_dim)
        z3 = self.reparameterize(mu3, logvar3)
        return z1,z2,z3
    def product_of_experts(self, mu_set_, log_var_set_):
        '''参考：hinton-受限玻尔兹曼机'''
        tmp = 0
        for i in range(len(mu_set_)):
            tmp += torch.div(1, torch.exp(log_var_set_[i]))

        poe_var = torch.div(1., tmp)
        poe_log_var = torch.log(poe_var)

        tmp = 0.
        for i in range(len(mu_set_)):
            tmp += torch.div(1., torch.exp(log_var_set_[i])) * mu_set_[i]
        poe_mu = poe_var * tmp
        return poe_mu, poe_log_var

class TripleVAE_downstream(nn.Module):
    def __init__(self,
                 pretrain_model_path,
                 downstream_cls_num,
                 known_modality,
                 device,
                 z_dim:int,
                 latent_dim_conv: int,
                 latent_dim_lstm:int,
                 in_channels:int,
                 out_channels:int,
                 out_features:int,
                 merge = 'concat',
                 dropout=0.2,
                 fixed = False
                 ) -> None:
        super(TripleVAE_downstream, self).__init__()
        self.pretrain_model_path = pretrain_model_path
        self.known_modality = known_modality
        self.cross_encoders = TripleVAE(z_dim=z_dim, latent_dim_conv=latent_dim_conv,latent_dim_lstm=latent_dim_lstm, in_channels=in_channels, out_channels=out_channels, out_features=out_features, merge=merge)
        self.merge = merge
        self.dropout = dropout
        if self.pretrain_model_path:
            print('load pretrain model')
            model_pretrain_dict = torch.load(self.pretrain_model_path, map_location=device)
            self.cross_encoders.load_state_dict(model_pretrain_dict)
        self.downstream_predictor = nn.Sequential(
            # AdaptiveConcatPool1d(),
            # Flatten(),
            nn.Linear(z_dim, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(64, downstream_cls_num)
        )
        if fixed:
            dfs_freeze(self.cross_encoders)
    def forward(self, input):
        z = self.cross_encoders.get_embeddings(input, self.known_modality)
        # Predict the downstream task
        output = self.downstream_predictor(z)
        return output




