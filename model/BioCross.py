import torch
import torch.nn as nn
from torchinfo import summary
from torch.nn import functional as F
from model.BioCross_base import TripleVAE
# from model.basic_conv1d import AdaptiveConcatPool1d,Flatten
from inspect import isfunction
from torch import nn, einsum
from einops import rearrange, repeat

def dfs_freeze(model):
    '''Freeze model parameters (Depth-First Search)'''
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def un_dfs_freeze(model):
    '''Unfreeze model parameters'''
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        un_dfs_freeze(child)

# Define 1D convolutional block
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
        x = x.permute(0, 2, 1)  # [batch_size, seq_length, input_size]

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm1(x)  # [batch_size, seq_length, hidden_size*2]
        latentFeature = self.avgpool(lstm_output.permute(0, 2, 1))
        latentFeature = latentFeature.view(latentFeature.size(0), -1)
        return latentFeature


# Design decoder block
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
    '''Reference: Hinton - Restricted Boltzmann Machines'''
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
        # Define temperature parameter
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

    def forward(self, left: torch.Tensor, right: torch.Tensor, third: torch.Tensor, batch_size: int):
        # Normalize the input embeddings
        left_normed = F.normalize(left, p=2, dim=-1)
        right_normed = F.normalize(right, p=2, dim=-1)
        third_normed = F.normalize(third, p=2, dim=-1)

        # Calculate logits
        logits_left_right = torch.matmul(left_normed, right_normed.T) * torch.exp(self.temperature)
        logits_left_third = torch.matmul(left_normed, third_normed.T) * torch.exp(self.temperature)
        logits_right_third = torch.matmul(right_normed, third_normed.T) * torch.exp(self.temperature)

        # Calculate probabilities
        prob_left_right = F.softmax(logits_left_right, dim=-1)
        prob_left_third = F.softmax(logits_left_third, dim=-1)
        prob_right_third = F.softmax(logits_right_third, dim=-1)

        # Create labels
        labels = torch.eye(batch_size, dtype=torch.float32, device=left.device)

        # Calculate cross-entropy loss
        loss_left_right = F.cross_entropy(prob_left_right, labels, reduction='sum')
        loss_left_third = F.cross_entropy(prob_left_third, labels, reduction='sum')
        loss_right_third = F.cross_entropy(prob_right_third, labels, reduction='sum')

        # Average the losses
        loss = (loss_left_right + loss_left_third + loss_right_third) / 3

        return loss

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v)) 

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.attn1 = CrossAttention(query_dim, heads=heads, dim_head=dim_head, dropout=dropout) # self-attention
        self.attn2 = CrossAttention(query_dim, context_dim=context_dim,heads=heads, dim_head=dim_head, dropout=dropout) # cross-attention
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        # self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None, mask=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        return x


class FFT_transform(nn.Module):
    def __init__(self):
        super(FFT_transform, self).__init__()
    def forward(self, x):
        fft_result = torch.fft.fft(x)
        result = fft_result
        # magnitude = torch.abs(fft_result)  
        # real = fft_result.real  
        # imag = fft_result.imag  
        return result
    def inverse(self, x):
        iff_result = torch.fft.ifft(x)
        return iff_result

class FreqFilter_multihead(nn.Module):
    '''freq filter layer w multihead design, either maxpooling or projection @https://github.com/apple/ml-famae/blob/main/bioFAME/models/transformer_utils.py#L95'''
    def __init__(self, head=4, length=64, mode='pool'):
        super().__init__()
        assert mode in ['pool', 'self']
        self.mode = mode
        self.head, self.dim = head, length
        self.complex_weight = nn.Parameter(torch.randn(head, length, 2, dtype=torch.float32) * 0.02)

        if self.mode == 'self':
            self.filter_weight = nn.Linear(length, head)

    def forward(self, x):
        '''input size: B x N x C'''
        B, N, C = x.shape
        device = x.device

        x = x.to(torch.float32)
        x = torch.fft.rfft(x, dim=(1), norm='ortho')

        if self.mode == 'self':
            head_weight = self.filter_weight(x.real)

        x = repeat(x, 'b p d -> b p h d', h=self.head)
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight

        if self.mode == 'pool':
            x = rearrange(x, 'b p h d -> (b p d) h')
            _, indices = F.max_pool1d(x.abs(), kernel_size=x.shape[-1], return_indices=True)
            batch_range = torch.arange(x.shape[0], device = x.device)[:, None]
            x = x[batch_range, indices][:, 0]
            x = rearrange(x, '(b p d) -> b p d', b=B, d=self.dim)

        elif self.mode == 'self':
            x = x * head_weight.unsqueeze(-1)
            x = torch.sum(x, dim=-2)

        x = torch.fft.irfft(x, n=N, dim=(1), norm='ortho')

        return x

class FFT_CrossAttentionBlock(nn.Module):
    '''FFT+CrossAttention'''
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.fourier_transform = FreqFilter_multihead(head=heads,length = query_dim,mode='self') #mode: 'pool' or 'self'
        # self.attn1 = CrossAttention(query_dim, heads=heads, dim_head=dim_head, dropout=dropout) # self-attention
        self.attn2 = CrossAttention(query_dim, context_dim=context_dim,heads=heads, dim_head=dim_head, dropout=dropout) # cross-attention
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        # self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None, mask=None):
        # x = self.attn1(self.norm1(x)) + x
        # print(x.shape)
        x = self.fourier_transform(self.norm1(x))+ x

        x = self.attn2(self.norm2(x), context=context) + x

        return x



class TripleVAE_cmeta3_fft(nn.Module):
    '''The final BioCross model with FFT cross-attention'''
    def __init__(self,
                 z_dim:int,
                 latent_dim_conv: int,
                 latent_dim_lstm:int,
                 in_channels:int,
                 out_channels:int,
                 out_features:int,
                 device,
                 pretrain_model_path=None,
                 merge='dropout',
                 initial_temperature: float = 1.0,
                 fixed=True,
                 ) -> None:
        super(TripleVAE_cmeta3_fft, self).__init__()

        # merge method
        self.merge = merge
        # self.latent_dim = latent_dim

        self.pretrain_model = TripleVAE(z_dim=z_dim, latent_dim_conv=latent_dim_conv, latent_dim_lstm=latent_dim_lstm,
                                      in_channels=in_channels, out_channels=out_channels, out_features=out_features,
                                      merge=merge)
        self.pretrain_model_path = pretrain_model_path
        if self.pretrain_model_path:
            print('load pretrain model')
            model_pretrain_dict = torch.load(self.pretrain_model_path, map_location=device)
            self.pretrain_model.load_state_dict(model_pretrain_dict)
        if fixed:
            dfs_freeze(self.pretrain_model)

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

        self.cross_attention = FFT_CrossAttentionBlock(query_dim=latent_dim_lstm*2, context_dim=4, heads=4, dim_head=64, dropout=0.2)


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

    def forward(self, input1,input2,input3, meta):

        multi_num=3
        # encoder
        result1, result2, result3 = self.pretrain_model.encode_feature(input1, input2, input3)

        result1 = result1.unsqueeze(1)
        result2 = result2.unsqueeze(1)
        result3 = result3.unsqueeze(1)
        meta = meta.unsqueeze(1)

        result1 = self.cross_attention(result1, meta)
        result2 = self.cross_attention(result2, meta)
        result3 = self.cross_attention(result3, meta)

        result1 = result1.squeeze(1)
        result2 = result2.squeeze(1)
        result3 = result3.squeeze(1)

        mu1, log_var1, mu2, log_var2, mu3, log_var3 = self.pretrain_model.encode_reparameterize(result1, result2, result3)
        z1 = self.reparameterize(mu1, log_var1)
        z2 = self.reparameterize(mu2, log_var2)
        z3 = self.reparameterize(mu3, log_var3)


        # Merge
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

        return output1,output2,output3,input1,input2,input3,mu1, log_var1, mu2, log_var2,mu3, log_var3

    def loss_function(self,
                      output1,output2,output3,input1,input2,input3,mu1, log_var1, mu2, log_var2,mu3, logvar3,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        kld_weight = 1  # Account for the minibatch samples from the dataset
        recons_loss_1 = F.mse_loss(output1, input1,reduction="sum")  # TODO:
        recons_loss_2 = F.mse_loss(output2, input2,reduction="sum")  # TODO:
        recons_loss_3 = F.mse_loss(output3, input3,reduction="sum")  # TODO:
        recons_loss = recons_loss_1 + recons_loss_2 + recons_loss_3

        kld_loss_1 = torch.mean(-0.5 * torch.sum(1 + log_var1 - mu1 ** 2 - log_var1.exp(), dim=1), dim=0)
        kld_loss_2 = torch.mean(-0.5 * torch.sum(1 + log_var2 - mu2 ** 2 - log_var2.exp(), dim=1), dim=0)
        kld_loss_3 = torch.mean(-0.5 * torch.sum(1 + logvar3 - mu3 ** 2 - logvar3.exp(), dim=1), dim=0)
        kld_loss = kld_loss_1 + kld_loss_2 + kld_loss_3

        # loss = recons_loss + kld_weight * kld_loss
        loss = recons_loss


        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(),'KLD':kld_loss.detach()}




    def get_embeddings(self, input,known_modality,meta):
        meta = meta.unsqueeze(1)
        if known_modality == 'ECG':
            # Encode the known modality to get latent variables
            result1,_,_ = self.pretrain_model.encode_feature(input, torch.zeros_like(input), torch.zeros_like(input))
            result1 = result1.unsqueeze(1)
            result1 = self.cross_attention(result1, meta)
            result1 = result1.squeeze(1)
            mu1, log_var1, _, _, _, _ = self.pretrain_model.encode_reparameterize(result1, torch.zeros_like(result1), torch.zeros_like(result1))
            z1 = self.reparameterize(mu1, log_var1)
            z = z1
        elif known_modality == 'PPG':
            _, result2, _ = self.pretrain_model.encode_feature(torch.zeros_like(input), input, torch.zeros_like(input))
            result2 = result2.unsqueeze(1)
            result2 = self.cross_attention(result2, meta)
            result2 = result2.squeeze(1)
            _, _, mu2, log_var2, _, _ = self.pretrain_model.encode_reparameterize(torch.zeros_like(result2), result2, torch.zeros_like(result2))
            z2 = self.reparameterize(mu2, log_var2)
            z = z2

        elif known_modality == 'ABP':
            _, _, result3 = self.pretrain_model.encode_feature(torch.zeros_like(input), torch.zeros_like(input), input)
            result3 = result3.unsqueeze(1)
            result3 = self.cross_attention(result3, meta)
            result3 = result3.squeeze(1)
            _, _, _, _, mu3, logvar3 = self.pretrain_model.encode_reparameterize(torch.zeros_like(result3), torch.zeros_like(result3), result3)
            z3 = self.reparameterize(mu3, logvar3)
            z = z3
        elif known_modality == 'ECG_PPG':
            input1 = input[:, [0], :]
            input2 = input[:, [1], :]
            result1, result2, _ = self.pretrain_model.encode_feature(input1, input2, torch.zeros_like(input1))
            result1 = result1.unsqueeze(1)
            result2 = result2.unsqueeze(1)
            result1 = self.cross_attention(result1, meta)
            result2 = self.cross_attention(result2, meta)
            result1 = result1.squeeze(1)
            result2 = result2.squeeze(1)
            mu1, log_var1, mu2, log_var2, _, _ = self.pretrain_model.encode_reparameterize(result1, result2, torch.zeros_like(result1))
            mu_set = []
            log_var_set = []
            mu_set.append(mu1)
            mu_set.append(mu2)
            log_var_set.append(log_var1)
            log_var_set.append(log_var2)
            poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
            z = self.reparameterize(poe_mu, poe_log_var)
        elif known_modality == 'all':
            input1 = input[:, [0], :]
            input2 = input[:, [1], :]
            input3 = input[:, [2], :]
            result1, result2, result3 = self.pretrain_model.encode_feature(input1, input2, input3)
            result1 = result1.unsqueeze(1)
            result2 = result2.unsqueeze(1)
            result3 = result3.unsqueeze(1)
            result1 = self.cross_attention(result1, meta)
            result2 = self.cross_attention(result2, meta)
            result3 = self.cross_attention(result3, meta)
            result1 = result1.squeeze(1)
            result2 = result2.squeeze(1)
            result3 = result3.squeeze(1)
            mu1, log_var1, mu2, log_var2, mu3, log_var3 = self.pretrain_model.encode_reparameterize(result1, result2, result3)
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

    def product_of_experts(self, mu_set_, log_var_set_):
        '''Reference: Hinton - Restricted Boltzmann Machines'''
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

if __name__== '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TripleVAE_cmeta3_fft(pretrain_model_path='',device=device,z_dim=128, latent_dim_conv=256, latent_dim_lstm=512,in_channels=1, out_channels=1, out_features=512, merge='dropout')
    # # # pretrained_model_path = 'C:\mancy\experiment\cross_modal_re\output\II_Pleth\DualVAE0_dropout\model.pth'
    # # # model = DualVAE_downstream2(pretrain_model_path=pretrained_model_path, downstream_cls_num=1, known_modality='PPG', device=device, z_dim=128, latent_dim_conv=256, latent_dim_lstm=512, in_channels=1, out_channels=1, out_features=512, merge='dropout', dropout=0.2, fixed=False)
    # # # model = VaeEncoder(in_channels=1, latent_dim_conv=256, latent_dim_lstm=512)
    batchisze = 128
    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
    summary(model, input_size=[(batchisze, 1, 512), (batchisze, 1, 512),(batchisze, 1, 512),(batchisze, 4)])
    # summary(model,input_size=(batchisze, 1, 512))
    # summary(model, input_size=[(batchisze, 1, 1024),(batchisze,1, 4)])

    # fft_module = FreqFilter_multihead(length=1024)
    # x = torch.randn(2,1,1024)
    # result = fft_module(x)
    # print(result.shape)

