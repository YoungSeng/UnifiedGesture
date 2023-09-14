import pdb

import numpy as np
import torch as t
import torch.nn as nn

from .encdec import Encoder, Decoder, assert_shape
from .bottleneck import NoBottleneck, Bottleneck
from .utils.logger import average_metrics

import sys
[sys.path.append(i) for i in ['.', '..', './models/dvq']]
from configs.parse_args import parse_args

args = parse_args()
mydevice = t.device('cuda:' + args.gpu)

def dont_update(params):
    for param in params:
        param.requires_grad = False

def update(params):
    for param in params:
        param.requires_grad = True

def calculate_strides(strides, downs):
    return [stride ** down for stride, down in zip(strides, downs)]

# def _loss_fn(loss_fn, x_target, x_pred, hps):
#     if loss_fn == 'l1':
#         return t.mean(t.abs(x_pred - x_target)) / hps.bandwidth['l1']
#     elif loss_fn == 'l2':
#         return t.mean((x_pred - x_target) ** 2) / hps.bandwidth['l2']
#     elif loss_fn == 'linf':
#         residual = ((x_pred - x_target) ** 2).reshape(x_target.shape[0], -1)
#         values, _ = t.topk(residual, hps.linf_k, dim=1)
#         return t.mean(values) / hps.bandwidth['l2']
#     elif loss_fn == 'lmix':
#         loss = 0.0
#         if hps.lmix_l1:
#             loss += hps.lmix_l1 * _loss_fn('l1', x_target, x_pred, hps)
#         if hps.lmix_l2:
#             loss += hps.lmix_l2 * _loss_fn('l2', x_target, x_pred, hps)
#         if hps.lmix_linf:
#             loss += hps.lmix_linf * _loss_fn('linf', x_target, x_pred, hps)
#         return loss
#     else:
#         assert False, f"Unknown loss_fn {loss_fn}"
def _loss_fn(x_target, x_pred):
    return t.mean(t.abs(x_pred - x_target)) 


class VQVAE_ulr(nn.Module):
    def __init__(self, hps, input_dim):
        super().__init__()
        self.VQVAE_1 = VQVAE(hps, 16*4)       # 20230419
        self.VQVAE_2 = VQVAE(hps, 16*2)
        self.VQVAE_3 = VQVAE(hps, 16*1)

    def forward(self, x1, x2, x3):
        y1, loss1, _ = self.VQVAE_1(x1)
        y2, loss2, _ = self.VQVAE_2(x2)
        y3, loss3, _ = self.VQVAE_3(x3)
        return y1, y2, y3, loss1, loss2, loss3

    # def get_code(self, x1, x2, x3):
    #     zs1 = self.VQVAE_1.encode(x1)
    #     zs2 = self.VQVAE_2.encode(x2)
    #     zs3 = self.VQVAE_3.encode(x3)
    #     return zs1, zs2, zs3
    #
    # def code_2_pose(self, zs1, zs2, zs3):
    #     z1 = self.VQVAE_1.decode(zs1)
    #     z2 = self.VQVAE_2.decode(zs2)
    #     z3 = self.VQVAE_3.decode(zs3)
    #     return z1, z2, z3

    def get_code(self, x1):         # 20230426 we only use x1(upper)
        zs1, distance_list = self.VQVAE_1.encode(x1)
        return zs1, distance_list

    def code_2_pose(self, zs1):
        z1 = self.VQVAE_1.decode(zs1)
        return z1

    def freeze_drop(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()


class VQVAE_ulr_2(nn.Module):
    def __init__(self, hps_1, hps_2):
        super().__init__()
        self.VQVAE_1 = VQVAE(hps_1, 16*4)       # 20230419
        self.VQVAE_2 = VQVAE(hps_2, 16*3)

    def forward(self, x1, x2):
        y1, loss1, _ = self.VQVAE_1(x1)
        y2, loss2, _ = self.VQVAE_2(x2)
        return y1, y2, loss1, loss2

    def get_code(self, x1, x2):
        zs1 = self.VQVAE_1.encode(x1)
        zs2 = self.VQVAE_2.encode(x2)
        return zs1, zs2

    def code_2_pose(self, zs1, zs2):
        z1 = self.VQVAE_1.decode(zs1)
        z2 = self.VQVAE_2.decode(zs2)
        return z1, z2

    def code_pose(self, zs1):
        z1 = self.VQVAE_1.decode(zs1)
        return z1


# from models.dvq.vqvae_2 import VQVAE as vqvae_2
# class VQVAE_ulr_easy(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.VQVAE_1 = vqvae_2(64)
#         self.VQVAE_2 = vqvae_2(32)
#         self.VQVAE_3 = vqvae_2(16)
#
#     def forward(self, x1, x2, x3):
#         y1, loss1, _ = self.VQVAE_1(x1)
#         y2, loss2, _ = self.VQVAE_2(x2)
#         y3, loss3, _ = self.VQVAE_3(x3)
#         return y1, y2, y3, loss1, loss2, loss3
#
#     def get_code(self, x1, x2, x3):
#         _, _, zs1 = self.VQVAE_1(x1)
#         _, _, zs2 = self.VQVAE_2(x2)
#         _, _, zs3 = self.VQVAE_3(x3)
#         return zs1, zs2, zs3


# from models.modules import VectorQuantizedVAE
# class VQVAE_ulr_easy(nn.Module):
#     def __init__(self):
#         super(VQVAE_ulr_easy, self).__init__()
#         self.VQVAE_1 = VectorQuantizedVAE(16, 256, 512)
#         self.VQVAE_2 = VectorQuantizedVAE(16, 256, 512)
#         self.VQVAE_3 = VectorQuantizedVAE(16, 256, 512)
#
#     def forward(self, x1, x2, x3):
#         y1, loss1 = self.VQVAE_1(x1)
#         y2, loss2 = self.VQVAE_2(x2)
#         y3, loss3 = self.VQVAE_3(x3)
#         return y1, y2, y3, loss1, loss2, loss3
#
#     def get_code(self, x1, x2, x3):
#         zs1 = self.VQVAE_1.encode(x1)
#         zs2 = self.VQVAE_2.encode(x2)
#         zs3 = self.VQVAE_3.encode(x3)
#         return zs1, zs2, zs3

'''
from vector_quantize_pytorch import VectorQuantize as ResidualVQ

class VQVAE_ulr_easy(nn.Module):
    def __init__(self):
        super(VQVAE_ulr_easy, self).__init__()
        self.VQVAE_1 = ResidualVQ(
                                dim = 4 * 16,
                                codebook_size = 512,
                                # num_quantizers = 1,
                                decay=0.8,
                                commitment_weight=1.0,
                                kmeans_init = True,   # set to True
                                kmeans_iters = 10     # number of kmeans iterations to calculate the centroids for the codebook on init
                                )
        self.VQVAE_2 = ResidualVQ(
                                dim = 2 * 16,
                                codebook_size = 512,
                                # num_quantizers = 1,
                                kmeans_init = True,   # set to True
                                kmeans_iters = 10     # number of kmeans iterations to calculate the centroids for the codebook on init
                                )
        self.VQVAE_3 = ResidualVQ(
                                dim = 1 * 16,
                                codebook_size = 512,
                                # num_quantizers = 1,
                                kmeans_init = True,   # set to True
                                kmeans_iters = 10     # number of kmeans iterations to calculate the centroids for the codebook on init
                                )

    def forward(self, x1, x2, x3):
        y1, _, loss1 = self.VQVAE_1(x1)
        y2, _, loss2 = self.VQVAE_2(x2)
        y3, _, loss3 = self.VQVAE_3(x3)
        return y1, y2, y3, loss1, loss2, loss3
'''


class VQVAE(nn.Module):
    def __init__(self, hps, input_dim=72):
        super().__init__()
        self.hps = hps

        input_shape = (hps.sample_length, input_dim)
        levels = hps.levels
        downs_t = hps.downs_t
        strides_t = hps.strides_t
        emb_width = hps.emb_width
        l_bins = hps.l_bins
        mu = hps.l_mu
        commit = hps.commit
        # spectral = hps.spectral
        # multispectral = hps.multispectral
        multipliers = hps.hvqvae_multipliers 
        use_bottleneck = hps.use_bottleneck
        if use_bottleneck:
            print('We use bottleneck!')
        else:
            print('We do not use bottleneck!')
        if not hasattr(hps, 'dilation_cycle'):
            hps.dilation_cycle = None
        block_kwargs = dict(width=hps.width, depth=hps.depth, m_conv=hps.m_conv, \
                        dilation_growth_rate=hps.dilation_growth_rate, \
                        dilation_cycle=hps.dilation_cycle, \
                        reverse_decoder_dilation=hps.vqvae_reverse_decoder_dilation)

        self.sample_length = input_shape[0]
        x_shape, x_channels = input_shape[:-1], input_shape[-1]
        self.x_shape = x_shape

        self.downsamples = calculate_strides(strides_t, downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)
        self.z_shapes = z_shapes = [(x_shape[0] // self.hop_lengths[level],) for level in range(levels)]

        self.levels = levels

        if multipliers is None:
            self.multipliers = [1] * levels
        else:
            assert len(multipliers) == levels, "Invalid number of multipliers"
            self.multipliers = multipliers
        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs

        encoder = lambda level: Encoder(x_channels, emb_width, level + 1,
                                        downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))     # different from supplemental
        decoder = lambda level: Decoder(x_channels, emb_width, level + 1,
                                        downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))

        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, emb_width, mu, levels)     # 512, 512, 0.99, 1
        else:
            self.bottleneck = NoBottleneck(levels)

        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.commit = commit
        self.reg = hps.reg if hasattr(hps, 'reg') else 0
        self.acc = hps.acc if hasattr(hps, 'acc') else 0
        self.vel = hps.vel if hasattr(hps, 'vel') else 0
        if self.reg is 0:
            print('No motion regularization!')
        # self.spectral = spectral
        # self.multispectral = multispectral

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        x = x.permute(0,2,1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0,2,1)
        return x

    def _decode(self, zs, start_level=0, end_level=None):
        # Decode
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantised = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)
        assert len(xs_quantised) == end_level - start_level

        # Use only lowest level
        decoder, x_quantised = self.decoders[start_level], xs_quantised[0:1]
        x_out = decoder(x_quantised, all_levels=False)
        x_out = self.postprocess(x_out)
        return x_out

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out = self._decode(zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        return t.cat(x_outs, dim=0)

    def _encode(self, x, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs = self.bottleneck.encode(xs)
        return zs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        distance_list = []
        for x_i in x_chunks:
            zs_i = self._encode(x_i, start_level=start_level, end_level=end_level)
            zs_i, distance = zs_i[0]
            zs_list.append([zs_i])
            distance_list.append(distance)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs, distance_list

    def sample(self, n_samples):
        zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape), device=mydevice) for z_shape in self.z_shapes]
        return self.decode(zs)

    def forward(self, x):       # (32, 240, 45)
        metrics = {}

        N = x.shape[0]

        # Encode/Decode
        x_in = self.preprocess(x)       # (32, 45, 240)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        # xs[0]: (32, 512, 30)
        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(xs)
        '''
        zs[0]: (32, 30)
        xs_quantised[0]: (32, 512, 30)
        commit_losses[0]: 0.0009
        quantiser_metrics[0]: 
            fit 0.4646
            pn 0.0791
            entropy 5.9596
            used_curr 512
            usage 512
            dk 0.0006
        '''
        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs_quantised[level:level+1], all_levels=False)
            assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)
        # x_outs[0]: (32, 45, 240)

        # Loss
        # def _spectral_loss(x_target, x_out, self.hps):
        #     if hps.use_nonrelative_specloss:
        #         sl = spectral_loss(x_target, x_out, self.hps) / hps.bandwidth['spec']
        #     else:
        #         sl = spectral_convergence(x_target, x_out, self.hps)
        #     sl = t.mean(sl)
        #     return sl

        # def _multispectral_loss(x_target, x_out, self.hps):
        #     sl = multispectral_loss(x_target, x_out, self.hps) / hps.bandwidth['spec']
        #     sl = t.mean(sl)
        #     return sl

        recons_loss = t.zeros(()).to(x.device)
        regularization = t.zeros(()).to(x.device)
        velocity_loss = t.zeros(()).to(x.device)
        acceleration_loss = t.zeros(()).to(x.device)
        # spec_loss = t.zeros(()).to(x.device)
        # multispec_loss = t.zeros(()).to(x.device)
        # x_target = audio_postprocess(x.float(), self.hps)
        x_target = x.float()

        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])     # (32, 240, 45)
            # x_out = audio_postprocess(x_out, self.hps)
            

            # this_recons_loss = _loss_fn(loss_fn, x_target, x_out, hps)
            this_recons_loss = _loss_fn(x_target, x_out)
            # this_spec_loss = _spectral_loss(x_target, x_out, hps)
            # this_multispec_loss = _multispectral_loss(x_target, x_out, hps)
            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            # metrics[f'spectral_loss_l{level + 1}'] = this_spec_loss
            # metrics[f'multispectral_loss_l{level + 1}'] = this_multispec_loss
            recons_loss += this_recons_loss
            # spec_loss += this_spec_loss
            # multispec_loss += this_multispec_loss
            regularization += t.mean((x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1])**2)

            velocity_loss +=  _loss_fn( x_out[:, 1:] - x_out[:, :-1], x_target[:, 1:] - x_target[:, :-1])
            acceleration_loss +=  _loss_fn(x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1], x_target[:, 2:] + x_target[:, :-2] - 2 * x_target[:, 1:-1])
        # if not hasattr(self.)
        commit_loss = sum(commit_losses)
        # loss = recons_loss + self.spectral * spec_loss + self.multispectral * multispec_loss + self.commit * commit_loss
        # pdb.set_trace()
        loss = recons_loss +  commit_loss * self.commit + self.reg * regularization + self.vel * velocity_loss + self.acc * acceleration_loss
        '''     x:-0.8474 ~ 1.1465
        0.2080
        5.5e-5 * 0.02
        0.0011
        0.0163 * 1
        0.0274 * 1
        '''

        with t.no_grad():
            # sc = t.mean(spectral_convergence(x_target, x_out, hps))
            # l2_loss = _loss_fn("l2", x_target, x_out, hps)
            l1_loss = _loss_fn(x_target, x_out)
            
            # linf_loss = _loss_fn("linf", x_target, x_out, hps)

        quantiser_metrics = average_metrics(quantiser_metrics)

        metrics.update(dict(
            recons_loss=recons_loss,
            # spectral_loss=spec_loss,
            # multispectral_loss=multispec_loss,
            # spectral_convergence=sc,
            # l2_loss=l2_loss,
            l1_loss=l1_loss,
            # linf_loss=linf_loss,
            commit_loss=commit_loss,
            regularization=regularization,
            velocity_loss=velocity_loss,
            acceleration_loss=acceleration_loss,
            **quantiser_metrics))

        for key, val in metrics.items():
            metrics[key] = val.detach()

        return x_out, loss, metrics




if __name__ == '__main__':
    '''
    cd codebook/
    python -m models.vqvae --config=./configs/codebook.yml --train --no_cuda 2 --gpu 2
    '''
    import yaml
    from pprint import pprint
    from easydict import EasyDict

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)

    x = t.rand(1, 2200, 16 * 4).to(mydevice)
    model = VQVAE_ulr(config.VQVAE, 16 * 7)     # n_joints * n_chanels

    model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
    model = model.to(mydevice)
    model = model.train()
    # output, loss, metrics = model(x)
    zs, distance = model.module.get_code(x)     # [32, 8]
    pdb.set_trace()

    # from torch import optim
    # model = VQVAE_ulr_easy().to(mydevice)
    # optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=[0.5, 0.999])
    # x1 = t.rand(32, 30, 4 * 16).to(mydevice)
    # x2 = t.rand(32, 30, 2 * 16).to(mydevice)
    # x3 = t.rand(32, 30, 1 * 16).to(mydevice)
    # y1, y2, y3, loss1, loss2, loss3 = model(x1, x2, x3)
    # pdb.set_trace()
