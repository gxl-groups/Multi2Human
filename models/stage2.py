import logging
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from torchvision.utils import save_image

from models.vqgan_arch import (Decoder, Encoder, VectorQuantizer,)

logger = logging.getLogger('base')


class ParsingModel():
    """Texture-Aware Diffusion based Transformer model.
    """

    def __init__(self, ):
        self.device = torch.device('cuda')
        self.is_train = True

        # VAE for segmentation mask
        self.segm_encoder = Encoder(
            ch=64,
            num_res_blocks=1,
            attn_resolutions= [16],
            ch_mult=[1, 1, 2, 2, 4],
            in_channels=24,
            resolution=512,
            z_channels=32,
            double_z=False,
            dropout=0.0).to(self.device)
        self.segm_quantizer = VectorQuantizer(
            1024,
            32,
            beta=0.25,
            sane_index_shape=True).to(self.device)
        self.segm_quant_conv = torch.nn.Conv2d(32,
                                               32,
                                               1).to(self.device)

        self.post_quant_conv = torch.nn.Conv2d(32,
                                               32,
                                               1).to(self.device)
        self.decoder = Decoder(
            in_channels=24,
            resolution=512,
            z_channels=32,
            ch=64,
            out_ch=24,
            num_res_blocks=1,
            attn_resolutions=[16],
            ch_mult=[1, 1, 2, 2, 4],
            dropout=0.0,
            resamp_with_conv=True,
            give_pre_end=False).to(self.device)
        self.load_pretrained_segm_vae()


    def load_pretrained_segm_vae(self):
        # load pretrained vqgan for segmentation mask
        segm_ae_checkpoint = torch.load('/home/user/wz/second_project/Multi2Human/pretrained_models/parsing_token.pth')
        self.segm_encoder.load_state_dict(
            segm_ae_checkpoint['encoder'], strict=True)
        self.segm_quantizer.load_state_dict(
            segm_ae_checkpoint['quantize'], strict=True)
        self.segm_quant_conv.load_state_dict(
            segm_ae_checkpoint['quant_conv'], strict=True)
        self.post_quant_conv.load_state_dict(
            segm_ae_checkpoint['post_quant_conv'], strict=True)
        self.decoder.load_state_dict(
            segm_ae_checkpoint['decoder'], strict=True)
        self.segm_encoder.eval()
        self.segm_quantizer.eval()
        self.segm_quant_conv.eval()
        self.post_quant_conv.eval()
        self.decoder.eval()


    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant) # torch.Size([4, 24, 512, 256])
        return dec


    def feed_data(self, data):
        self.segm = data.to(self.device) # parsing
        recon_segm,self.segm_tokens = self.get_quantized_segm(self.segm)
        segm_tokens = self.segm_tokens.view(self.segm.size(0), -1) # T_seg b*512 （同样是VQ的索引）
        return recon_segm,segm_tokens

    @torch.no_grad()
    def get_quantized_segm(self, segm):
        segm_one_hot = F.one_hot(
            segm.squeeze(1).long(),
            num_classes=24).permute(
                0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        encoded_segm_mask = self.segm_encoder(segm_one_hot)
        encoded_segm_mask = self.segm_quant_conv(encoded_segm_mask)
        segm_vq_feature, _, [_, _, segm_tokens] = self.segm_quantizer(encoded_segm_mask) # token
        recon_segm = self.decode(segm_vq_feature)

        return recon_segm,segm_tokens
