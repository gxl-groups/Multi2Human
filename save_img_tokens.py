from numpy.core.fromnumeric import mean
import torch
import numpy as np
import copy
import time
import torch.nn.functional as F
from torchvision.utils import save_image
import os
from tqdm import tqdm
from models import VQAutoEncoder
from utils.data_utils import get_data_loaders
from utils.sampler_utils import generate_latent_ids, generate_latent_ids1, retrieve_autoencoder_components_state_dicts
from utils.log_utils import log,set_up_visdom,config_log,start_training_log, \
    save_stats, load_stats, save_model, load_model, save_images,\
    display_images
from hparams import get_sampler_hparams
os.environ['CUDA_VISIBLE_DEVICES']='0'
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enable =True

def main(H):
    # stage1保存训练的img_token
    latents_filepath = f'tokens'
    train_with_validation_dataset = False
    # if H.steps_per_eval:  # 0
    #     train_with_validation_dataset = True
    if not os.path.exists(latents_filepath):  # load VQGAN
        ae_state_dict = retrieve_autoencoder_components_state_dicts(
            H, ['encoder', 'quantize', 'generator']
        )
        ae = VQAutoEncoder(H)
        ae.load_state_dict(ae_state_dict, strict=False)
        # val_loader will be assigned to None if not training with validation dataest
        train_loader,image_fnames = get_data_loaders(
            H.dataset,
            H.img_size,
            H.batch_size,
            drop_last=False,
            shuffle=False,
            get_flipped=H.horizontal_flip,
            get_val_dataloader=False
        )

        log("Transferring autoencoder to GPU to generate latents...")
        ae = ae.cuda()  # put ae on GPU for generating
        # !!
        generate_latent_ids1(H, ae, train_loader,image_fnames)
        log("Deleting autoencoder to conserve GPU memory...")
        ae = ae.cpu()
        ae = None

        print('finished!!!!')


if __name__ == '__main__':
    H = get_sampler_hparams()
    main(H)