from numpy.core.fromnumeric import mean
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
import numpy as np
import copy
import time
import torch.nn.functional as F
from torchvision.utils import save_image
import random
from tqdm import tqdm
from models import VQAutoEncoder, Generator,VQGAN
from models.stage2 import ParsingModel
from models.solver import Solver
from models.tokenize import Tokenize
from hparams import get_sampler_hparams
from utils.data_utils import get_data_loaders, cycle,DeepFashionAttrSegmDataset,DeepFashionAttrSegmDataset_test
from utils.sampler_utils import generate_latent_ids, generate_latent_ids1,get_latent_loaders, retrieve_autoencoder_components_state_dicts,\
    get_samples, get_sampler,get_samples_test
from utils.train_utils import EMA, optim_warmup
from utils.log_utils import log, log_stats, set_up_visdom, config_log, start_training_log, \
    save_stats, load_stats, save_model, load_model, save_images,save_images_test, \
    display_images


# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enable =True
def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(H, vis):
    # stage1保存训练的img_token
    # latents_filepath = f'img_tokens'
    # train_with_validation_dataset = False
    # # if H.steps_per_eval:  # 0
    # #     train_with_validation_dataset = True
    # if not os.path.exists(latents_filepath):  # load VQGAN
    #     ae_state_dict = retrieve_autoencoder_components_state_dicts(
    #         H, ['encoder', 'quantize', 'generator']
    #     )
    #     ae = VQAutoEncoder(H)
    #     ae.load_state_dict(ae_state_dict, strict=False)
    #     # val_loader will be assigned to None if not training with validation dataest
    #     train_loader,image_fnames = get_data_loaders(
    #         H.dataset,
    #         H.img_size,
    #         H.batch_size,
    #         drop_last=False,
    #         shuffle=False,
    #         get_flipped=H.horizontal_flip,
    #         get_val_dataloader=train_with_validation_dataset
    #     )
    #
    #     log("Transferring autoencoder to GPU to generate latents...")
    #     ae = ae.cuda()  # put ae on GPU for generating
    #     # !!
    #     generate_latent_ids1(H, ae, train_loader,image_fnames)
    #     log("Deleting autoencoder to conserve GPU memory...")
    #     ae = ae.cpu()
    #     ae = None
    #
    #     print('finished!!!!')

    # random seed
#    seed = 2021

    test_dataset = DeepFashionAttrSegmDataset_test(
        img_dir='/home/user/wz/a_new_project/WAVE_aug+StageII/new_b4_attnx1-parsingQ/best_img_tokens',
        segm_dir='/home/user/wz/a_new_project/Text2Human-main/datasets/segm',
        # pose_dir='/home/user/wz/Text2Human-main/datasets/densepose',
#        text_dir ='/home/user/wz/a_new_project/final_prompt/prompt/test_prompt.txt',
        text_dir='/home/user/wz/a_new_project/final_prompt/prompt/test_prompt.txt',
        ann_dir='/home/user/wz/a_new_project/Text2Human-main/datasets/texture_ann/test',
        xflip=False)

    # val_dataset = DeepFashionAttrSegmDataset(
    #     img_dir='/home/user/wz/parsing-unleashing/re_img_tokens',
    #     segm_dir='/home/user/wz/Text2Human-main/datasets/segm',
    #     # pose_dir='/home/user/wz/Text2Human-main/datasets/densepose',
    #     ann_dir='/home/user/wz/Text2Human-main/datasets/texture_ann/val', )

    # latents_fp_suffix = '_flipped' if H.horizontal_flip else '' # ''
    # latents_filepath = f'latents/{H.dataset}_{H.latent_shape[-1]}_train_latents{latents_fp_suffix}'
    # # latents/Deepfashion_16_train_latents



    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=H.batch_size     )
    # val_loader = torch.utils.data.DataLoader(
    #         dataset=val_dataset, batch_size=H.batch_size, shuffle=True)


    #     # load VQGAN
    # ae_state_dict = retrieve_autoencoder_components_state_dicts(
    #         H, ['encoder', 'quantize', 'generator']
    #     )
    # ae = VQAutoEncoder(H)
    # ae.load_state_dict(ae_state_dict, strict=False)
    # ae = ae.cuda()  # put ae on GPU for generating

    # train_latent_loader, val_latent_loader = get_latent_loaders(H, get_validation_loader=train_with_validation_dataset)
    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['quantize', 'generator'],
        remove_component_from_key=True
    )
    embedding_weight = quanitzer_and_generator_state_dict.pop(
        'embedding.weight') # torch.Size([2048, 256])

    if H.deepspeed:
        embedding_weight = embedding_weight.half()
    embedding_weight = embedding_weight.cuda()
    generator = Generator(H)

    generator.load_state_dict(quanitzer_and_generator_state_dict, strict=False)
    generator = generator.cuda()
    # # ! sampler  吸收扩散+Transformer
    # embedding_weight = None
    sampler = get_sampler(H, embedding_weight).cuda()
    # sampler = get_sampler(H, embedding_weight).cuda()
    sampler = load_model(sampler, f'{H.sampler}_ema', H.load_step, H.load_dir)
    #sampler.n_samples = 5  # get samples in 5x5 grid#
    # sampler = Solver(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)


    # initialise before loading so as not to overwrite loaded stats

    start_step = 0
    log_start_step = 0
    # if H.load_step > 0:
    #     start_step = H.load_step + 1
    #
    #     sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir).cuda()
    #     if H.ema: #
    #         # if EMA has not been generated previously, recopy newly loaded model
    #         try:
    #             ema_sampler = load_model(
    #                 ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir)
    #         except Exception:
    #             ema_sampler = copy.deepcopy(sampler)
    #     if H.load_optim:
    #         optim = load_model(
    #             optim, f'{H.sampler}_optim', H.load_step, H.load_dir)
    #         # only used when changing learning rates and reloading from checkpoint
    #         for param_group in optim.param_groups:
    #             param_group['lr'] = H.lr
    #
    #     try:
    #         train_stats = load_stats(H, H.load_step)
    #     except Exception:
    #         train_stats = None
    #
    #     if train_stats is not None:
    #         losses, mean_losses, val_losses, elbo, H.steps_per_log
    #
    #         losses = train_stats["losses"],
    #         mean_losses = train_stats["mean_losses"],
    #         val_losses = train_stats["val_losses"],
    #         val_elbos = train_stats["val_elbos"]
    #         elbo = train_stats["elbo"],
    #         H.steps_per_log = train_stats["steps_per_log"]
    #         log_start_step = 0
    #
    #         losses = losses[0]
    #         mean_losses = mean_losses[0]
    #         val_losses = val_losses[0]
    #         val_elbos = val_elbos[0]
    #         elbo = elbo[0]
    #
    #         # initialise plots
    #         vis.line(
    #             mean_losses,
    #             list(range(log_start_step, start_step, H.steps_per_log)),
    #             win='loss',
    #             opts=dict(title='Loss')
    #         )
    #         vis.line(
    #             elbo,
    #             list(range(log_start_step, start_step, H.steps_per_log)),
    #             win='ELBO',
    #             opts=dict(title='ELBO')
    #         )
    #         vis.line(
    #             val_losses,
    #             list(range(H.steps_per_eval, start_step, H.steps_per_eval)),
    #             win='Val_loss',
    #             opts=dict(title='Validation Loss')
    #         )
    #     else:
    #         log('No stats file found for loaded model, displaying stats from load step only.')
    #         log_start_step = start_step
    # scaler = torch.cuda.amp.GradScaler()
    test_iterator = cycle(test_loader)
    colorize = torch.randn(3, 24, 1, 1).cuda()

    for step in range(start_step, 1149):
        step_start_time = time.time()
        # lr warmup
        # if H.warmup_iters: # 30k
        #     if step <= H.warmup_iters:
        #         optim_warmup(H, step, optim)

        x = next(test_iterator)
        # img_z = x['image']
        seg_z = x['segm']
        #  !!img tokens & parsing tokens
        seg_z = seg_z.cuda() # torch.Size([b, 1, 512, 256])
        name = x['img_name']
        prompt = x['prompt']
        
        text = Tokenize().get_tokens(prompt)
        text = text['token']

        def to_rgb(x):
            x = F.conv2d(x, weight=colorize)
            x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
            return x

        recon_segm,seg_tokens = ParsingModel().feed_data(seg_z) # # T_seg b*512 （同样是VQ的索引） #  torch.Size([1, 512])
        # output recon_segm
        # xrec = torch.argmax(recon_segm, dim=1, keepdim=True)
        # xrec = F.one_hot(xrec, num_classes=24)
        # xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
        # xrec = to_rgb(xrec)
        #
        # img_cat =  xrec.detach()
        # img_cat = ((img_cat + 1) / 2)
        # img_cat = img_cat.clamp_(0, 1)
        # save_image(img_cat, f'logs/{step}.png', nrow=1, padding=4)
        # print('save!!!!')

        n_sample = 1
        img_tokens = torch.zeros(H.batch_size,512)
        #images_list=[]
        for i in range(n_sample):
            image = get_samples_test(H,[img_tokens,text,seg_tokens], generator, sampler)
            # images_list.append(image)
        # display_images(vis, images, H, win_name=f'{H.sampler}_samples')
#            save_images_test(f'{name[0][:-4]}_{i}.png',image, 'sample_attn2_475', step, H.log_dir, H.save_individually)
            save_images_test(f'{name[0][:-4]}.png', image, 'sampling_results', step, H.log_dir, H.save_individually)
            print('finished',step,i)
        print('save111', step)


if __name__ == '__main__':
    H = get_sampler_hparams()
    vis = set_up_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')
    start_training_log(H)
    main(H, vis)
