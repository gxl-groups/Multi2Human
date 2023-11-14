from numpy.core.fromnumeric import mean
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import numpy as np
import copy
import time
import torch.nn.functional as F
from torchvision.utils import save_image

from tqdm import tqdm
from models import VQAutoEncoder, Generator,VQGAN
from models.stage2 import ParsingModel
from models.solver import Solver
from hparams import get_sampler_hparams
from utils.data_utils import get_data_loaders, cycle,DeepFashionAttrSegmDataset,DeepFashionAttrSegmDataset_test
from utils.sampler_utils import generate_latent_ids, generate_latent_ids1,get_latent_loaders, retrieve_autoencoder_components_state_dicts,\
    get_samples, get_sampler
from utils.train_utils import EMA, optim_warmup
from utils.log_utils import log, log_stats, set_up_visdom, config_log, start_training_log, \
    save_stats, load_stats, save_model, load_model, save_images, \
    display_images

#os.environ['CUDA_VISIBLE_DEVICES']='1,0'
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enable =True

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
    #print('1111')
    train_dataset = DeepFashionAttrSegmDataset(
        img_dir='/home/user/wz/second_project/Multi2Human/deepfashion_data/best_img_tokens',
        segm_dir='/home/user/wz/second_project/Multi2Human/deepfashion_data/segm',
        # pose_dir='/home/user/wz/Text2Human-main/datasets/densepose',
        text_dir ='/home/user/wz/second_project/Multi2Human/deepfashion_data/final_prompt/prompt/train_prompt.txt',
        ann_dir='/home/user/wz/second_project/Multi2Human/deepfashion_data/texture_ann/train',
        xflip=True)


    # val_dataset = DeepFashionAttrSegmDataset(
    #     img_dir='/home/user/wz/text-unleashing/re_img_tokens',
    #     segm_dir='/home/user/wz/Text2Human-main/datasets/segm',
    #     # pose_dir='/home/user/wz/Text2Human-main/datasets/densepose',
    #     ann_dir='/home/user/wz/Text2Human-main/datasets/texture_ann/val', )

    # latents_fp_suffix = '_flipped' if H.horizontal_flip else '' # ''
    # latents_filepath = f'latents/{H.dataset}_{H.latent_shape[-1]}_train_latents{latents_fp_suffix}'
    # # latents/Deepfashion_16_train_latents

    train_with_validation_dataset = False
    if H.steps_per_eval: # 0
        train_with_validation_dataset = True


    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=H.batch_size,        )
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
    # sampler = Solver(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)
    optim = torch.optim.Adam(sampler.parameters(), lr=H.lr)

    if H.ema: #
        ema = EMA(H.ema_beta)
        ema_sampler = copy.deepcopy(sampler)

    # initialise before loading so as not to overwrite loaded stats
    losses = np.array([])
    mean_losses = np.array([])
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
    scaler = torch.cuda.amp.GradScaler()
    train_iterator = cycle(train_loader)
    #val_iterator = cycle(val_loader)
    log(f"Sampler params total: {sum(p.numel() for p in sampler.parameters())}")
    colorize = torch.randn(3, 24, 1, 1).cuda()

    for step in range(start_step, H.train_steps):
        step_start_time = time.time()
        # lr warmup
        if H.warmup_iters: # 30k
            if step <= H.warmup_iters:
                optim_warmup(H, step, optim)

        x = next(train_iterator)
        img_z = x['image']
        seg_z = x['segm']
        #  !!img tokens & parsing tokens
        seg_z = seg_z.cuda() # torch.Size([b, 1, 512, 256])
        prompt = x['prompt']

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

        img_tokens = img_z.cuda()  # torch.Size([b,1, 512])
        img_tokens = img_tokens.view(img_tokens.shape[0],-1)
        # print('img_tokens',img_tokens)

        if H.amp: #
            optim.zero_grad()
            with torch.cuda.amp.autocast():
                #!
                stats,text_token = sampler.train_iter([img_tokens,prompt,seg_tokens])
                #print('text_token',text_token.shape)

            scaler.scale(stats['loss']).backward()
            scaler.step(optim)
            scaler.update()
        else:
            stats = sampler.train_iter(x)

            if torch.isnan(stats['loss']).any():
                log(f'Skipping step {step} with NaN loss')
                continue
            optim.zero_grad()
            stats['loss'].backward()
            optim.step()

        losses = np.append(losses, stats['loss'].item())

        if step % H.steps_per_log == 0: #
            step_time_taken = time.time() - step_start_time
            stats['step_time'] = step_time_taken
            mean_loss = np.mean(losses)
            stats['mean_loss'] = mean_loss
            mean_losses = np.append(mean_losses, mean_loss)
            losses = np.array([])

            vis.line(
                np.array([mean_loss]),
                np.array([step]),
                win='loss',
                update=('append' if step > 0 else 'replace'),
                opts=dict(title='Loss')
            )
            log_stats(step, stats)

            # if H.sampler == 'absorbing':
            #     elbo = np.append(elbo, stats['vb_loss'].item())
            #     vis.bar(
            #         sampler.loss_history,
            #         list(range(sampler.loss_history.size(0))),
            #         win='loss_bar',
            #         opts=dict(title='loss_bar')
            #     )
            #     vis.line(
            #         np.array([stats['vb_loss'].item()]),
            #         np.array([step]),
            #         win='ELBO',
            #         update=('append' if step > 0 else 'replace'),
            #         opts=dict(title='ELBO')
            #     )

        if H.ema and step % H.steps_per_update_ema == 0 and step > 0:
            ema.update_model_average(ema_sampler, sampler)

        # images = None
        # print('okokok!!!!!!!!!!')
        if step % H.steps_per_display_output == 0 and step > 0:
        #if step % 10 == 0 and step > 0:
            # ！！！！Reverse阶段
            ema_sampler.eval()
            images = get_samples(H,[img_tokens,text_token,seg_tokens], generator, ema_sampler if H.ema else sampler)
            # display_images(vis, images, H, win_name=f'{H.sampler}_samples')
            save_images(images, 'stage2_recon', step, H.log_dir, H.save_individually)
            print('save111', images.shape)
            ema_sampler.train()

        # # if step % H.steps_per_save_output == 0 and step > 0:
        # if step == 0:
        #     if images is None:
        #         ema_sampler.eval()
        #         # ！！！！Reverse阶段
        #         images = get_samples(H,[img_tokens,seg_tokens], generator, ema_sampler if H.ema else sampler)
        #         print('save111', images.shape)
        #     save_images(images, 'samples', step, H.log_dir, H.save_individually)
        #     ema_sampler.train()




        # if H.steps_per_eval and step % H.steps_per_eval == 0 and step > 0:
        #     # calculate validation loss
        #     valid_loss, valid_elbo, num_samples = 0.0, 0.0, 0
        #     eval_repeats = 5
        #     log("Evaluating")
        #     for _ in tqdm(range(eval_repeats)):
        #         for x in val_iterator:
        #             with torch.no_grad():
        #                 stats = sampler.train_iter(x.cuda())
        #                 valid_loss += stats['loss'].item()
        #                 if H.sampler == 'absorbing': #
        #                     valid_elbo += stats['vb_loss'].item()
        #                 num_samples += x.size(0)
        #     valid_loss = valid_loss / num_samples
        #     if H.sampler == 'absorbing': #
        #         valid_elbo = valid_elbo / num_samples
        #
        #     val_losses = np.append(val_losses, valid_loss)
        #     val_elbos = np.append(val_elbos, valid_elbo)
        #     vis.line(
        #         np.array([valid_loss]),
        #         np.array([step]),
        #         win='Val_loss',
        #         update=('append' if step > 0 else 'replace'),
        #         opts=dict(title='Validation Loss')
        #     )
        #     if H.sampler == 'absorbing': #
        #         vis.line(
        #             np.array([valid_elbo]),
        #             np.array([step]),
        #             win='Val_elbo',
        #             update=('append' if step > 0 else 'replace'),
        #             opts=dict(title='Validation ELBO')
        #         )

        if step % H.steps_per_checkpoint == 0 and step > H.load_step:
            save_model(sampler, H.sampler, step, H.log_dir)
            save_model(optim, f'{H.sampler}_optim', step, H.log_dir)

            if H.ema:
                save_model(ema_sampler, f'{H.sampler}_ema', step, H.log_dir)

            train_stats = {
                'losses': losses,
                'mean_losses': mean_losses,
                'steps_per_log': H.steps_per_log,
                'steps_per_eval': H.steps_per_eval,
            }
            save_stats(H, train_stats, step)


if __name__ == '__main__':
    H = get_sampler_hparams()
    vis = set_up_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')
    start_training_log(H)
    main(H, vis)
