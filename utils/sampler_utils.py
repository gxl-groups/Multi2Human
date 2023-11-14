import os
import torch
from tqdm import tqdm
from .log_utils import save_latents, log
from models import Transformer, AbsorbingDiffusion,Text2ImageTransformer


def get_sampler(H, embedding_weight):

    # denoise_fn = Text2ImageTransformer(H).cuda()
    denoise_fn = Text2ImageTransformer().cuda()
    sampler = AbsorbingDiffusion(
            H, denoise_fn, H.codebook_size, embedding_weight) #  embedding_weight torch.Size([2048, 256])

    return sampler


@torch.no_grad()
def get_samples(H,batch,generator, sampler):

    if H.sampler == "absorbing": #
        latents = sampler.sample11(batch=batch)

    elif H.sampler == "autoregressive":
        latents = sampler.sample11(H.temp)

    latents_one_hot = latent_ids_to_onehot(latents, H.latent_shape, H.codebook_size)
    q = sampler.embed(latents_one_hot)
    images = generator(q.float())

    return images


def get_samples_test(H,batch,generator, sampler):

    if H.sampler == "absorbing": #
        latents = sampler.sample11(batch=batch,filter_ratio = [0])

    elif H.sampler == "autoregressive":
        latents = sampler.sample11(H.temp)

    latents_one_hot = latent_ids_to_onehot(latents, H.latent_shape, H.codebook_size)
    q = sampler.embed(latents_one_hot)
    images = generator(q.float())

    return images
def get_latents_stage1(x,H, generator, sampler):

    latents = x

    latents_one_hot = latent_ids_to_onehot(latents, H.latent_shape, H.codebook_size)
    q = sampler.embed(latents_one_hot)
    images = generator(q.float())

    return images


def latent_ids_to_onehot(latent_ids, latent_shape, codebook_size):
    min_encoding_indices = latent_ids.view(-1).unsqueeze(1)
    encodings = torch.zeros(
        min_encoding_indices.shape[0],
        codebook_size
    ).to(latent_ids.device)
    encodings.scatter_(1, min_encoding_indices, 1)
    one_hot = encodings.view(
        latent_ids.shape[0],
        latent_shape[1],
        latent_shape[2],
        codebook_size
    )
    return one_hot.reshape(one_hot.shape[0], -1, codebook_size)


@torch.no_grad()
def generate_latent_ids(H, ae, train_loader, val_loader=None):
    train_latent_ids = generate_latents_from_loader_single(H, ae, train_loader) # 5168 * 2batch  * (32*16) 离散索引
    if val_loader is not None: #
        val_latent_ids = generate_latents_from_loader_single(H, ae, val_loader)
    else:
        val_latent_ids = None

    # save_latents(H, train_latent_ids, val_latent_ids)
    return train_latent_ids, val_latent_ids

def generate_latent_ids1(H, ae, train_loader,image_fnames):
    generate_latents_from_loader(H, ae,train_loader,image_fnames) # 5168 * 2batch  * (32*16) 离散索引
    #
    # save_latents(H,name,train_latent_ids)

def generate_latents_from_loader(H, autoencoder,dataloader,image_fnames):
    save_dir = "img_tokens/"
    os.makedirs(save_dir, exist_ok=True)
    for i, batch_data in enumerate(dataloader):
        if i==2000000:
            break
        else:
            # x = batch_data['image']
            # x = x.cuda()
            # y = batch_data['img_name'][0].replace('.png','')
            # latents = autoencoder.encoder(x)  # B, emb_dim, H, W  torch.Size([b, 256, 32, 16])
            # latents = latents.permute(0, 2, 3, 1).contiguous()  # B, H, W, emb_dim torch.Size([b, 32, 16, 256])
            # latents_flattened = latents.view(-1, H.emb_dim)  # B*H*W, emb_dim torch.Size([b*512, 256])
            # # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            # distances = (latents_flattened ** 2).sum(dim=1, keepdim=True) + \
            #     (autoencoder.quantize.embedding.weight**2).sum(1) - \
            #     2 * torch.matmul(latents_flattened, autoencoder.quantize.embedding.weight.t())
            # # print('1',autoencoder.quantize.embedding.weight.shape) # torch.Size([2048, 256])
            # # print('2',distances.shape) torch.Size([1024, 2048])
            # min_encoding_indices = torch.argmin(distances, dim=1) # 1024
            # print(i)
            # train_latent_ids = min_encoding_indices.reshape(x.shape[0], -1).cpu().contiguous() # (2,512) batch个离散索引z
            # # print('train_latent_ids',train_latent_ids)
            # train_latents_fp = f"re_img_tokens/{y}"
            # torch.save(train_latent_ids, train_latents_fp)

            x = batch_data[0]
            x = x.cuda()
            y = image_fnames[i].replace('.png', '')
            latents = autoencoder.encoder(x)  # B, emb_dim, H, W  torch.Size([b, 256, 32, 16])
            latents = latents.permute(0, 2, 3, 1).contiguous()  # B, H, W, emb_dim torch.Size([b, 32, 16, 256])
            latents_flattened = latents.view(-1, H.emb_dim)  # B*H*W, emb_dim torch.Size([b*512, 256])
            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            distances = (latents_flattened ** 2).sum(dim=1, keepdim=True) + \
                        (autoencoder.quantize.embedding.weight ** 2).sum(1) - \
                        2 * torch.matmul(latents_flattened, autoencoder.quantize.embedding.weight.t())
            # print('1',autoencoder.quantize.embedding.weight.shape) # torch.Size([2048, 256])
            # print('2',distances.shape) torch.Size([1024, 2048])
            min_encoding_indices = torch.argmin(distances, dim=1)  # 1024
            print(i)
            train_latent_ids = min_encoding_indices.reshape(x.shape[0], -1).cpu().contiguous()  # (2,512) batch个离散索引z
            # print('train_latent_ids',train_latent_ids)
            train_latents_fp = f"img_tokens/{y}"
            # print('是否对应-----------',y,train_latent_ids)
            torch.save(train_latent_ids, train_latents_fp)
def generate_latents_from_loader_single(H, autoencoder, dataloader):
    latent_ids = []
    x= dataloader # 5168
    x = x.cuda()
    latents = autoencoder.encoder(x)  # B, emb_dim, H, W  torch.Size([b, 256, 32, 16])
    latents = latents.permute(0, 2, 3, 1).contiguous()  # B, H, W, emb_dim torch.Size([b, 32, 16, 256])
    latents_flattened = latents.view(-1, H.emb_dim)  # B*H*W, emb_dim torch.Size([b*512, 256])
    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
    distances = (latents_flattened ** 2).sum(dim=1, keepdim=True) + \
            (autoencoder.quantize.embedding.weight**2).sum(1) - \
            2 * torch.matmul(latents_flattened, autoencoder.quantize.embedding.weight.t())
    # print('1',autoencoder.quantize.embedding.weight.shape) # torch.Size([2048, 256])
    # print('2',distances.shape) torch.Size([1024, 2048])
    min_encoding_indices = torch.argmin(distances, dim=1) # 1024
    latent_ids.append(min_encoding_indices.reshape(x.shape[0], -1).cpu().contiguous()) # (2,512) batch个离散索引z
    return torch.cat(latent_ids, dim=0)
@torch.no_grad()
def get_latent_loaders(H, get_validation_loader=True, shuffle=True):
    latents_fp_suffix = "_flipped" if H.horizontal_flip else ""

    train_latents_fp = f"latents/{H.dataset}_{H.latent_shape[-1]}_train_latents{latents_fp_suffix}"
    train_latent_ids = torch.load(train_latents_fp)
    train_latent_loader = torch.utils.data.DataLoader(train_latent_ids, batch_size=H.batch_size, shuffle=shuffle)

    if get_validation_loader:
        val_latents_fp = f"latents/{H.dataset}_{H.latent_shape[-1]}_val_latents{latents_fp_suffix}"
        val_latent_ids = torch.load(val_latents_fp)
        val_latent_loader = torch.utils.data.DataLoader(val_latent_ids, batch_size=H.batch_size, shuffle=shuffle)
    else: #
        val_latent_loader = None

    return train_latent_loader, val_latent_loader


# TODO: rethink this whole thing - completely unnecessarily complicated
def retrieve_autoencoder_components_state_dicts(H, components_list, remove_component_from_key=False):
    state_dict = {}
    # default to loading ema models first
    ae_load_path = f"logs/{H.ae_load_dir}/saved_models/vqgan_ema_{H.ae_load_step}.th"
    if not os.path.exists(ae_load_path):
        ae_load_path = f"logs/{H.ae_load_dir}/saved_models/vqgan_{H.ae_load_step}.th"
    log(f"Loading VQGAN from {ae_load_path}") # logs/vqgan_fashion/saved_models/vqgan_ema_400000.th
    full_vqgan_state_dict = torch.load(ae_load_path, map_location="cpu")
    # print('full_vqgan_state_dict',full_vqgan_state_dict.keys())

    for key in full_vqgan_state_dict:
        # print('key',key)
        for component in components_list:
            if component in key:
                new_key = key[3:]  # remove "ae."
                # print('new_key',new_key)
                if remove_component_from_key:
                    new_key = new_key[len(component)+1:]  # e.g. remove "quantize."

                state_dict[new_key] = full_vqgan_state_dict[key]

    return state_dict
