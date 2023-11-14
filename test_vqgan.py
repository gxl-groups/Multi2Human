import torch
from utils.data_utils import get_data_loaders, cycle
from models import VQAutoEncoder,VQGAN
from hparams import get_sampler_hparams
import torchvision
from utils.sampler_utils import get_sampler, get_samples, retrieve_autoencoder_components_state_dicts
import os
# path = '/home/xswz/data/unleashing-transformers-master/logs/vqgan_fashion/saved_models/vqgan_ema_400000.th'
# checkpoint=torch.load(path)
H = get_sampler_hparams()
# model1 = VQGAN(H)
# model1 = model1.to(torch.device("cuda:0"))
# model1.eval()
os.environ['CUDA_VISIBLE_DEVICES']='1'

train_with_validation_dataset = False
if H.steps_per_eval:
    train_with_validation_dataset = True
ae_state_dict = retrieve_autoencoder_components_state_dicts(
        H, ['encoder', 'quantize', 'generator']
    )
ae = VQAutoEncoder(H)
ae.load_state_dict(ae_state_dict, strict=False)
test_loader, val_loader = get_data_loaders(
        H.dataset,
        H.img_size,
        H.batch_size,
        drop_last=False,
        shuffle=False,
        get_flipped=H.horizontal_flip,
        get_val_dataloader=train_with_validation_dataset
    )

# log("Transferring autoencoder to GPU to generate latents...")
ae = ae.cuda()  # put ae on GPU for generating

test_iterator = cycle(test_loader)
#print('test_iterator',test_iterator)

# log_dir = '/home/user/wz/unleashing-transformers-master/codebooksize/testFID'
log_dir1='/home/user/wz/second_project/Multi2Human/logs/vqgan_fashion/recon_results'
for i in range(0, 1149):
    x = next(test_iterator)  # 2*3*(512,256)
    # print('x',x[0].shape)
    x =x[0].cuda()
    x1, a,b = ae(x)
    c = torch.cat((x,x1),3)
    print(i)
    # torchvision.utils.save_image(torch.clamp(x1, 0, 1), f"{log_dir}/stage1_recon_{i}.png")
    # torchvision.utils.save_image(torch.clamp(x, 0, 1), f"{log_dir}/stage1_origin_{i}.png")
#    torchvision.utils.save_image(torch.clamp(c, 0, 1), f"{log_dir}/stage1_{i}.png")
    torchvision.utils.save_image(torch.clamp(x1, 0, 1), f"{log_dir1}/stage1_{i}.png")
