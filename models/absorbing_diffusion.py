import math
import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from tqdm import tqdm
from .sampler import Sampler
from torch.cuda.amp import autocast
from .transformer_utils import Text2ImageTransformer,CrossAttention
from .dalle_mask_image_embedding import DalleMask_ImageEmbedding
from .tokenize import Tokenize
from .clip_text_embedding import CLIPTextEmbedding
from torch import nn

eps = 1e-8

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes) # torch.Size([b, 512, 18432])
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order) #  torch.Size([b, 18432, 512])
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

def alpha_schedule(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999):
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1
    # print('att',att)
    att = np.concatenate(([1], att))
    at = att[1:]/att[:-1]
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct
    bt = (1-at-ct)/N
    att = np.concatenate((att[1:], [1]))  # at_
    ctt = np.concatenate((ctt[1:], [0])) # rt_
    btt = (1-att-ctt)/N # bt_
    return at, bt, ct, att, btt, ctt


class AbsorbingDiffusion(Sampler):
    def __init__(self, H, denoise_fn, mask_id, embedding_weight, aux_weight=0.01):
        super().__init__(H, embedding_weight=embedding_weight)

        self.num_classes = H.codebook_size+1 # 18433
        self.latent_emb_dim = H.emb_dim # 256
        self.shape = tuple(H.latent_shape) #latent_shape: [1, 32, 16]
        self.shape1 = 512
        self.num_timesteps =100 # H.total_steps # 256

        self.mask_id = mask_id # codebook_size 2048
        self._denoise_fn = denoise_fn # Transformer
        self.n_samples = H.batch_size # 2
        self.mask_schedule = H.mask_schedule # random
        self.aux_weight = aux_weight
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps+1))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps+1))
        self.register_buffer('loss_history', torch.zeros(self.num_timesteps+1))
        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        self.mask_weight = [1, 1]
        self.auxiliary_loss_weight = 0
        self.adaptive_auxiliary_loss = True
        self.transformer  = Text2ImageTransformer()
        # at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=self.num_classes-1) 原代码为什么-1
        at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=self.num_classes-1)
        # 100,100,100,101,101,101

        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct) # log(1-rt)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct) # log(1-rt_)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())
        self.content_seq_len = 512

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))
        self.zero_vector = None

        assert self.mask_schedule in ['random', 'fixed']

        self.loss_type = 'vb_stochastic'
        self.content_info ={'key': 'image'}
        self.condition_info = {'key': 'text'}
        self.parsing_emb = DalleMask_ImageEmbedding()
        self.condition_codec = Tokenize()
        self.condition_emb = CLIPTextEmbedding()
        self.prior_ps = 1024
        self.prior_rule = 0
        self.prior_weight = 0
        self.n_sample = [1, 10] + [11, 10, 10] * 32 + [11, 11]
        self.parametrization = 'x0'
        self.shared_conv = nn.Linear(77, 512)

        self.attn_fusion = CrossAttention(
            condition_seq_len = '',
            n_embd=512,
            condition_embd=512,
            n_head=16,
            seq_len=512,
            attn_pdrop=0,
            resid_pdrop=0,
        )
        self.attn_fusion1 = CrossAttention(
            condition_seq_len='',
            n_embd=512,
            condition_embd=512,
            n_head=16,
            seq_len=512,
            attn_pdrop=0,
            resid_pdrop=0,
        )
        self.attn_fusion2 = CrossAttention(
            condition_seq_len='',
            n_embd=512,
            condition_embd=512,
            n_head=16,
            seq_len=512,
            attn_pdrop=0,
            resid_pdrop=0,
        )
        self.attn_fusion3 = CrossAttention(
            condition_seq_len='',
            n_embd=512,
            condition_embd=512,
            n_head=16,
            seq_len=512,
            attn_pdrop=0,
            resid_pdrop=0,
        )
        self.attn_fusion4 = CrossAttention(
            condition_seq_len='',
            n_embd=512,
            condition_embd=512,
            n_head=16,
            seq_len=512,
            attn_pdrop=0,
            resid_pdrop=0,
        )
        self.attn_fusion5 = CrossAttention(
            condition_seq_len='',
            n_embd=512,
            condition_embd=512,
            n_head=16,
            seq_len=512,
            attn_pdrop=0,
            resid_pdrop=0,
        )
        self.attn_fusion6 = CrossAttention(
            condition_seq_len='',
            n_embd=512,
            condition_embd=512,
            n_head=16,
            seq_len=512,
            attn_pdrop=0,
            resid_pdrop=0,
        )
        self.attn_fusion7 = CrossAttention(
            condition_seq_len='',
            n_embd=512,
            condition_embd=512,
            n_head=16,
            seq_len=512,
            attn_pdrop=0,
            resid_pdrop=0,
        )
        self.attn_fusion8 = CrossAttention(
            condition_seq_len='',
            n_embd=512,
            condition_embd=512,
            n_head=16,
            seq_len=512,
            attn_pdrop=0,
            resid_pdrop=0,
        )
        self.attn_fusion9 = CrossAttention(
            condition_seq_len='',
            n_embd=512,
            condition_embd=512,
            n_head=16,
            seq_len=512,
            attn_pdrop=0,
            resid_pdrop=0,
        )

    def sample_time(self, b, device, method='uniform'): #
        if method == 'importance':#
            if not (self.Lt_count > 10).all(): # torch.zeros(self.num_timesteps)
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform': #
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def q_sample_mlm(self, x_0, t):
        # samples q(x_t | x_0)
        # fixed noise schedule, masks exactly int(t/T * latent_size) tokens
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.zeros_like(x_t).to(torch.bool)

        # TODO: offset so each n_masked_tokens is picked with equal probability
        n_masked_tokens = (t.float() / self.num_timesteps) * x_t.size(1)
        n_masked_tokens = torch.round(n_masked_tokens).to(torch.int64)
        n_masked_tokens[n_masked_tokens == 0] = 1
        ones = torch.ones_like(mask[0]).to(torch.bool).to(x_0.device)

        for idx, n_tokens_to_mask in enumerate(n_masked_tokens):
            index = torch.randperm(x_0.size(1))[:n_tokens_to_mask].to(x_0.device)
            mask[idx].scatter_(dim=0, index=index, src=ones)

        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask


    def multinomial_kl(self, log_prob1, log_prob2): #  # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred(self, log_x_start, t):#  # q(xt|x0)
        # log_x_start can be onehot or not
        t =(t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)  # at~ torch.Size([b, 1, 1])
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)  # bt~ torch.Size([b, 1, 1])
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~ torch.Size([b, 1, 1])
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)  # 1-ct~

        log_probs = torch.cat([log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at, log_cumprod_bt),
             log_add_exp(log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct)
            ],
            dim=1
        )

        return log_probs

    def q_pred_one_timestep(self, log_x_t, t): #        # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:,:-1,:]+log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )

        return log_probs

    def q_posterior(self, log_x_start, log_x_t, t): # # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.content_seq_len)

        log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:,:-1,:]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes-1, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector


        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct
        ct_vector = log_ct.expand(-1, self.num_classes-1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector

        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:,:-1,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def predict_start(self, log_x_t, cond_emb, t):  #        # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t) # .argmax(1)
        out = self._denoise_fn(x_t, cond_emb, t) # # torch.Size([1,18432,512]) # Text2ImageTransformer
        # print('56765',x_t.shape,out.shape,cond_emb.shape)
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-1
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out.double(), dim=1).float()

        batch_size = log_x_t.size()[0]
        if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:  #
            self.zero_vector = torch.zeros(batch_size, 1, self.content_seq_len).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)  # torch.Size([1, 18433, 512])
        log_pred = torch.clamp(log_pred, -70, 0) # torch.Size([1, 18433, 512])

        return log_pred

    def train_iter(self, x):
        img_token,prompt,seg_token = x[0],x[1],x[2]

        # parsing_emb
        cond_emb_seg = self.parsing_emb(seg_token) # torch.Size([4, 512, 512])  # parsing_emb

        # text_emb
        text_token = self.condition_codec.get_tokens(prompt) # text_token['mask'], text_token['token'] b*77
        with autocast(enabled=False):
            with torch.no_grad():
                # input['condition_token'] b*77
                cond_emb_text = self.condition_emb(text_token['token'].cuda())
                cond_emb_text = cond_emb_text.float() # b*77*512
                #cond_emb_text = cond_emb_text  # b*77*512
                # b = torch.zeros(cond_emb_text.shape[0],cond_emb_seg.shape[1] - cond_emb_text.shape[1], 512)
                # print('111',cond_emb_seg.shape[1])
                # cond_emb_text1 = torch.cat([cond_emb_text,b.cuda()],dim=1) # text_emb
                cond_emb_text = cond_emb_text.permute(0,2,1)
        cond_emb_text1 = self.shared_conv(cond_emb_text).permute(0,2,1)
        
        # 多模态融合
        cond_emb,_ = self.attn_fusion(cond_emb_seg,cond_emb_text1) #q\k\v torch.Size([4, 512, 512])
        # only text
        #cond_emb = cond_emb_text1 #([4, 512, 512])


        log_model_prob, loss = self._train_loss(img_token,cond_emb)
        loss = loss.sum() / (img_token.size()[0] * img_token.size()[1])
        out = {'loss': loss,'logits':torch.exp(log_model_prob)}
        return out,text_token['token'].cuda()


    def _train_loss(self, x, cond_emb, is_train=True):  # get the KL loss
        b, device = x.size(0), x.device
        assert self.loss_type == 'vb_stochastic'
        # x0
        x_start = x  # b*512
        t, pt = self.sample_time(b, device, 'importance') # b个t
        # log_x0
        log_x_start = index_to_log_onehot(x_start, self.num_classes)  #  torch.Size([b, 18433, 512])
        # 加噪 log_xt
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)  # xt #  torch.Size([1, 18433, 512])
        # xt
        xt = log_onehot_to_index(log_xt)  # .argmax(1)

        ############### go to p_theta function ###############
        # P_theta(x0|xt,y)
        #recon_logx0
        log_x0_recon = self.predict_start(log_xt, cond_emb, t=t)  #  torch.Size([1, 18433, 512])

        # go through q(xt_1|xt,x0)
        # log_xt_1
        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t) # torch.Size([1, 18433, 512])

        ################## compute acc list ################
        # recon_x0
        x0_recon = log_onehot_to_index(log_x0_recon)
        # x0
        x0_real = x_start
        # recon_xt_1
        xt_1_recon = log_onehot_to_index(log_model_prob)
        # recon_xt
        xt_recon = log_onehot_to_index(log_xt)
        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (x0_recon[index] == x0_real[index]).sum().cpu() / x0_real.size()[1]  # acc
            # print('rate',same_rate)
            self.diffusion_acc_list[this_t] = same_rate.item() * 0.1 + self.diffusion_acc_list[this_t] * 0.9
            same_rate = (xt_1_recon[index] == xt_recon[index]).sum().cpu() / xt_recon.size()[1]
            self.diffusion_keep_list[this_t] = same_rate.item() * 0.1 + self.diffusion_keep_list[this_t] * 0.9

        # compute log_true_prob now
        #  # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))   # log_x0 log_xt
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        kl = self.multinomial_kl(log_true_prob, log_model_prob)  # KL散度
        mask_region = (xt == self.num_classes - 1).float()
        mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        kl = kl * mask_weight
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1. - mask) * kl  # loss

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0 and is_train == True:  #
            kl_aux = self.multinomial_kl(log_x_start[:, :-1, :], log_x0_recon[:, :-1, :])
            kl_aux = kl_aux * mask_weight
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:  #
                addition_loss_weight = (1 - t / self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss += loss2

        return log_model_prob, vb_loss

    # def sample(self, temp=1.0, sample_steps=None):
    #     b, device = self.n_samples, 'cuda'
    #     x_t = torch.ones((b, np.prod(self.shape)), device=device).long() * self.mask_id
    #     unmasked = torch.zeros_like(x_t, device=device).bool()
    #     sample_steps = list(range(1, sample_steps+1))
    #
    #     for t in reversed(sample_steps): # backwards
    #         print(f'Sample timestep {t:4d}', end='\r')
    #         t = torch.full((b,), t, device=device, dtype=torch.long)
    #
    #         # where to unmask
    #         changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1)
    #         # don't unmask somewhere already unmasked
    #         changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
    #         # update mask with changes
    #         unmasked = torch.bitwise_or(unmasked, changes)
    #
    #         x_0_logits = self._denoise_fn(x_t, t=t) # transformer
    #         # scale by temperature
    #         x_0_logits = x_0_logits / temp   # 温度temp取值好像和多样性有关
    #         x_0_dist = dists.Categorical(
    #             logits=x_0_logits)
    #         x_0_hat = x_0_dist.sample().long()
    #         x_t[changes] = x_0_hat[changes]
    #
    #     return x_t

    @torch.no_grad()
    def prepare_condition(self, batch, condition=None):
        condition_token = batch
        context_length = 77 # 512
        # if torch.is_tensor(condition_token):
        #     condition_token = condition_token.to(self.device)
        condition_mask = torch.zeros(len(condition_token), context_length, dtype=torch.bool)
        return condition_token,condition_mask

    @autocast(enabled=False)
    @torch.no_grad()
    def prepare_content(self, batch, with_mask=False):
        content_token = batch
        content_length = 512
        # if torch.is_tensor(content_token):
        #     content_token = content_token.to(self.device)
        content_mask = torch.zeros(len(content_token), content_length, dtype=torch.bool)
        return content_token,content_mask

    # def update_n_sample(self):
    #     if self.num_timesteps == 100:
    #         if self.prior_ps <= 10:
    #             self.n_sample = [1, 6] + [11, 10, 10] * 32 + [11, 15]
    #         else:
    #             self.n_sample = [1, 10] + [11, 10, 10] * 32 + [11, 11]
    #     elif self.num_timesteps == 50:
    #         self.n_sample = [10] + [21, 20] * 24 + [30]
    #     elif self.num_timesteps == 25:
    #         self.n_sample = [21] + [41] * 23 + [60]
    #     elif self.num_timesteps == 10:
    #         self.n_sample = [69] + [102] * 8 + [139]

    def cf_predict_start(self, log_x_t, cond_emb, t):
        return self.predict_start(log_x_t, cond_emb, t)

    def p_pred(self, log_x, cond_emb, t):             # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
        if self.parametrization == 'x0':
            log_x_recon = self.cf_predict_start(log_x, cond_emb, t)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, cond_emb, t)
        else:
            raise ValueError
        return log_model_pred, log_x_recon

    @torch.no_grad()
    def p_sample(self, log_x, cond_emb, t, sampled=None,
                 to_sample=None):  # sample q(xt-1) for next step from  xt, actually is p(xt-1|xt)
        model_log_prob, log_x_recon = self.p_pred(log_x, cond_emb, t)

        max_sample_per_step = self.prior_ps  # max number to sample per step
        if t[0] > 0 and self.prior_rule > 0 and to_sample is not None:  # prior_rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
            log_x_idx = log_onehot_to_index(log_x)

            if self.prior_rule == 1:
                score = torch.ones((log_x.shape[0], log_x.shape[2])).to(log_x.device)
            elif self.prior_rule == 2:
                score = torch.exp(log_x_recon).max(dim=1).values.clamp(0, 1)
                score /= (score.max(dim=1, keepdim=True).values + 1e-10)

            if self.prior_rule != 1 and self.prior_weight > 0:
                # probability adjust parameter, prior_weight: 'r' in Equation.11 of Improved VQ-Diffusion
                prob = ((1 + score * self.prior_weight).unsqueeze(1) * log_x_recon).softmax(dim=1)
                prob = prob.log().clamp(-70, 0)
            else:
                prob = log_x_recon

            out = self.log_sample_categorical(prob)
            out_idx = log_onehot_to_index(out)

            out2_idx = log_x_idx.clone()
            _score = score.clone()
            if _score.sum() < 1e-6:
                _score += 1
            _score[log_x_idx != self.num_classes - 1] = 0

            for i in range(log_x.shape[0]):
                n_sample = min(to_sample - sampled[i], max_sample_per_step)
                if to_sample - sampled[i] - n_sample == 1:
                    n_sample = to_sample - sampled[i]
                if n_sample <= 0:
                    continue
                sel = torch.multinomial(_score[i], n_sample)
                out2_idx[i][sel] = out_idx[i][sel]
                sampled[i] += ((out2_idx[i] != self.num_classes - 1).sum() - (
                            log_x_idx[i] != self.num_classes - 1).sum()).item()

            out = index_to_log_onehot(out2_idx, self.num_classes)
        else: #
            # Gumbel sample
            out = self.log_sample_categorical(model_log_prob)
            sampled = [1024] * log_x.shape[0]

        if to_sample is not None:
            return out, sampled
        else:
            return out

    def log_sample_categorical(self, logits):  # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):  # diffusion step, q(xt|x0) and sample xt
        # q(xt|x0)
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)  # torch.Size([2, 2888, 1024]) logits
        # sample xt
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    @torch.no_grad()
    def sample11(
            self,
            batch,
            clip=None,
            temperature=1.,
            return_rec=True,
            filter_ratio=[0, 0.5, 1.0],
            content_ratio=[1],  # the ratio to keep the encoded content tokens
            return_att_weight=False,
            return_logits=False,
            sample_type="normal"
    ):
        # self.eval()
        x,text,parsing = batch[0],batch[1],batch[2]

        # 训练时需要注释掉
#        text = self.condition_codec.get_tokens(text)
#        text = text['token']

        z = self.parsing_emb(parsing)


        content_token,content_mask = self.prepare_content(x)
        condition_token, condition_mask = self.prepare_condition(text.cuda())  #

        # content_samples = {'input_image': batch[self.content_info['key']]}
        # if return_rec:
        #     content_samples['reconstruction_image'] = self.content_codec.decode(content_token)

        for fr in filter_ratio:
            for cr in content_ratio:
                num_content_tokens = int((content_token.shape[1] * cr))
                if num_content_tokens < 0:
                    continue
                else: # [0, 0.5, 1.0]
                    content_token = content_token[:, :num_content_tokens]
                trans_out = self.sample1(condition_token=condition_token,
                                         parsing_emb = z,
                                                              condition_mask=condition_mask,
                                                              condition_embed=None,
                                                              content_token=content_token,
                                                              filter_ratio=fr,
                                                              temperature=temperature,
                                                              return_att_weight=return_att_weight,
                                                              return_logits=return_logits,
                                                              content_logits=None)
                # content_samples['cond1_cont{}_fr{}_image'.format(cr, fr)] = self.content_codec.decode(trans_out['content_token'])

        # self.train()
        return trans_out





    def sample1(
            self,
            condition_token,
            parsing_emb,
            condition_mask,
            condition_embed,
            content_token=None,
            filter_ratio=0.5, # [0, 0.5, 1.0] 实际上用的是1
            temperature=1.0,
            return_att_weight=False,
            return_logits=False,
            content_logits=None,
            print_log=True):
        input = {'condition_token': condition_token,
                 'content_token': content_token,
                 'condition_mask': condition_mask,
                 'condition_embed_token': condition_embed, # None,
                 'content_logits': content_logits, # None,
                 'parsing_emb':parsing_emb
                 }

        batch_size = input['condition_token'].shape[0]

        device = self.log_at.device
        start_step = int(self.num_timesteps * filter_ratio) # 0,50,100

        # get cont_emb and cond_emb
        if content_token != None: #
            sample_image = input['content_token'].type_as(input['content_token'])

        if self.condition_emb is not None:  # do this
            with torch.no_grad():
                cond_emb1 = input['parsing_emb']  # B x Ld x D   #256*1024
                cond_emb2 = self.condition_emb(input['condition_token']).float()  # B x Ld x D   #256*1024
                # b = torch.zeros(cond_emb2.shape[0], 512-77, 512)
                # cond_emb_text = torch.cat([cond_emb2, b.cuda()], dim=1).float()
                cond_emb2 = cond_emb2.permute(0, 2, 1)
            cond_emb_text = self.shared_conv(cond_emb2).permute(0, 2, 1)
            #cond_emb = cond_emb_text+cond_emb1
            cond_emb, _ = self.attn_fusion(cond_emb1,cond_emb_text)  # q\k\vtorch.Size([4, 512, 512])
            # cond_emb, _ = self.attn_fusion1(cond_emb_text, cond_emb)
            # cond_emb, _ = self.attn_fusion2(cond_emb_text, cond_emb)
            # cond_emb, _ = self.attn_fusion3(cond_emb_text, cond_emb)
            # cond_emb, _ = self.attn_fusion4(cond_emb_text, cond_emb)
            #
            # cond_emb, _ = self.attn_fusion5(cond_emb_text, cond_emb)
            # cond_emb, _ = self.attn_fusion6(cond_emb_text, cond_emb)
            # cond_emb, _ = self.attn_fusion7(cond_emb_text, cond_emb)
            # cond_emb, _ = self.attn_fusion8(cond_emb_text, cond_emb)
            # cond_emb, _ = self.attn_fusion9(cond_emb_text, cond_emb)
        else:  # share condition embeding with content
            if input.get('condition_embed_token', None) != None:
                cond_emb = input['condition_embed_token'].float()
            else:
                cond_emb = None

        # 三种sample策略
        #1,从X_T开始采样得到x_0_hat
        if start_step == 0: #
            # use full mask sample
            zero_logits = torch.zeros((batch_size, self.num_classes - 1, self.shape1), device=device)
            one_logits = torch.ones((batch_size, 1, self.shape1), device=device)
            mask_logits = torch.cat((zero_logits, one_logits), dim=1)
            log_z = torch.log(mask_logits)
            start_step = self.num_timesteps # 100
            with torch.no_grad():
                for diffusion_index in range(start_step - 1, -1, -1): # 99->0
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long) # t = 99->0
                    sampled = [0] * log_z.shape[0] #[0, 0,..., 0] b个
                    while min(sampled) < self.n_sample[diffusion_index]:
                        log_z, sampled = self.p_sample(log_z, cond_emb, t, sampled,
                                                       self.n_sample[diffusion_index])  # log_z is log_onehot p(xt-1|xt)
                    # 逆向100次得到预测的recon_img_token

        # 2,从X_t开始采样，t=50/100 （）
        else: #
            t = torch.full((batch_size,), start_step - 1, device=device, dtype=torch.long)
            log_x_start = index_to_log_onehot(sample_image, self.num_classes)
            # 加噪到x_t
            log_xt = self.q_sample(log_x_start=log_x_start, t=t)
            log_z = log_xt
            with torch.no_grad():
                for diffusion_index in range(start_step - 1, -1, -1): # 0～49
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                    log_z = self.p_sample(log_z, cond_emb, t)  # log_z is log_onehot p(xt-1|xt)
                    # # 逆向50/100次得到预测的recon_img_token

        content_token = log_onehot_to_index(log_z)

        # output = {'content_token': content_token}
        # if return_logits:
        #     output['logits'] = torch.exp(log_z)
        return content_token


    def sample_mlm(self, temp=1.0, sample_steps=None):
        b, device = self.n_samples, 'cuda'
        x_0 = torch.ones((b, np.prod(self.shape)), device=device).long() * self.mask_id
        sample_steps = np.linspace(1, self.num_timesteps, num=sample_steps).astype(np.long)

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, _, _ = self.q_sample(x_0, t)
            x_0_logits = self._denoise_fn(x_t, t=t)
            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(
                logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_0[x_t == self.mask_id] = x_0_hat[x_t == self.mask_id]

        return x_0

    @torch.no_grad()
    def elbo(self, x_0):
        b, device = x_0.size(0), x_0.device
        elbo = 0.0
        for t in reversed(list(range(1, self.num_timesteps+1))):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, x_0_ignore, _ = self.q_sample(x_0=x_0, t=t)
            x_0_hat_logits = self._denoise_fn(x_t, t=t).permute(0, 2, 1)
            cross_entropy_loss = F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none').sum(1)
            elbo += cross_entropy_loss / t
        return elbo

    def sample_shape(self, shape, num_samples, time_steps=1000, step=1, temp=0.8):
        device = 'cuda'
        x_t = torch.ones((num_samples,) + shape, device=device).long() * self.mask_id # 8,32,16  2048
        x_lim, y_lim = shape[0] - self.shape[1], shape[1] - self.shape[2] # 0,0

        unmasked = torch.zeros_like(x_t, device=device).bool() # 8,32,16  0

        autoregressive_step = 0
        for t in tqdm(list(reversed(list(range(1, time_steps+1))))):
            t = torch.full((num_samples,), t, device='cuda', dtype=torch.long) # t*batch

            unmasking_method = 'random'
            if unmasking_method == 'random':
                # where to unmask
                changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1).unsqueeze(-1)
                # don't unmask somewhere already unmasked
                changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
                # update mask with changes
                unmasked = torch.bitwise_or(unmasked, changes)
            elif unmasking_method == 'autoregressive':
                changes = torch.zeros(x_t.shape, device=device).bool() # 8,32,16  0
                index = (int(autoregressive_step / shape[1]), autoregressive_step % shape[1])
                changes[:, index[0], index[1]] = True
                unmasked = torch.bitwise_or(unmasked, changes)
                autoregressive_step += 1

            # keep track of PoE probabilities
            x_0_probs = torch.zeros((num_samples,) + shape + (self.codebook_size,), device='cuda')
            # keep track of counts
            count = torch.zeros((num_samples,) + shape, device='cuda')

            # TODO: Monte carlo approximate this instead
            for i in range(0, x_lim+1, step):
                for j in range(0, y_lim+1, step):
                    # collect local noisy area
                    x_t_part = x_t[:, i:i+self.shape[1], j:j+self.shape[2]]

                    # increment count
                    count[:, i:i+self.shape[1], j:j+self.shape[2]] += 1.0

                    # flatten
                    x_t_part = x_t_part.reshape(x_t_part.size(0), -1)

                    # denoise
                    x_0_logits_part = self._denoise_fn(x_t_part, t=t)

                    # unflatten
                    x_0_logits_part = x_0_logits_part.reshape(x_t_part.size(0), self.shape[1], self.shape[2], -1)

                    # multiply probabilities
                    # for mixture
                    x_0_probs[:, i:i+self.shape[1], j:j+self.shape[2]] += torch.softmax(x_0_logits_part, dim=-1)

            # Mixture with Temperature
            x_0_probs = x_0_probs / x_0_probs.sum(-1, keepdim=True)
            C = torch.tensor(x_0_probs.size(-1)).float()
            x_0_probs = torch.softmax((torch.log(x_0_probs) + torch.log(C)) / temp, dim=-1)

            x_0_dist = dists.Categorical(probs=x_0_probs)
            x_0_hat = x_0_dist.sample().long()

            # update x_0 where anything has been masked
            x_t[changes] = x_0_hat[changes]

        return x_t









