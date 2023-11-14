import torch
import torch.nn as nn
from .clip.clip import tokenize
from .base_codec import BaseCodec
from .clip.simple_tokenizer import SimpleTokenizer

class Tokenize(BaseCodec):
    def __init__(self, context_length:int = 256,
                 add_start_and_end:bool = False,
                 just_token = False,
                 with_mask:bool = True,
                 pad_value:int = 0,
                 clip_embedding = False,
                 condition_emb_config = None,                 ):
        """
        This is a wrapper class for tokenize of texts.
        For CLIP and DALLE-pytorch tokenize, the default
        arguments are different:

        CLIP based:
            context_length: 77
            add_start_and_end: True

        DALLE-pytorch based:
            context_length: 256
            add_start_and_end: False
        
        """
        super().__init__()
        self.context_length = 77
        self.add_start_and_end = True
        self.with_mask = True
        self.pad_value = 0
        self.just_token = False
        self.trainable = False
        self.condition_emb = None
        self.clip_embedding = False

        self.tokenizer = SimpleTokenizer()
    
    def __repr__(self):
        rep = "Tokenize for text\n\tcontent_length: {}\n\tadd_start_and_end: {}\n\twith_mask: {}"\
                .format(self.context_length, self.add_start_and_end, self.with_mask)
        return rep

    def check_length(self, token):
        return len(token) <= self.context_length

## text token    clip_encoder
    @torch.no_grad()
    def get_tokens(self, text,**kwargs):
        text_token = tokenize(text, context_length=self.context_length,
                         add_start_and_end=self.add_start_and_end,
                         with_mask=self.with_mask, pad_value=self.pad_value,
                         tokenizer=self.tokenizer,
                         just_token=self.just_token)
        return text_token





