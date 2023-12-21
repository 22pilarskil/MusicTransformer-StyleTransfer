import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random

from utilities.constants import *
from utilities.device import get_device

from .positional_encoding import PositionalEncoding
from .rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR


# MusicTransformer
class Reconstructor(nn.Module):


    def __init__(self, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence=1000, rpr=False, style_layer=5):
        super(Reconstructor, self).__init__()

        self.dummy      = DummyDecoder()

        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq    = max_sequence
        self.rpr        = rpr
        self.style_layer = style_layer

        # Input embedding
        # self.continuous_input_handler = nn.Linear(self.d_model, self.d_model)
        # self.style_attention = nn.MultiheadAttention(d_model, num_heads)

        # Positional encoding
        # self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)

        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * d_model, d_model),  # Fusion layer to blend style and position
            nn.ReLU(),                        # Non-linear activation
            nn.Dropout(p=dropout),            # Dropout for regularization
            nn.LayerNorm(d_model)             # Layer normalization
        )

        encoder_norm_main = LayerNorm(self.d_model)
        encoder_layer_main = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
        encoder_main = TransformerEncoderRPR(encoder_layer_main, self.nlayers, encoder_norm_main)

        self.transformer_main = nn.Transformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
            num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
            dim_feedforward=self.d_ff, custom_decoder=self.dummy, custom_encoder=encoder_main
        )

        # Final output is a softmaxed linear layer
        self.Wout       = nn.Linear(self.d_model, VOCAB_SIZE)


    def forward(self, style_emb, pos_emb, mask=False):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Takes an input sequence and outputs predictions using a sequence to sequence method.

        A prediction at one index is the "next" prediction given all information seen previously.
        ----------
        """

        '''
        pos_emb = self.continuous_input_handler(pos_emb)
        style_emb = self.continuous_input_handler(style_emb)

        attn_output, _ = self.style_attention(query=pos_emb, key=style_emb, value=style_emb)
        # Input shape is (max_seq, batch_size, d_model)
        combined_output = attn_output + style_emb
        '''
        combined_emb = torch.cat((style_emb, pos_emb), dim=-1)
        # combined_emb = torch.cat((style_emb, style_emb), dim=-1)
        combined_output = self.fusion_layer(combined_emb)

        x_out = self.transformer_main(src=combined_output, tgt=combined_output, src_mask=None)
        # Back to (batch_size, max_seq, d_model)
        x_out = x_out.permute(1,0,2)

        logits = self.Wout(x_out)
        # logits = (logits - logits.mean()) / (logits.std() + 1e-6)
        logits = logits.permute(1, 0, 2)

        return logits



# Used as a dummy to nn.Transformer
# DummyDecoder
class DummyDecoder(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    A dummy decoder that returns its input. Used to make the Pytorch transformer into a decoder-only
    architecture (stacked encoders with dummy decoder fits the bill)
    ----------
    """

    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask,tgt_key_padding_mask,memory_key_padding_mask,tgt_is_causal, memory_is_causal):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Returns the input (memory)
        ----------
        """

        return memory
