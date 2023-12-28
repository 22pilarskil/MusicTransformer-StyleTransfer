import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random

from utilities.constants import *
from utilities.device import get_device

from .positional_encoding import PositionalEncoding
from .rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR



class TransformerDecoderLayerRPR(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, er_len=None):
        super(TransformerDecoderLayerRPR, self).__init__()
        # Self-attention for the decoder
        self.self_attn = MultiheadAttentionRPR(d_model, nhead, dropout=dropout, er_len=er_len)

        # Cross-attention layer where queries come from the previous layer, keys and values come from encoder output
        self.multihead_attn = MultiheadAttentionRPR(d_model, nhead, dropout=dropout, er_len=er_len)

        # Feed forward network
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        # Dropout layers
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed forward network
        tgt2 = self.linear2(self.dropout(relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoderRPR(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoderRPR, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output


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

        encoder_norm = LayerNorm(self.d_model)
        encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
        encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)

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
