import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random

from utilities.constants import *
from utilities.device import get_device

from .positional_encoding import PositionalEncoding
from .rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR, MultiheadAttentionRPR


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

        self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)
        self.linear_embeddings = nn.Linear(256, d_model)

        self.cross_attn = MultiheadAttentionRPR(self.d_model, self.nhead, self.dropout)
        self.cross_attn_layer_norm = LayerNorm(d_model)

        encoder_norm = LayerNorm(self.d_model)
        encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
        encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)

        self.transformer = nn.Transformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
            num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
            dim_feedforward=self.d_ff, custom_decoder=self.dummy, custom_encoder=encoder
        )

        # Final output is a softmaxed linear layer
        self.Wout       = nn.Linear(self.d_model, VOCAB_SIZE)
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, x, style_embedding, content_embedding, mask=True):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Takes an input sequence and outputs predictions using a sequence to sequence method.

        A prediction at one index is the "next" prediction given all information seen previously.
        ----------
        """

        if mask is True:
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())
        else:
            mask = None

        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        
        embeddings = torch.cat([style_embedding, content_embedding], dim=1)  # Shape: (batch_size, 512)
        embeddings = self.linear_embeddings(embeddings)  # Shape: (batch_size, d_model)
        embeddings = embeddings.unsqueeze(1).repeat(1, x.size(1), 1)  # Shape: (batch_size, sequence_length, d_model)

        attn_output, attn_weights = self.cross_attn(query=x, key=embeddings, value=embeddings)
        attn_output = self.cross_attn_layer_norm(attn_output + x)
        x_out = self.transformer(src=attn_output, tgt=attn_output, src_mask=mask)
        

        # x_out = self.transformer(src=x, tgt=x, src_mask=mask)
        x_out = x_out.permute(1, 0, 2)
        y = self.Wout(x_out)

        del mask
        return y


    def generate(self, style_embedding, content_embedding, target_seq_length=1024):
        assert (not self.training), "Cannot generate while in training mode"

        print("Generating sequence of max length:", target_seq_length)

        # Initialize gen_seq with TOKEN_START and the rest as TOKEN_PAD
        gen_seq = torch.full((1, target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq[..., 0] = TOKEN_START

        cur_i = 1  # Start from index 1 as the first token is TOKEN_START
        while(cur_i < target_seq_length):
            y = self.softmax(self.forward(gen_seq[..., :cur_i], style_embedding, content_embedding))[..., :(TOKEN_END + 1)]
            token_probs = y[:, cur_i-1, :]
            print(gen_seq[:100])
            # Select the argmax of the logits
            next_token = torch.argmax(token_probs, dim=-1)
            gen_seq[:, cur_i] = next_token

            # Check for the end of sequence token
            if(next_token == TOKEN_END):
                print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                break

            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", target_seq_length)

        return gen_seq[:, :cur_i]



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
