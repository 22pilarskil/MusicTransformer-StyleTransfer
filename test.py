from model.rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR, MultiheadAttentionRPR

import torch

cross_attn = MultiheadAttentionRPR(512, 8, 0.1)
x = torch.zeros((1000, 24, 512))
embeddings = torch.zeros((24, 1000, 512))
output, weights = cross_attn(query=x, key=embeddings, value=embeddings)
print(output.shape)

