from model.music_transformer import MusicTransformer
import torch
from utilities.device import get_device

'''
x = torch.zeros((1, 1000, 483)).to(get_device())
model = MusicTransformer(max_sequence=1000, rpr=True).to(get_device())
y, style_embedding, positional_embedding = model(x)
print(style_embedding.shape, positional_embedding.shape)
'''

x = torch.zeros((1, 1000)).int().to(get_device())
model = MusicTransformer(max_sequence=1000, rpr=True).to(get_device())
y, style_embedding, positional_embedding = model(x)
print(style_embedding.shape, positional_embedding.shape)
