import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(EmbeddingLoss, self).__init__()
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, input_embeddings, output_sequence):
        """
        Computes the loss between the feature extracted from y and the target x.
        
        Parameters:
        x (torch.Tensor): Target tensor of shape (2000, 1, 512)
        y (torch.Tensor): Input tensor to be passed through the feature extractor
        
        Returns:
        torch.Tensor: Computed loss
        """
        y, style_embedding, positional_embedding = self.feature_extractor(output_sequence)
        output_embeddings = torch.concat((style_embedding, positional_embedding), 0)
        
        loss = F.mse_loss(output_embeddings, input_embeddings)
        return loss

