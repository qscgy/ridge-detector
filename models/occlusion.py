import torch
import torch.nn as nn

class GraphConvolutionLayer(nn.Module):
    def __init__(self, n):
        super().__init__()
