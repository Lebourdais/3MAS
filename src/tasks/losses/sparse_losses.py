import torch
from torch import nn, Tensor


class L1(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, ):
        loss = torch.norm(x, p=1, dim=(1,2)).mean()
        return loss/x.shape[1]/x.shape[2] 


def get_sparse_loss(loss_type: str='l1', **kwargs):

    if loss_type.upper()=='L1':
        return L1()
    else:
        raise ValueError
