from torch import nn, Tensor


class MSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor):
        return ((x-y)**2).mean()


def get_recons_loss(loss_type: str='mse', **kwargs):

    if loss_type.upper()=='MSE':
        return MSE()
    else:
        raise ValueError
