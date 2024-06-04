from torch import nn, Tensor
import torch


def binary_cross_entropy(p, q):
    return torch.sum(
        - p * torch.log(q) - (1 - p) * torch.log(1 - q),
        axis=1
    ).mean()


def binary_kl_divergence(p, q):
    return torch.sum(
        p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q)),
        axis=-1
    ).mean()


class CELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: Tensor, teacher_logits: Tensor=None, labels: Tensor=None):

        #return  binary_cross_entropy(torch.sigmoid(logits), labels)
        return torch.nn.functional.binary_cross_entropy(torch.sigmoid(logits),labels.type(torch.float))


class KDLoss(nn.Module):

    def __init__(
        self,
        alpha: float=1.,
        temperature: float=1.
    ):
        super().__init__()
        assert alpha>=0 and alpha<=1, "alpha must be between 0 and 1"
        self.alpha = alpha
        self.temperature = temperature


    def forward(self, logits: Tensor, teacher_logits: Tensor, labels: Tensor=None):

        teacher_loss = binary_kl_divergence(
            torch.sigmoid(logits/self.temperature), 
            torch.sigmoid(teacher_logits/self.temperature)
        )

        if self.alpha==1.:
            return teacher_loss

        else:
            assert labels is not None, "labels cannot be None for alpha lower than one"
            labels_loss = binary_cross_entropy(torch.sigmoid(logits), labels)
            return (self.alpha*self.temperature**2)*teacher_loss + labels_loss*(1-self.alpha)
        

def get_class_loss(loss_type: str='ce', **kwargs):

    if loss_type.upper()=='CE':
        return CELoss(**kwargs)
    elif loss_type.upper()=='KD':
        return KDLoss(**kwargs)
    else:
        raise ValueError
