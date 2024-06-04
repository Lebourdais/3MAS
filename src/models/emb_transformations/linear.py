from torch import nn


class Linear(nn.Module):

    """Just a simple Linear Transformation
    Args:
        in_chan (int): Number of input filters.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
    """

    def __init__(
        self,
        in_chan,
        out_chan=None,
    ):
        super(Linear, self).__init__()
        self.in_chan = in_chan
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan

        self.ll = nn.Linear(in_chan, out_chan)


    def forward(self, mixture_w):
        """
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape
                [batch, n_filters, n_frames]
        Returns:
            :class:`torch.Tensor`:
                new embeddings [batch, n_filters, n_frames]
        """
        return self.ll(mixture_w.permute(0,2,1)).permute(0,2,1)
