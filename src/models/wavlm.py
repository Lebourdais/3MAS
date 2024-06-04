import torch
from torch import nn
from collections import OrderedDict
import pytorch_lightning as pl
from transformers import AutoModel


def apply_along_axis(function, x, axis: int = 0):
    return torch.stack([function(x_i) for x_i in torch.unbind(x, dim=axis)], dim=axis)

class WavLM_Feats(pl.LightningModule):
    """Pretrained WavLM feature extractor (https://arxiv.org/pdf/2110.13900.pdf)
    implementation inspired from https://github.com/microsoft/UniSpeech/blob/main/downstreams/speaker_verification/models/ecapa_tdnn.py
    Adapted from SiDiar

    Args:
        update_extract (bool): allows finetuning if True, can be a regexp or array or regexp to finetune some layers
        channels_dropout (float in [0, 1]): channel dropout probability
    """

    def __init__(
        self,
        update_extract=False,
        feat_type="wavlm_large",
        return_all_layers=False,
        return_layer=None,
        channels_dropout=0.0,
        cache_dir="~/.cache/models/",
    ):
        super(WavLM_Feats, self).__init__()
        if return_layer is not None and return_all_layers:
            raise ValueError(
                "You can't ask for one layer and all layers at the same time ..."
            )
        self.feat_type = feat_type
        self.is_small = "base" in feat_type
        # self.feature_extract = torch.hub.load('s3prl/s3prl', self.feat_type)
        # supress import error for other ssl pre-trained model than wavlm.
        self.feature_extract = torch.hub.load("s3prl/s3prl", self.feat_type)
        #self.feature_extract =  AutoModel.from_pretrained("microsoft/wavlm-large",cache_dir=cache_dir) 
        self.update_extract = update_extract
        self.feature_selection = "hidden_states"
        self.return_all_layers = return_all_layers
        self.return_layer = return_layer
        self.sr = 16000
        self.feat_num = self.get_feat_num()
        if not self.is_small:
            self.instance_norm = nn.InstanceNorm1d(1024)
        else:  # Base plus model
            self.instance_norm = nn.InstanceNorm1d(768)
        self.channels_dropout = channels_dropout
        self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))
        freeze_list = [
            "final_proj",
            "label_embs_concat",
            "mask_emb",
            "project_q",
            "quantizer",
        ]
        for name, param in self.feature_extract.named_parameters():
            for freeze_val in freeze_list:
                if freeze_val in name:
                    param.requires_grad = False
                    break
        # print("==============================",self.update_extract)
        if not self.update_extract:
            for param in self.feature_extract.parameters():
                param.requires_grad = False
        else:
            if (
                type(self.update_extract) == bool
            ):  # If it is True (not false and boolean)
                for param in self.feature_extract.parameters():
                    param.requires_grad = True

            else:  # Regexp
                if len(self.update_extract) > 1:
                    pattern = " | ".join(self.update_extract)
                else:
                    pattern = self.update_extract[0]
                prog = re.compile(pattern)
                for name, param in self.feature_extract.named_parameters():
                    # print(name)
                    # print(prog.match(name))
                    if prog.match(name) is not None:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

    def get_feat_num(self):
        """

        :return:
        """
        self.feature_extract.eval()
        wav = [torch.randn(self.sr).to(next(self.feature_extract.parameters()).device)]
        features = self.feature_extract(wav)
        select_feature = features[self.feature_selection]
        if isinstance(select_feature, (list, tuple)):
            if not self.is_small:  # Dirty fix
                return len(select_feature)
            else:
                return len(select_feature) - 1
        else:
            return 1

    def get_feat(self, x):
        """

        :param x:
        :return:
        """
        if self.update_extract:
            x = self.feature_extract([sample for sample in x])
        else:
            with torch.no_grad():
                x = self.feature_extract([sample for sample in x])

        x = x[self.feature_selection]
        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)
        else:
            x = x.unsqueeze(0)

        if not self.is_small:  # Base+ model
            norm_weights = (
                nn.functional.softmax(self.feature_weight, dim=-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
        else:
            if x.shape[0] == self.feature_weight.shape[0]:
                norm_weights = (
                    nn.functional.softmax(self.feature_weight, dim=-1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
            else:
                norm_weights = (
                    nn.functional.softmax(
                        nn.Parameter(torch.zeros(x.shape[0])).to(
                            next(self.feature_extract.parameters(), None).device
                        ),
                        dim=-1,
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )  # not always the same length (13 and 12)

        if self.return_layer is not None:
            x = torch.transpose(x, 2, 3) + 1e-6
            x = x[self.return_layer]
            x = self.instance_norm(x)
            x = torch.nan_to_num(x, nan=0.0)
            return x

        if self.return_all_layers:
            x = torch.transpose(x, 2, 3) + 1e-6  # Layer x B x T x C
            x = apply_along_axis(self.instance_norm, x, axis=0)
            x = torch.nan_to_num(x, nan=0.0)
            return x

        x = (norm_weights * x).sum(dim=0)
        x = torch.transpose(x, 1, 2) + 1e-6

        x = self.instance_norm(x)
        x = torch.nan_to_num(x, nan=0.0)

        if self.training:
            x *= nn.functional.dropout(
                torch.ones((1, 1, x.shape[2]), device=x.device), p=self.channels_dropout
            )

        return x

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = x.squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.get_feat(x)

    def feature_size(self):
        return 1024
