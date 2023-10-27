# other
from typing import Optional

import torch
from einops import rearrange

# pyannote
from pyannote.audio.core.model import Model, Introspection
from pyannote.audio.core.task import Task
from pyannote.audio.utils.params import merge_dict

from .blocks import TCN, Leaf


class Seq2Seq_Leaf(Model):
    LEAF_DEFAULTS = {"use_legacy_complex": False, "initializer": "default"}

    TCN_DEFAULTS = {  # Original names
        "in_chan": 40,  # Number of channels from leaf
        "out_chan": 2,  # Number of dimension wanted
        "n_src": 1,  # Number of audio channels
        "n_blocks": 3,
        "n_repeats": 5,
        "bn_chan": 64,
        "hid_chan": 128,
        "kernel_size": 3,
        "norm_type": "gLN",
    }

    def __init__(
        self,
        leaf: dict = LEAF_DEFAULTS,
        tcn: dict = TCN_DEFAULTS,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        if sample_rate != 16000:
            raise NotImplementedError("Seq2Sseq only supports 16kHz audio for now.")

        leaf = merge_dict(self.LEAF_DEFAULTS, leaf)  # WavLM params

        tcn = merge_dict(self.TCN_DEFAULTS, tcn)  # TCN params

        self.save_hyperparameters("leaf", "tcn")

        # self.wavlm = WavLM_Feats(**self.hparams.wavlm)
        self.leaf = Leaf(**self.hparams.leaf)

    def build(self):
        self.tcn = TCN(**self.hparams.tcn)
        self.activation = (
            self.default_activation()
        )  # Sigmoid in multilabel (softmax don't work at all)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        outputs = self.leaf(
            waveforms
        )  # Extraction of leaf characteristics, without considering it in backprop
        outputs = self.tcn(outputs)
        outputs = rearrange(outputs, "batch classes frame -> batch frame classes")
        out = self.activation(outputs)
        # print(out.shape)
        # with open("allies.csv",'a') as fout:
        #    for val2 in out:
        #        for val in val2:
        #            fout.write(f"{val[0]:.20f},{val[1]:.20f}\n")

        return out
