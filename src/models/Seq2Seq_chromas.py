# other
from typing import Optional

import torch
from einops import rearrange
import numpy as np
from torchaudio.transforms import MFCC, ComputeDeltas

# pyannote
from pyannote.audio.core.model import Model, Introspection

from pyannote.audio.core.task import Task
from pyannote.audio.utils.params import merge_dict

from .blocks import (
    TCN,
    WavLM_Feats,
    ChromaExtractor,
    Features_Fusion,
    Permute,
    Unsqueeze,
    Squeeze,
)


class Seq2Seq_chromas(Model):
    WAVLM_DEFAULTS = {
        "update_extract": False,
        "feat_type": "wavlm_large",
        "channels_dropout": 0.0,
    }

    TCN_DEFAULTS = {  # Original names
        "in_chan": 96,  # Number of channels from wavlm
        "out_chan": 4,  # Number of dimension wanted
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
        tcn: dict = TCN_DEFAULTS,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        if sample_rate != 16000:
            raise NotImplementedError("Seq2Sseq only supports 16kHz audio for now.")

        self.chroma = ChromaExtractor(
            sr=self.hparams.sample_rate,
            n_fft=512,
            n_chroma=12,
            hop_length=int(self.hparams.sample_rate * 20e-3),  # Align to WavLM
        )

        self.mfcc = MFCC(
            sample_rate=self.hparams.sample_rate,
            n_mfcc=20,
            dct_type=2,
            norm="ortho",
            log_mels=True,
            melkwargs={
                "n_fft": 512,
                "win_length": int(self.hparams.sample_rate * 25e-3),
                "hop_length": int(self.hparams.sample_rate * 20e-3),
                "n_mels": 128,
                "center": False,
            },
        )
        self.deriv = ComputeDeltas()
        self.CMVN_mfcc = torch.nn.InstanceNorm1d(20)
        self.CMVN_chroma = torch.nn.InstanceNorm1d(12)

        tcn = merge_dict(self.TCN_DEFAULTS, tcn)  # TCN params

        self.save_hyperparameters("tcn")

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
        with torch.no_grad():
            chroma = self.chroma(waveforms)
            chroma = self.CMVN_chroma(chroma.squeeze(1))
            chroma = chroma.unsqueeze(1)
            chroma = torch.cat(
                (chroma, self.deriv(chroma), self.deriv(self.deriv(chroma))), dim=2
            )
            mfcc = self.mfcc(waveforms)
            mfcc = self.CMVN_mfcc(mfcc.squeeze(1))
            mfcc = mfcc.unsqueeze(1)
            mfcc = torch.cat(
                (mfcc, self.deriv(mfcc), self.deriv(self.deriv(mfcc))), dim=2
            )
            feat_acous = torch.cat((mfcc, chroma), dim=2)
        feat_acous = rearrange(feat_acous, "b 1 f t -> b t f")

        features = feat_acous

        outputs = rearrange(features, "batch frame classes -> batch classes frame")
        outputs = self.tcn(outputs)
        outputs = rearrange(outputs, "batch classes frame -> batch frame classes")
        out = self.activation(outputs)
        # assert out.shape[1] == 99, print(out.shape)
        # with open("allies.csv",'a') as fout:
        #    for val2 in out:
        #        for val in val2:
        #            fout.write(f"{val[0]:.20f},{val[1]:.20f}\n")

        return out
