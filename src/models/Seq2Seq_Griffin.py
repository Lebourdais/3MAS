# other
from typing import Optional

import torch
from einops import rearrange

# pyannote
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.utils.params import merge_dict
from torchaudio.transforms import ComputeDeltas,GriffinLim, Spectrogram
from .blocks import TCN, WavLM_Feats


class Seq2Seq_Griffin(Model):
    WAVLM_DEFAULTS = {
        "update_extract": False,
        "feat_type": "wavlm_large",
        "channels_dropout": 0.0,
        "return_all_layers": False,
        "return_layer": None,
    }

    TCN_DEFAULTS = {  # Original names
        "in_chan": 1024,  # Number of channels from wavlm
        "out_chan": 1,  # Number of dimension wanted
        "n_src": 1,  # Number of audio channels
        "n_blocks": 3,
        "n_repeats": 5,
        "bn_chan": 64,
        "hid_chan": 128,
        "kernel_size": 3,
        "norm_type": "gLN",
        "representation": False,
    }

    def __init__(
        self,
        wavlm: dict = WAVLM_DEFAULTS,
        tcn: dict = TCN_DEFAULTS,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        if sample_rate != 16000:
            raise NotImplementedError("Seq2Sseq only supports 16kHz audio for now.")

        wavlm = merge_dict(self.WAVLM_DEFAULTS, wavlm)  # WavLM params

        tcn = merge_dict(self.TCN_DEFAULTS, tcn)  # TCN params
        self.spectro = Spectrogram(n_fft=512)
        self.griffin = GriffinLim(n_fft=512) 
        self.save_hyperparameters("wavlm", "tcn")
        self.wavlm = WavLM_Feats(**self.hparams.wavlm)

    def build(self):
        self.tcn = TCN(**self.hparams.tcn)
        self.activation = (
            self.default_activation()
        )  # Sigmoid in multilabel (softmax don't work at all)
        self.wavlm.feature_weight.requires_grad = True


    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        spectro = self.spectro(waveforms)
        waveforms = self.griffin(spectro)
        outputs = self.wavlm(
            waveforms.squeeze(dim=1)
        )  # Extraction of wavLM characteristics
        outputs = self.tcn(outputs)
        outputs = rearrange(outputs, "batch classes frame -> batch frame classes")
        out = self.activation(outputs)
        return out
