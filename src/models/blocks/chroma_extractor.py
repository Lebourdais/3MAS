import torch
import librosa
import numpy as np
from einops import rearrange
from torchaudio.transforms import Spectrogram
import pytorch_lightning as pl


class ChromaExtractor(pl.LightningModule):
    def __init__(self, sr=16000, n_fft=512, n_chroma=12, hop_length=512):
        super(ChromaExtractor, self).__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.n_chroma = n_chroma
        self.hop_length = hop_length

        self.spec = Spectrogram(
            n_fft=self.n_fft,
            win_length=int(self.sr * 25e-3),
            hop_length=self.hop_length,
            power=1,  # 1 for magnitude, 2 for power
            center=False,  # This needs to be false to match MFCC framerate
        )

        # Precompute chroma filter
        chroma_filter = librosa.filters.chroma(
            sr=self.sr, n_fft=self.n_fft, n_chroma=self.n_chroma
        )

        self.register_buffer("chroma_gpu", torch.from_numpy(chroma_filter))

    def forward(self, x):
        # expected input is (batch, channels, time)
        # output is (batch, channels, time, chroma)
        chroma_gpu = self.chroma_gpu.to(x)
        batch, channels, samples = x.size()
        # considering multichannel input makes things harder, so we'll just assume single channel
        if channels > 1:
            raise NotImplementedError(
                "Warning: ChromaExtractor expects a single channel input."
            )

        # shape is (batch, freq, time)
        out = self.spec(x)
        out = out.squeeze(1).transpose(1, 2)

        # (batch, time, freq) x (freq, chroma) -> (batch, time, chroma)
        chroma = torch.matmul(out, chroma_gpu.transpose(0, 1))
        out = chroma.unsqueeze(1).transpose(2, 3)
        return out


# Test
if __name__ == "__main__":
    sr = 16000
    chroma_extractor = ChromaExtractor(
        sr=sr, n_fft=512, n_chroma=12, hop_length=int(sr * 10e-3)
    )

    # Assume input is single channel
    x = torch.randn(64, 1, 16000 * 4)

    print(x.size())
    chroma = chroma_extractor(x)
    print(chroma.size())
