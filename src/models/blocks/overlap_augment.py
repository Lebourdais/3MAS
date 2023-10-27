import random
from pathlib import Path
from typing import Union, List, Optional

import torch
from torch import Tensor
import numpy as np
from torch_audiomentations.core.transforms_interface import (
    BaseWaveformTransform,
    EmptyPathException,
)
from torch_audiomentations.utils.dsp import calculate_rms
from torch_audiomentations.utils.file import find_audio_files_in_paths
from torch_audiomentations.utils.io import Audio
from torch_audiomentations.utils.object_dict import ObjectDict
import torchaudio

# from codetiming import Timer


class OverlapAugment(BaseWaveformTransform):
    """
    Additive mixture of two waveforms.
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    # Note: This transform has only partial support for multichannel audio. Noises that are not
    # mono get mixed down to mono before they are added to all channels in the input.
    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    def __init__(
        self,
        ov_class: int = None,
        speech_class: int = None,
        music_class: int = None,
        noise_class: int = None,
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        """
        :param min_snr_in_db: minimum SNR in dB.
        :param max_snr_in_db: maximum SNR in dB.
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """

        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

        # index class indicates which class is the background noise, meant to be used to modify the targets accordingly
        # targets are one hot encoded, this dimension marks what dimension need to be changed
        # Suppose we are adding noise to a 3 class speech, music and noise segmentation task
        # Then, we want to modify the targets such that the noise class 1, and the other classes remain the same
        self.p = p
        self.ov_class = ov_class
        self.speech_class = speech_class
        self.music_class = music_class
        self.noise_class = noise_class
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")
        self.register_buffer(
            "transform1",
            torch.tensor(
                [  # speech_main -1,0,1
                    [  # speech_back -1,0,1
                        [  # ov_main -1,0,1
                            [-1, -1, -1],  # ov_back -1,0,1
                            [-1, -1, -1],  # ov_back -1,0,1
                            [-1, -1, -1],  # ov_back -1,0,1
                        ],
                        [  # ov_main -1,0,1
                            [-1, -1, -1],  # ov_back -1,0,1
                            [-1, -1, -1],  # ov_back -1,0,1
                            [-1, -1, -1],  # ov_back -1,0,1
                        ],
                        [  # ov_main -1,0,1
                            [-1, -1, -1],  # ov_back -1,0,1
                            [-1, -1, -1],  # ov_back -1,0,1
                            [-1, -1, -1],  # ov_back -1,0,1
                        ],
                    ],
                    [  # speech_back -1,0,1
                        [  # ov_main -1,0,1
                            [-1, -1, -1],  # ov_back -1,0,1
                            [-1, -1, -1],  # ov_back -1,0,1
                            [-1, -1, -1],  # ov_back -1,0,1
                        ],
                        [  # ov_main -1,0,1
                            [-1, -1, 1],  # ov_back -1,0,1
                            [-1, 0, 1],  # ov_back -1,0,1
                            [1, 1, 1],  # ov_back -1,0,1
                        ],
                        [  # ov_main -1,0,1
                            [-1, -1, 1],  # ov_back -1,0,1
                            [-1, 0, 1],  # ov_back -1,0,1
                            [1, 1, 1],  # ov_back -1,0,1
                        ],
                    ],
                    [  # speech_back -1,0,1
                        [  # ov_main -1,0,1
                            [-1, -1, -1],  # ov_back -1,0,1
                            [-1, -1, -1],  # ov_back -1,0,1
                            [-1, -1, -1],  # ov_back -1,0,1
                        ],
                        [  # ov_main -1,0,1
                            [-1, -1, 1],  # ov_back -1,0,1
                            [-1, 0, 1],  # ov_back -1,0,1
                            [1, 1, 1],  # ov_back -1,0,1
                        ],
                        [  # ov_main -1,0,1
                            [1, 1, 1],  # ov_back -1,0,1
                            [1, 1, 1],  # ov_back -1,0,1
                            [1, 1, 1],  # ov_back -1,0,1
                        ],
                    ],
                ]
            ),
        )
        self.register_buffer(
            "transform2",
            torch.tensor(
                [  # noise_main -1,0,1
                    [-1, -1, 1],  # noise_back -1,0,1
                    [-1, 0, 1],  # noise_back -1,0,1
                    [1, 1, 1],  # noise_back -1,0,1
                ]
            ),
        )
        self.register_buffer(
            "transform_speech",
            torch.tensor(
                [  # noise_main -1,0,1
                    [-1, -1, -1],  # noise_back -1,0,1
                    [-1, 0, 1],  # noise_back -1,0,1
                    [-1, 1, 1],  # noise_back -1,0,1
                ]
            ),
        )

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """

        :params samples: (batch_size, num_channels, num_samples)
        """

        batch_size, _, num_samples = samples.shape
        snr_distribution = torch.distributions.Uniform(
            low=torch.tensor(
                self.min_snr_in_db, dtype=torch.float32, device=samples.device
            ),
            high=torch.tensor(
                self.max_snr_in_db, dtype=torch.float32, device=samples.device
            ),
            validate_args=True,
        )
        self.transform_parameters["snr_in_db"] = snr_distribution.sample(
            sample_shape=(batch_size,)
        )

        # sp_target = targets[:,:,:,self.speech_class]

        # num_speakers: torch.Tensor = torch.sum(torch.any(targets, dim=-2), dim=-1)

        self.transform_parameters["sample_idx"] = torch.arange(
            batch_size, dtype=torch.int64, device=samples.device
        )

        # samples_with_speakers = torch.where(torch.any((sp_target > 0), dim=-1))[0]

        # num_samples_with_speakers = len(samples_with_speakers)
        candidates = torch.arange(len(samples), device=samples.device)
        num_candidates = len(candidates)

        rand = torch.randint(
            0,
            num_candidates,
            (num_candidates,),
            device=samples.device,
        )
        # rand = torch.where(rand)

        # rand = torch.tensor([i if i != ii else ((i + 1) % rand.shape[0]) for ii, i in enumerate(rand.numpy())],device=samples.device)
        selected_candidates = candidates[rand]
        self.transform_parameters["sample_idx"][candidates] = selected_candidates
        # self.transform_parameters["sample_idx"][
        #     targets
        # ] = selected_candidates

    def merge_speech(self, main, background):
        speech_main = main[:, :, :, self.speech_class].long()
        speech_back = background[:, :, :, self.speech_class].long()
        return self.transform_speech[speech_main + 1, speech_back + 1]

    def merge_music(self, main, background):
        music_main = main[:, :, :, self.music_class].long()
        music_back = background[:, :, :, self.music_class].long()
        return self.transform2[music_main + 1, music_back + 1]

    def merge_noise(self, main, background):
        noise_main = main[:, :, :, self.noise_class].long()
        noise_back = background[:, :, :, self.noise_class].long()
        return self.transform2[noise_main + 1, noise_back + 1]

    def merge_overlap(self, main, background):
        speech_main = main[:, :, :, self.speech_class].long()
        speech_back = background[:, :, :, self.speech_class].long()
        ov_main = main[:, :, :, self.ov_class].long()
        ov_back = background[:, :, :, self.ov_class].long()
        return self.transform1[
            speech_main + 1, speech_back + 1, ov_main + 1, ov_back + 1
        ]

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        snr = self.transform_parameters["snr_in_db"]
        idx = self.transform_parameters["sample_idx"]
        # targets[:,:,:,self.ov_class] = torch.clamp(input=(targets[:,:,:,self.ov_class]),min=0)
        # batch_size, num_channels, num_samples = samples.shape
        batch_size, num_channels, num_samples = samples.shape
        # for ii,v in enumerate(samples):
        #     torchaudio.save(f"outputs/{ii}_sample.wav", v, sample_rate)
        samples = torch.nan_to_num(samples, nan=1e-6, posinf=1e-6, neginf=1e-6)
        samples += (1e-6 * torch.randn(*samples.shape, device=samples.device)).long()
        background_samples = samples[idx]
        # for ii,v in enumerate(targets):
        #     with open(f"outputs/{ii}_target.csv",'w') as fout:
        #         for t in v[0]:
        #             fout.write(",".join(map(str,t.numpy())))
        #             fout.write('\n')

        # for ii,v in enumerate(background_samples):
        #     torchaudio.save(f"outputs/{ii}_background_sample.wav", v, sample_rate)

        speech_power = samples.norm(p=2, dim=-1)
        if torch.any(speech_power == 0):
            speech_power += 1e-6
        noise_power = background_samples.norm(p=2, dim=-1)
        snr = 10 ** (snr.unsqueeze(dim=-1) / 20)
        scale = snr * noise_power / speech_power
        # background_rms = calculate_rms(samples) / (10 ** (snr.unsqueeze(dim=-1) / 20))

        mixed_samples = (
            torch.add(samples, background_samples * scale.view(-1, 1, 1)) / 2
        )

        if torch.any(torch.isnan(mixed_samples)):
            print(mixed_samples)
            print(speech_power)
            print(noise_power)
            print(scale)
            raise ValueError("NaN In mixed samples")

        # print(mixed_samples,samples)

        # for ii, v in enumerate(mixed_samples):
        #     torchaudio.save(f"outputs/{ii}_mixed_sample.wav", v, sample_rate)

        # If no targets are provided this part can be just skipped
        if targets is None:
            return ObjectDict(
                samples=mixed_samples,
                sample_rate=sample_rate,
                targets=targets,
                target_rate=target_rate,
            )

        background_targets = targets[idx]
        # for ii,v in enumerate(background_targets):
        #     with open(f"outputs/{ii}_background_target.csv",'w') as fout:
        #         for t in v[0]:
        #             fout.write(",".join(map(str,t.numpy())))
        #             fout.write('\n')
        if (background_samples[idx] == samples[idx]).all():
            assert torch.all(targets <= 1)
            assert not torch.any(torch.isnan(targets))
            assert not torch.any(torch.isnan(samples))
            out = ObjectDict(
                samples=samples,
                sample_rate=sample_rate,
                targets=targets,
                target_rate=target_rate,
            )
            return out
        else:
            new_targets = targets.clone()
            # new_targets = new_targets.numpy()

            new_targets[:, :, :, self.speech_class] = self.merge_speech(
                new_targets, background_targets
            )
            new_targets[:, :, :, self.ov_class] = self.merge_overlap(
                new_targets, background_targets
            )
            if self.music_class is not None:
                new_targets[:, :, :, self.music_class] = self.merge_music(
                    new_targets, background_targets
                )
            if self.noise_class is not None:
                new_targets[:, :, :, self.noise_class] = self.merge_noise(
                    new_targets, background_targets
                )
            # new_targets[:,:,:,self.speech_index] = torch.clamp(input=(new_targets[:,:,:,self.speech_index]), max=1)

            # for ii,v in enumerate(new_targets):
            #     with open(f"outputs/{ii}_mixed_target.csv",'w') as fout:
            #         for t in v[0]:
            #             fout.write(",".join(map(str,t.numpy())))
            #             fout.write('\n')
            # The goal is to modify targets so that if we are adding noise or music this is accordingly marked as a positive class in target labels
            # targets are (batch, channel, frames, classes)
            # right now labels can be 1, 0 or -1, index class dimension should be 1 after the augmentation

            # targ = torch.from_numpy(new_targets)
            assert not torch.any(torch.isnan(mixed_samples))
            assert not torch.any(torch.isnan(new_targets))
            targ = new_targets
            out = ObjectDict(
                samples=mixed_samples,
                sample_rate=sample_rate,
                targets=targ,
                target_rate=target_rate,
            )
        return out
