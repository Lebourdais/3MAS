#!/usr/bin/env python
# coding: utf-8


import os
import argparse
import datetime
import re
import time
from pathlib import Path

# Other
import pytorch_lightning as pl
import torch
from torch_audiomentations import AddBackgroundNoise, OneOf, Compose
from torchmetrics import (
    AUROC,
    Accuracy,
    CalibrationError,
    F1Score,
    Metric,
    Precision,
    Recall,
    MetricCollection,
)

# Pyannote
from pyannote.core import Timeline, Annotation
from pyannote.audio import Model
from pyannote.audio.tasks import MultiLabelSegmentation

from pyannote.audio.models.segmentation import PyanNet
from pyannote.database.registry import registry
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping
import yaml
from src.models import (
    Seq2Seq,
    Seq2Seq_chromas,
    Seq2Seq_Leaf,
)
from path import PATH_TO_PYANNOTE_DB,PATH_TO_NOISE,PATH_TO_MUSIC
from pytorch_lightning.callbacks import (
    RichProgressBar,
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from src.models.blocks import AdditiveAugment, OverlapAugment
from src.utils.preprocessors import annotation_treat
from src.pyannote_audio_alt.multilabel import (
    LoggableHistogram,
    MultiLabelSegmentationAlt,
)

USE_PYANNOTE_ALT = False
# Models
parser = argparse.ArgumentParser()
parser.add_argument("--model_typ", type=str, default="tcn")
parser.add_argument("--dataset", type=str, default="X.Segmentation.Full")

parser.add_argument("--duration", type=float, default=2.0, help="input duration")
parser.add_argument("--name", type=str)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-workers", type=int, default=30)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--initial-validate",
    type=bool,
    action=argparse.BooleanOptionalAction,
    help="Precede the .fit() with a .validate().",
)

parser.add_argument(
    "--augment", type=bool, default=True, action=argparse.BooleanOptionalAction
)
parser.add_argument(
    "--disabled-class-augmentations", type=str, nargs="+", default=[], help="list, allowed: mu/no/ov",
)

args = parser.parse_args()

# Load Database
tstart = time.time()
registry.load_database(PATH_TO_PYANNOTE_DB)

print(f"Loaded database in {time.time() - tstart} seconds")


def replace_subdirectory_with_list(path_in, subdirectory="subdirectory", names=[]):
    path_components = path_in.split(os.sep)
    # Find the index of the subdirectory in the path components
    subdir_index = path_components.index(subdirectory)
    # Create a list of new paths by replacing the subdirectory with each name in the list
    list_dirs = [
        os.sep.join(
            path_components[:subdir_index]
            + [name]
            + path_components[subdir_index + 1 :]
        )
        for name in names
    ]

    return list_dirs



tstart = time.time()
corpus = registry.get_protocol(
    args.dataset, preprocessors={"annotation": annotation_treat}
)
print(f"Loaded corpus in {time.time() - tstart} seconds")

# 1 - Generate a list of directories with the noise files
# Base path, subdirectory will be replaced with the noise name
long_path = PATH_TO_NOISE
# noise names

names = ["free-sound", "sound-bible"]


list_path_noises = replace_subdirectory_with_list(
    long_path, subdirectory="subdirectory", names=names
)

long_path = PATH_TO_MUSIC
# music names
names = ["fma", "fma-western-art", "hd-classical", "jamendo", "rfm"]
list_path_music = replace_subdirectory_with_list(
    long_path, subdirectory="subdirectory", names=names
)


classes = ["sp", "no", "mu", "ov"]
#classes = ["ov"]
augmentation = None
augment_noise = None
augment_music = None
if "no" in classes and args.augment and "no" not in args.disabled_class_augmentations:
    augment_noise = AdditiveAugment(
        background_paths=list_path_noises,
        index_class=classes.index("no"),
        max_snr_in_db=20,
        min_snr_in_db=5,
        p=0.2,
        mode="per_example",
        p_mode="per_example",
        output_type="dict",
    )
    augmentation = augment_noise
if "mu" in classes and args.augment and "mu" not in args.disabled_class_augmentations:
    augment_music = AdditiveAugment(
        background_paths=list_path_music,
        index_class=classes.index("mu"),
        max_snr_in_db=20,
        min_snr_in_db=1,
        p=0.5,
        mode="per_example",
        p_mode="per_example",
        output_type="dict",
    )
    augmentation = augment_music

if augment_noise is not None and augment_music is not None:
    augmentation = OneOf([augment_noise, augment_music], p=1.0, output_type="dict")

if "ov" in classes and args.augment and "ov" not in args.disabled_class_augmentations:
    augment_overlap = OverlapAugment(
        ov_class=classes.index("ov"),
        speech_class=classes.index("sp"),
        noise_class=classes.index("no") if "no" in classes else None,
        music_class=classes.index("mu") if "mu" in classes else None,
        max_snr_in_db=10,
        min_snr_in_db=1,
        p=0.3,
        mode="per_example",
        p_mode="per_example",
        output_type="dict",
    )
    if augmentation is not None:
        augmentation = Compose([augmentation, augment_overlap], output_type="dict")
    else:
        augmentation = augment_overlap
print(f"Augmentation: {augmentation}")


# taskhooks
taskhooks = []

# histograms

if USE_PYANNOTE_ALT:
    loggables = None
    full_seg = MultiLabelSegmentationAlt(
        corpus,  # defines the training and validation sets
        classes=classes,  # classes of interest
        duration=args.duration,  # the model will ingest 2s audio chunks
        augmentation=augmentation,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        metric_classwise=metric_classwise,
        loggables=loggables,
        taskhooks=taskhooks,
        augment_location=args.augment_location,
    )
else:
    full_seg = MultiLabelSegmentation(
        corpus,  # defines the training and validation sets
        classes=classes,  # classes of interest
        duration=args.duration,  # the model will ingest 2s audio chunks
        augmentation=augmentation,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    tstart = time.time()
    if args.model_typ == "tcn":
        model = Seq2Seq(task=full_seg, tcn={"out_chan": len(classes)})
    elif args.model_typ == "tcn_chromas":
        model = Seq2Seq_chromas(task=full_seg)
    elif args.model_typ == "tcn_leaf":
        filters = 60
        model = Seq2Seq_Leaf(
            task=full_seg,
            leaf={"n_filters": filters},
            tcn={"in_chan": filters, "out_chan": len(classes)},
        )
    else:
        raise ValueError(f"Model unknown : {args.model_typ}")

    model.setup("fit")
    val_check, val_direction = full_seg.val_monitor
    # TODO: remove this when pyannote.audio fixes it
    if val_check == "ValLoss":
        val_check = "loss/val"
    outname = f"{datetime.date.today().strftime('%y_%m_%d')}-checkpoint-{args.model_typ}-{args.dataset}-{{epoch:02d}}-{{{val_check}:.3f}}-{args.name}"
    print(f"Loaded model in {time.time() - tstart} seconds")

    # Get the best models without overfitting

    print("validation metric :", val_check, val_direction)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=val_check,
        mode=val_direction,
        dirpath="results/models/",
        filename=outname,
        auto_insert_metric_name=False,
    )
    # bar = RichProgressBar()
    bar = TQDMProgressBar(refresh_rate=10)
    early = EarlyStopping(monitor=val_check, patience=10, mode=val_direction)

    logger = TensorBoardLogger("", version=args.name)
    # csv_logger = CSVLogger("",version=args.name)
    os.makedirs("results/models/", exist_ok=True)
    with open(f"results/models/{'-'.join(outname.split('-')[0:3])}-{outname.split('-')[-1]}.log",'w') as log_train:
        for arg,value in sorted(vars(args).items()):
            log_train.write(f"Argument {arg}:{value}\n")
        log_train.write(f"Classes : {classes}")
    if args.seed is not None:
        pl.seed_everything(args.seed)

    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=[checkpoint_callback, bar, early],
        logger=[logger],
        deterministic=(args.seed is not None),
    )
    ckpt_path = None
    trainer.fit(model, ckpt_path=ckpt_path)
