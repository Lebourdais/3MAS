#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import time
from typing import List, Tuple
import random
import string
# Other
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import datetime
from pyannote.core import Annotation
from pyannote.database.protocol import Protocol
import torch
import pickle
from path import PATH_TO_PYANNOTE_DB 
# Pyannote
from pyannote.audio import Inference, Model
from pyannote.database import FileFinder

# Models architectures
from pyannote.audio.utils.signal import Binarize
from pyannote.database.registry import registry
from pyannote.database.util import load_rttm
from rich.progress import track
from rich.console import Console

from eval_full import evaluate

from src.utils.preprocessors import annotation_treat

DEFAULT_DATASET = "X.Segmentation.Benchmark"
DEFAULT_REPORT_DATASETS = [
    "ALLIES.Segmentation.Allies_Clean.test",
    "ALLIES.Segmentation.Test_D.test",
    "ALLIES.Segmentation.Test_E.test",
    "DIHARD.Segmentation.Benchmark.test",
    "AragonRadio.Segmentation.Benchmark.test",
    "X.Segmentation.Benchmark.test",
]


def parse_params(params):
    print("Parameters parsing")
    out = {}
    argum = params.split(",")
    for arg in argum:
        key, value = arg.split(":")
        if value in ["True", "False"]:
            value = value == "True"
            print(f"Got the boolean {key} at {value}")
            out.update({key: value})
        else:
            try:
                value = float(value)
                out.update({key: value})
                print(f"Got the float {key} : {value}")
            except:
                print(f"Got the string {key} : {value}")
                out.update({key: value})
    return out


parser = argparse.ArgumentParser(description="Prediction parameters")
parser.add_argument(
    "checkpoint", help="Path to the model checkpoint to evaluate, mandatory"
)
parser.add_argument(
    "--name",
    type=str,
    default=''.join(random.choices(string.ascii_lowercase, k=5)),
    help="Optional name that should be appended to the result folder name",
)
parser.add_argument(
    "--erase",
    action=argparse.BooleanOptionalAction,
    help="Do the prediction should erase existing files",
)
parser.add_argument(
    "--posteriors",
    action=argparse.BooleanOptionalAction,
    help="Do we save the posteriors",
)
parser.add_argument(
    "--disable-inference",
    type=bool,
    action=argparse.BooleanOptionalAction,
    help="Useful for getting a csv/parquet files with targets only",
    default=False,
)
parser.add_argument(
    "--disable-inference-sampling-rate",
    type=int,
    help="When inference is disabled, what sampling rate to use for discretizing targets.",
    default=49,  # 20 ms
)
parser.add_argument(
    "--step",
    type=float,
    help="Inference sliding window step size (as a %\ of window size)",
    default=0.1,
)
parser.add_argument("--batch-size", type=int, help="Inference batch size", default=128)
parser.add_argument(
    "--dataset",
    type=str,
    help="Dataset to evaluate on",
    default=DEFAULT_DATASET,
)
parser.add_argument(
    "--parquet",
    action=argparse.BooleanOptionalAction,
    help="Use the parquet format for the output (way lighter and faster)",
)
parser.add_argument(
    "--disable-rttm",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Don't save the RRTMs.",
)
parser.add_argument(
    "--debug-skip-files",
    type=int,
    default=1,
    help="Will evaluate one in N files, useful to make quick debug runs.",
)
parser.add_argument("--parameters", type=str, default="")
parser.add_argument(
    "--report-datasets",
    type=str,
    nargs="+",
    default=DEFAULT_REPORT_DATASETS,
)
parser.add_argument(
    "--db-yml-paths",
    type=str,
    nargs="+",
    help="Paths/glob patterns for pyannote.database yaml files",
    default=[PATH_TO_PYANNOTE_DB],
)
args = parser.parse_args()
if args.parameters != "":
    params = parse_params(args.parameters)
else:
    params = {}

# check args are valid
if args.disable_inference and not args.posteriors:
    print("WARNING: inference disabled AND not saving posteriors ?")
if args.dataset != DEFAULT_DATASET and args.report_datasets == DEFAULT_REPORT_DATASETS:
    print(
        "WARNING: specified --dataset but not --report-datasets, the reports probably won't be what you want !"
    )
if args.debug_skip_files < 1:
    raise ValueError("debug_skip_files must be >= 1 (for doing one in n file)")
for report_dataset in args.report_datasets:
    if len(report_dataset.split(".")) != 4:
        raise ValueError(
            f"report_dataset {report_dataset} is not a valid protocol subset name (expected DATABASE.TASK.PROTOCOL.SUBSET)"
        )

# load db yml
for path_pattern in args.db_yml_paths:
    for path in glob.glob(path_pattern):
        registry.load_database(path)

dataset = args.dataset
corpus = registry.get_protocol(
    dataset, preprocessors={"annotation": annotation_treat, "audio": FileFinder()}
)
length = len(list(corpus.test_iter()))

# place model name here
checkpoint = Path(args.checkpoint)  # Name of the checkpoint to train

# Output directory
full_exname = f"{datetime.date.today().strftime('%y_%m_%d')}-Benchmark-{args.name.replace(' ','_')}"
out_dir = os.path.join("results", "eval", full_exname)
rttm_dir = os.path.join(out_dir, "rttm")
posteriors_dir = os.path.join(out_dir, "posteriors")
reports_dir = os.path.join(out_dir, "reports")

os.makedirs(out_dir, exist_ok=True)
os.makedirs(rttm_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)
if args.posteriors:
    os.makedirs(posteriors_dir, exist_ok=True)
with open(f"{reports_dir}/config.log",'w') as log_conf:
    for arg,value in sorted(vars(args).items()):
        log_conf.write(f"Argument {arg}: {value}\n")
# Set model instance here
# m = "results/models/kmeans_100_X.Segmentation.Full.pkl"
if not args.disable_inference:
    tstart = time.time()
    model = Model.from_pretrained(checkpoint, **params)
    # Without cuda the inference is very slow, remember to do "module load cuda"
    device = torch.device("cuda")
    model.to(device)
    print(f"loaded model in {time.time() - tstart}s")

    inference = Inference(
        model,
        device=device,
        batch_size=args.batch_size,
        step=args.step * model.specifications.duration,
    )

if args.posteriors:
    extension = "posteriors" if not args.parquet else "pposteriors"
    out_posteriors = os.path.join(posteriors_dir, f"{full_exname}.{extension}")
    # posteriors = pd.DataFrame(columns=['dataset','uri',*classes,*list(map(lambda c: c+"_ref",classes))])
    posteriors: pd.DataFrame = None

for file_idx, file in enumerate(
    track(corpus.test(), total=length)
):  # Iterate on Benchmark files to generate the rttms
    show = file["uri"]
    print(f"show = {show}")

    if file_idx % args.debug_skip_files != 0:
        print("  DEBUG MODE: skipped file.")
        if args.posteriors:
            print("    WARNING : you should probably use --erase with posteriors")
        continue

    rrtm_file_path = os.path.join(rttm_dir, file["uri"] + ".rttm")
    # Do not overwrite results by default
    if os.path.exists(rrtm_file_path) and not args.erase and not args.disable_rttm:
        continue
    else:
        reference: Annotation = file["annotation"]
        uem = file["annotated"]

        output = None
        # If inference enabled (or we dont have an example sliding window), do it and write the rttm
        if not args.disable_inference:
            output = inference(file)
            output.labels = model.specifications.classes
            # add relevant classes if we predict confidence as separate output
            if output.data.shape[1] == 2 * len(output.labels):
                output.labels += [f"{c}_conf" for c in output.labels]
            elif output.data.shape[1] == len(output.labels) + 1:
                output.labels.append("conf")

            if not args.disable_rttm:
                decision = Binarize(onset=0.5)(output)
                # really only keep the predicted classes (in case we also output confidence)
                decision.subset(model.specifications.classes)
                decision.uri = file["uri"]
                decision.write_rttm(open(rrtm_file_path, "w"))

        # Posterior saving
        if args.posteriors:
            resolution = (
                output.sliding_window
                if output is not None
                else 1.0 / args.disable_inference_sampling_rate
            )
            ref_t: np.ndarray = reference.discretize(
                support=uem.extent(),
                resolution=resolution,
                labels=file["classes"],
            ).crop(uem, mode="strict")

            df_tmp = pd.DataFrame()

            if output is not None:
                # TODO: get rid of cutting the final part ?
                out_t: np.ndarray = output.crop(uem, mode="strict")[: ref_t.shape[0], :]
                ref_t = ref_t[: out_t.shape[0]]
                assert (
                    out_t.shape[0] == ref_t.shape[0]
                ), f"expected {out_t.shape }, got {ref_t.shape}"

                df_out = pd.DataFrame(out_t, columns=output.labels)
                df_tmp = df_tmp.join(df_out, how="outer")

            # add the dataset, URI and priors to the dataframe list
            ref_columns: list[str] = [f"{c}_ref" for c in file["classes"]]
            df_ref = pd.DataFrame(ref_t, columns=ref_columns)
            df_tmp = df_tmp.join(df_ref, how="outer")

            # add file info to every row of the dataframe
            df_file_info = pd.DataFrame([{"dataset": file["database"], "uri": show}])
            df_tmp = df_tmp.join(df_file_info, how="cross")

            posteriors = pd.concat((posteriors, df_tmp))

if args.posteriors and posteriors is not None:
    posteriors.rename(columns={"overlap": "ov"}, inplace=True)  # for compatibility
    posteriors.fillna(-1, inplace=True)
    if args.parquet:
        posteriors.to_parquet(out_posteriors, compression="brotli", index=False)
    else:
        posteriors.to_csv(out_posteriors, index=False)

# Can crash if not enough data ?
if not args.disable_rttm:
    try:
        evaluate(
            rttm_dir, dataset, log=os.path.join(reports_dir, "results.log")
        )  # evaluation of rttms
    except Exception as e:
        print(f"Error while evaluating {rttm_dir}: {e}")

# Generate report on posteriors (doing it after evaluate, in case it crashes)(it shouldn't)
if args.posteriors:
    if posteriors is None:
        posteriors = out_posteriors

    relevant_datasets = args.report_datasets

    protocol_subsets = protocol_fullname_to_protocol_subset_tuples(relevant_datasets)
    generate_reports(
        [posteriors],
        reports_dir,
        formats=["md", "csv", "plot"],
        protocol_subsets=protocol_subsets,
    )
    print("Created metric reports on posteriors.")
