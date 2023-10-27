import argparse
import os
import glob
import pandas as pd
# Pyannote
from pyannote.audio import Inference
# Models architectures
from pyannote.core import Timeline
from pyannote.database.util import load_rttm
from pyannote.database.registry import registry
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure,DetectionAccuracy
from rich.progress import track
from rich.console import Console
from rich.table import Table
import shutil

# parser = argparse.ArgumentParser(description="Prediction parameters")
# parser.add_argument('folder', type=str)
# parser.add_argument('--dataset', type=str, default="X.Segmentation.Full_WP2")
# args = parser.parse_args()

def evaluate(folder,dataset,log="None",compress=False):

    path = "/gpfsdswork/projects/rech/wcp/commun/datasets/pyannote_db/"
    if not os.path.isdir(folder):
        raise ValueError(f"{folder} is not a directory")

    files = glob.glob(f"{folder}/*.rttm")
    if len(files) == 0:
        raise ValueError(f"No rttm files in {folder}")

    # Corpora mapping:
    allies_clean = [l.strip() for l in open(f"{path}/lists/ALLIES/allies_clean.lst")]
    test_D = [l.strip() for l in open(f"{path}/lists/ALLIES/Test-D.lst")]
    test_E = [l.strip() for l in open(f"{path}/lists/ALLIES/Test-E.lst")]
    dihard = [l.strip() for l in open(f"{path}/lists/DIHARD/full.test.uris.lst")]
    domains = pd.read_table(f"{path}/lists/DIHARD/recordings.tbl", comment='#', delim_whitespace=True)
    dom_dict = {}
    for ii,l in domains.iterrows():
        if l['domain'] not in dom_dict:
            dom_dict[l['domain']] = [l["uri"]]
        else:
            dom_dict[l['domain']].append(l['uri'])

    aragon_radio_t = [l.strip() for l in open(f"{path}/lists/ALBAYZIN/lists/AragonRadio/test.txt")]

    def get_score(liste, metric, scores,classes=["sp"]):
        glob_res = {l:None for l in classes}

        for f in liste:
            try:
                res = scores[f]
            except KeyError:
                continue
                #print(f"File {f} not predicted")
                #return {k:("NA","NA","NA") for k in classes}
            for c in classes:

                if glob_res[c] is None:
                    glob_res[c] = res[c]
                else:
                    glob_res[c] = {k: glob_res[c][k] + res[c][k] for k in glob_res[c] if k in res[c]}

        results = {l:None for l in classes}
        for c in classes:
            if glob_res[c] is None:
                results[c] = ("NA","NA","NA")
            else:
                results[c] = metric.compute_metrics(detail=glob_res[c])
        return results


    def get_overlap_same_sp(annot) -> "Annotation":
        """Get overlapping parts of the annotation.

        A simple illustration:

            annotation
            A |------|    |------|      |----|
            B  |--|    |-----|      |----------|
            C |--------------|      |------|

            annotation.get_overlap()
              |------| |-----|      |--------|

            annotation.get_overlap(for_labels=["A", "B"])
               |--|       |--|          |----|

        Parameters
        ----------
        labels : optional list of labels
            Labels for which to consider the overlap

        Returns
        -------
        overlap : `pyannote.core.Timeline`
           Timeline of the overlaps.
       """
        annotation = annot.subset(['sp'])

        overlaps_tl = Timeline(uri=annotation.uri)
        for (s1, t1), (s2, t2) in annotation.co_iter(annotation):
            # if labels are the same for the two segments, skipping
            if t1 >= t2:
                continue
            overlaps_tl.add(s1 & s2)
        return overlaps_tl.support().to_annotation()


    def treat(file):
	    annotation = file["annotation"]
	    mapping_speech = {label: "sp" for label in annotation.labels() if label not in ['sp', 'ov', 'mu', 'no']}
	    speech_base = annotation.rename_labels(mapping_speech)
	    ov = get_overlap_same_sp(speech_base)
	    # speech = annotation.get_timeline()  # Extract speech areas
	    # speech = speech.to_annotation()
	    mapping_ov = {label: "ov" for label in
			          ov.labels()}  # Assign the corresponding label as speakers in separated Annotations
	    # mapping_speech = {label: "speech" for label in speech.labels()}
	    ov_clean = ov.rename_labels(mapping_ov).support()
	    # speech_clean = speech.rename_labels(mapping_speech).support()
	    speech_base.update(ov_clean)  # Fuse the annotations
	    return speech_base
    scores = {}

    registry.load_database("/gpfswork/rech/wcp/commun/datasets/pyannote_db/database.yaml")
    corpus = registry.get_protocol(dataset, preprocessors={"annotation": treat})
    length = len(list(corpus.test_iter()))
    metric = DetectionPrecisionRecallFMeasure()

    for file in track(corpus.test(), total=length):
        uem = file['annotated']
        uri = file['uri']
        res_f = {}
        try:
            decision = load_rttm(f"{folder}/{uri}.rttm")[uri]
            decision = decision.rename_labels({"overlap":"ov"})
        except Exception as f:
            continue

        for c in file['classes']:
            res = {'retrieved': 0}
            res.update(metric.compute_components(file["annotation"].subset([c]), decision.subset([c]),
                                                 uem=uem))  # Evaluate the overlap part

            res_f[c] = res
        scores[uri] = res_f

    allies_clean_scores = get_score(allies_clean,metric,scores,classes=["sp","ov",'mu'])
    allies_e = get_score(test_E,metric,scores,classes=["sp","ov"])
    allies_d = get_score(test_D,metric,scores,classes=["sp","ov"])
    dihard_score = get_score(dihard,metric,scores,classes=["sp","ov"])

    dihard_dom = {l:get_score(dom_dict[l],metric,scores,classes=["sp","ov"]) for l in dom_dict}

    aragon_radio = get_score(aragon_radio_t,metric,scores,classes=["sp","mu","no"])

    global_ov = get_score(allies_clean+test_E+test_D+dihard,metric,scores,classes=["ov"])
    global_speech = get_score(allies_clean+test_E+test_D+dihard+aragon_radio_t,metric,scores,classes=["sp"])
    global_mu = get_score(allies_clean+aragon_radio_t,metric,scores,classes=["mu"])
    speech_part = Table(title="Results per partition for speech")
    speech_part.add_column("Corpus", justify="right", style="cyan", no_wrap=True)
    speech_part.add_column("Precision")
    speech_part.add_column("Recall")
    speech_part.add_column("F1-score", justify="right", style="green")

    overlap_part = Table(title="Results per partition for overlap")
    overlap_part.add_column("Corpus", justify="right", style="cyan", no_wrap=True)
    overlap_part.add_column("Precision")
    overlap_part.add_column("Recall")
    overlap_part.add_column("F1-score", justify="right", style="green")

    speech_domain = Table(title="Results per domains for Dihard in speech")
    speech_domain.add_column("Domain", justify="right", style="cyan", no_wrap=True)
    speech_domain.add_column("Precision")
    speech_domain.add_column("Recall")
    speech_domain.add_column("F1-score", justify="right", style="green")

    overlap_domain = Table(title="Results per domains for Dihard in overlap")
    overlap_domain.add_column("Domain", justify="right", style="cyan", no_wrap=True)
    overlap_domain.add_column("Precision")
    overlap_domain.add_column("Recall")
    overlap_domain.add_column("F1-score", justify="right", style="green")

    music_part = Table(title="Results per partition for music")
    music_part.add_column("Corpus", justify="right", style="cyan", no_wrap=True)
    music_part.add_column("Precision")
    music_part.add_column("Recall")
    music_part.add_column("F1-score", justify="right", style="green")

    noise_part = Table(title="Results per partition for noise")
    noise_part.add_column("Corpus", justify="right", style="cyan", no_wrap=True)
    noise_part.add_column("Precision")
    noise_part.add_column("Recall")
    noise_part.add_column("F1-score", justify="right", style="green")
    
    global_tab = Table(title="Global results")
    global_tab.add_column("Task", justify="right", style="cyan", no_wrap=True)
    global_tab.add_column("Precision")
    global_tab.add_column("Recall")
    global_tab.add_column("F1-score", justify="right", style="green")

    console = Console(record=True)
    
    speech_part.add_row("Allies_clean",
                        "NA" if allies_clean_scores['sp'][0] == 'NA' else f"{allies_clean_scores['sp'][0]:.3f}" ,
                        "NA" if allies_clean_scores['sp'][1] == 'NA' else f"{allies_clean_scores['sp'][1]:.3f}",
                        "NA" if allies_clean_scores['sp'][2] == 'NA' else f"{allies_clean_scores['sp'][2]:.3f}")
    speech_part.add_row(f"Allies_E",
                        "NA" if allies_e['sp'][0] == 'NA' else f"{allies_e['sp'][0]:.3f}",
                        "NA" if allies_e['sp'][1] == 'NA' else f"{allies_e['sp'][1]:.3f}",
                        "NA" if allies_e['sp'][2] == 'NA' else f"{allies_e['sp'][2]:.3f}")
    speech_part.add_row(f"Allies_D",
                        "NA" if allies_d['sp'][0] == 'NA' else f"{allies_d['sp'][0]:.3f}",
                        "NA" if allies_d['sp'][1] == 'NA' else f"{allies_d['sp'][1]:.3f}",
                        "NA" if allies_d['sp'][2] == 'NA' else f"{allies_d['sp'][2]:.3f}")
    speech_part.add_row(f"Dihard",
                        "NA" if dihard_score['sp'][0] == 'NA' else f"{dihard_score['sp'][0]:.3f}",
                        "NA" if dihard_score['sp'][1] == 'NA' else f"{dihard_score['sp'][1]:.3f}",
                        "NA" if dihard_score['sp'][2] == 'NA' else f"{dihard_score['sp'][2]:.3f}")
    speech_part.add_row(f"Aragon_radio",
                        "NA" if aragon_radio['sp'][0] == 'NA' else f"{aragon_radio['sp'][0]:.3f}",
                        "NA" if aragon_radio['sp'][1] == 'NA' else f"{aragon_radio['sp'][1]:.3f}",
                        "NA" if aragon_radio['sp'][2] == 'NA' else f"{aragon_radio['sp'][2]:.3f}")
    console.print(speech_part)

    overlap_part.add_row("Allies_clean",
                         "NA" if allies_clean_scores['ov'][0] == 'NA' else f"{allies_clean_scores['ov'][0]:.3f}",
                         "NA" if allies_clean_scores['ov'][1] == 'NA' else f"{allies_clean_scores['ov'][1]:.3f}",
                         "NA" if allies_clean_scores['ov'][2] == 'NA' else f"{allies_clean_scores['ov'][2]:.3f}")
    overlap_part.add_row(f"Allies_E",
                         "NA" if allies_e['ov'][0] == 'NA' else f"{allies_e['ov'][0]:.3f}",
                         "NA" if allies_e['ov'][1] == 'NA' else f"{allies_e['ov'][1]:.3f}",
                         "NA" if allies_e['ov'][2] == 'NA' else f"{allies_e['ov'][2]:.3f}")
    overlap_part.add_row(f"Allies_D",
                         "NA" if allies_d['ov'][0] == 'NA' else f"{allies_d['ov'][0]:.3f}",
                         "NA" if allies_d['ov'][1] == 'NA' else f"{allies_d['ov'][1]:.3f}",
                         "NA" if allies_d['ov'][2] == 'NA' else f"{allies_d['ov'][2]:.3f}")
    overlap_part.add_row(f"Dihard" ,
                         "NA" if dihard_score['ov'][0] == 'NA' else f"{dihard_score['ov'][0]:.3f}",
                         "NA" if dihard_score['ov'][1] == 'NA' else f"{dihard_score['ov'][1]:.3f}",
                         "NA" if dihard_score['ov'][2] == 'NA' else f"{dihard_score['ov'][2]:.3f}")
    console.print(overlap_part)

    global_tab.add_row("Overlap",
                       "NA" if global_ov['ov'][0] == 'NA' else f"{global_ov['ov'][0]:.3f}",
                       "NA" if global_ov['ov'][1] == 'NA' else f"{global_ov['ov'][1]:.3f}",
                       "NA" if global_ov['ov'][2] == 'NA' else f"{global_ov['ov'][2]:.3f}")
    global_tab.add_row("Speech",
                       "NA" if global_speech['sp'][0] == 'NA' else f"{global_speech['sp'][0]:.3f}",
                       "NA" if global_speech['sp'][1] == 'NA' else f"{global_speech['sp'][1]:.3f}",
                       "NA" if global_speech['sp'][2] == 'NA' else f"{global_speech['sp'][2]:.3f}")
    global_tab.add_row("Music",
                       "NA" if global_mu['mu'][0] == 'NA' else f"{global_mu['mu'][0]:.3f}",
                       "NA" if global_mu['mu'][1] == 'NA' else f"{global_mu['mu'][1]:.3f}",
                       "NA" if global_mu['mu'][2] == 'NA' else f"{global_mu['mu'][2]:.3f}")
    global_tab.add_row("Noise",
                       "NA" if aragon_radio['no'][0] == 'NA' else f"{aragon_radio['no'][0]:.3f}",
                       "NA" if aragon_radio['no'][1] == 'NA' else f"{aragon_radio['no'][1]:.3f}",
                       "NA" if aragon_radio['no'][2] == 'NA' else f"{aragon_radio['no'][2]:.3f}")
    console.print(global_tab)

    for d in dihard_dom:
        speech_domain.add_row(d,
                              "NA" if dihard_dom[d]['sp'][0] == 'NA' else f"{dihard_dom[d]['sp'][0]:.3f}",
                              "NA" if dihard_dom[d]['sp'][1] == 'NA' else f"{dihard_dom[d]['sp'][1]:.3f}",
                              "NA" if dihard_dom[d]['sp'][2] == 'NA' else f"{dihard_dom[d]['sp'][2]:.3f}")
    console.print(speech_domain)



    for d in dihard_dom:
        overlap_domain.add_row(d,
                               "NA" if dihard_dom[d]['ov'][0] == 'NA' else f"{dihard_dom[d]['ov'][0]:.3f}",
                               "NA" if dihard_dom[d]['ov'][1] == 'NA' else f"{dihard_dom[d]['ov'][1]:.3f}",
                               "NA" if dihard_dom[d]['ov'][2] == 'NA' else f"{dihard_dom[d]['ov'][2]:.3f}")
    console.print(overlap_domain)


    music_part.add_row(f"Aragon_radio",
                       "NA" if aragon_radio['mu'][0] == 'NA' else f"{aragon_radio['mu'][0]:.3f}",
                       "NA" if aragon_radio['mu'][1] == 'NA' else f"{aragon_radio['mu'][1]:.3f}",
                       "NA" if aragon_radio['mu'][2] == 'NA' else f"{aragon_radio['mu'][2]:.3f}")
    music_part.add_row(f"Allies_clean",
                       "NA" if allies_clean_scores['mu'][0] == 'NA' else f"{allies_clean_scores['mu'][0]:.3f}",
                       "NA" if allies_clean_scores['mu'][1] == 'NA' else f"{allies_clean_scores['mu'][1]:.3f}",
                       "NA" if allies_clean_scores['mu'][2] == 'NA' else f"{allies_clean_scores['mu'][2]:.3f}")
    console.print(music_part)
    
    noise_part.add_row(f"Aragon_radio",
                       "NA" if aragon_radio['no'][0] == 'NA' else f"{aragon_radio['no'][0]:.3f}",
                       "NA" if aragon_radio['no'][1] == 'NA' else f"{aragon_radio['no'][1]:.3f}",
                       "NA" if aragon_radio['no'][2] == 'NA' else f"{aragon_radio['no'][2]:.3f}")
    
    # result line
    res = []

    for dataset,classes in [(global_speech,["sp"]),(global_ov,["ov"]),(aragon_radio,["mu"]),(aragon_radio,["no"]),
                    (dihard_score,["sp","ov"]),(allies_clean_scores,["sp","ov","mu","no"]),(allies_e,["sp","ov"]),
                    (allies_d,["sp","ov"]),(aragon_radio,["sp","mu","no"])]:
        if dataset == "global_music":
            res.append(f"{aragon_radio['mu'][0]:.3f}".replace('.',','))
            res.append(f"{aragon_radio['mu'][1]:.3f}".replace('.',','))
            res.append(f"{aragon_radio['mu'][2]:.3f}".replace('.',','))
        elif dataset == "NA":
            res.append("NA")
            res.append("NA")
            res.append("NA")
        else:
            for c in classes:
                if c in dataset:
                    res.extend(map(lambda a:'NA' if a == 'NA' else f"{a:.3f}".replace('.',','),dataset[c]))
                else:
                    res.extend(("NA","NA","NA"))
    EPS = 1e-7
    if global_speech['sp'][2] == "NA":
        global_speech['sp'] = (EPS,EPS,EPS)

    if global_ov["ov"][2] == "NA":
        global_ov["ov"] = (EPS,EPS,EPS)
    if aragon_radio["mu"][2]=="NA":
        aragon_radio["mu"]=(EPS,EPS,EPS)
    if aragon_radio["no"][2]=="NA":
        aragon_radio["no"] = (EPS,EPS,EPS)
    harmonic_mean = ((global_speech['sp'][2]**-1 + global_ov["ov"][2]**-1 + aragon_radio["mu"][2]**-1 + aragon_radio["no"][2]**-1)/4)**-1
    console.print(f"Harmonic Mean = {harmonic_mean:.3f}")
    console.print("String to paste in the table of result")
    console.print(";".join(res))
    console.save_text(log)
    if compress:
        shutil.make_archive(folder, 'zip', folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prediction parameters")
    parser.add_argument('folder', type=str)
    parser.add_argument('--dataset', type=str, default="X.Segmentation.Benchmark")
    parser.add_argument('--zip', action=argparse.BooleanOptionalAction, help="Save as zip file ?")
    args = parser.parse_args()
    evaluate(args.folder,args.dataset,log="test.log",compress=args.zip)
