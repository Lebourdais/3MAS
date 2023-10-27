from pyannote.core import Annotation
from pyannote.database import ProtocolFile
from .pyannote_core import get_overlap_same_sp


def annotation_treat(file: ProtocolFile) -> Annotation:
    annotation = file["annotation"]
    mapping_mu = {"music": "mu"}
    mapping_speech = {
        label: "sp"
        for label in annotation.labels()
        if label not in ["sp", "ov", "mu", "music", "no"]
    }
    mapping_mu.update(mapping_speech)
    speech_base = annotation.rename_labels(mapping_mu)

    ov = get_overlap_same_sp(speech_base)
    # speech = annotation.get_timeline()  # Extract speech areas
    # speech = speech.to_annotation()
    mapping_ov = {
        label: "ov" for label in ov.labels()
    }  # Assign the corresponding label as speakers in separated Annotations
    # mapping_speech = {label: "speech" for label in speech.labels()}
    ov_clean = ov.rename_labels(mapping_ov).support()
    # speech_clean = speech.rename_labels(mapping_speech).support()
    speech_base.update(ov_clean)  # Fuse the annotations
    return speech_base
