from pyannote.core import Annotation, Timeline


def get_overlap_same_sp(annot: Annotation) -> Annotation:
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
    annotation = annot.subset(["sp"])

    overlaps_tl = Timeline(uri=annotation.uri)
    for (s1, t1), (s2, t2) in annotation.co_iter(annotation):
        # if labels are the same for the two segments, skipping
        if t1 >= t2:
            continue
        overlaps_tl.add(s1 & s2)
    return overlaps_tl.support().to_annotation()
