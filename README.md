# 3MAS

This code is associated with the article "3MAS: A MULTITASK, MULTILABEL, MULTIDATASET AUDIO SEGMENTATION MODEL"

This work has been done during the 2023 JSALT Workshop that took place in Le Mans.

## Collaborators

This work is a collaboration between [Alexis Plaquet (IRIT)](https://frenchkrab.github.io/), [Pablo Gimeno (Unizar)](https://sites.google.com/unizar.es/pablogj/?pli=1), Martin Lebourdais (LIUM)

## Getting Started

### Prerequisites
To run the code, you will need:

1. Python 3.9

2. pyannote.audio, preferably latest commit in develop (latest verified to work: f393546)

Any additional libraries specified in `requirements.txt`[coming soon, mainly torchmetrics]

### Installation
1. Clone this repository.
2. Install the required packages using `pip install -r requirements.txt` into your prefered environment manager
   
## Usage
This repository *heavily* rely on pyannote for the structure, those used to work with it shouldn't be fazed by the pipeline.
### Training

1. Prepare a Pyannote Database with your favorite datasets

2. Fill a path.py file containing
```python
PATH_TO_PYANNOTE_DB = "PATH_TO_YOUR_database.yaml"
PATH_TO_NOISE = "PATH_TO_THE_NOISE_AUG_DIR"
PATH_TO_MUSIC = "PATH_TO_THE_NOISE_MUS_DIR"
PATH_TO_DATA_HUB = "PATH_TO_THE_MAIN_DATA_DIR"
```

3. Use the script train_full.py

Example :

Train the 4 classes system on a data protocol named `X.Segmentation.Main`, the default model is a WavLM + TCN, the best window duration observed is 4 seconds 

```sh
python3 train_full.py --model_typ tcn --dataset X.Segmentation.Main --duration 4.0 --name dummy_output_model_name
```
If you want to only train an overlapped speech detector, for example on DIHARD III corpus.
```sh
python3 train_full.py --model_typ tcn --dataset DIHARD.SpeakerDiarization.Full --duration 4.0 --name dummy_output_model_name_overlap --classes ov
```
### Predict
1. Use the script predict_full.py

To evaluate the model `dummy_output_model_name` on the test partition of `X.Segmentation.Main`
```sh
python3 predict_full.py --dataset X.Segmentation.Main --name output_results dummy_output_model_name.ckpt
```

# NMF
This code is also associated with the paper "Explainable by-design Audio Segmentation through Non-Negative Matrix
Factorization and Probing", by Martin Lebourdais, Théo Mariotte, Antonio Almudévar, Marie Tahon, Alfonso Ortega

The probes and visualisation notebooks used for the article are available in the NMF_visualisation folder