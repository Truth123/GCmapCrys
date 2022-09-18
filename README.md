# GCmapCrys:Integrating graph attention network with predicted contact map for multi-stage protein crystallization propensity prediction

## Dataset
The dataset we use comes from DCFCrystal's BD_CRYS and the raw dataser can be obtained from this [link](http://202.119.84.36:3079/dcfcrystal/Data.html).

## Environment
- python 3.7
- pytorch 1.7
- torch-geometric 2.0.4
- biopython
- h5py
- numpy
- tqdm
- yaml
- tensorboard
## Test

**input**

1. Protein sequences have to be saved in a fasta format.
```txt
>protein_id1
XXXXXXXXXXX
>protein_id2
XXXXXXXXXXXXXX
...
```
2. The model input also requires multiple sequence-based features, and the following script can be used to obtain the corresponding feature files.
```python
python generate_featrues.py input.fasta
```

**run inference**

First you need to set the input file and feature dir in the [config/test.yaml](./config/test.yaml) configuration file. The program will go to the feature directory to find the sequence-based feature file based on the corresponding protein ID.

- input_file: "input.fasta"
- ffeature_dir: "feature"

You can also change other parameters in the configuration file according to your needs, such as *output*, *batch_size*, *device*, and *load_pth*. Then you need to inference through the following script. The output results are saved to the *out.csv* file by default

```python
python inference.py ./config/test.yaml
```

## Training

If you need to retrain the model on your own data, you will first need to reorganize your fasta file in the following format.
```
-- DATASET_NAME
  -- train
    -- sequence.fasta
    -- label.txt
  -- val
    -- sequence.fasta
    -- label.txt
  -- test
    -- sequence.fasta
    -- label.txt
```
Then you need to call the [generate_features.py](./generate_features.py) script on your own data to generate the corresponding feature files.

Finally, set the [config/trainval.yaml](./config/trainval.yaml) to suit your needs, and call the following script to start the training.
```python
python train.py ./config/trainval.yaml
```
