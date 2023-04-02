# GCmapCrys:Integrating graph attention network with predicted contact map for multi-stage protein crystallization propensity prediction

## Dataset
The dataset we use comes from DCFCrystal's BD_CRYS and the raw dataser can be obtained from this [link](http://202.119.84.36:3079/dcfcrystal/Data.html).

## Requirement
**Install  BLAST**

We use PSI-BLAST to generate  position-specific scoring matrix (PSSM) by searching the SWISS-Prot database.
```
cd pkgs
wget -c ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.13.0+-x64-linux.tar.gz
tar -zxvf ncbi-blast-2.13.0+-x64-linux.tar.gz
```

**Install SCRATCH-1D**

We use SCRATCH-1D to predict protein secondary structure and relative solvent accessibility.
```
cd pkgs
wget -c https://download.igb.uci.edu/SCRATCH-1D_1.2.tar.gz
tar -zvxf SCRATCH-1D_1.2.tar.gz
cd SCRATCH-1D_1.2
./install
```

**Install PconsC4 and HHsuite**

We use PconsC4 software to predict protein contact map. Specifically, for a query sequence , we use HHblits software to search the [UniClust30](https://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/) database to generate the corresponding multiple sequence alignment (MSA), which is further fed to PconsC4 for contact map prediction. For the installation process of [PconsC4](https://github.com/ElofssonLab/PconsC4) and [HHsuite](https://github.com/soedinglab/hh-suite), please refer to their respective official websites.

**Install IPC2**

We use IPC2.0 to calculate the protein isoelectric point (pI).
```
cd pkgs
wget -c http://ipc2.mimuw.edu.pl/ipc-2.0.1.zip
unzip ipc-2.0.1.zip
```

**python environment**
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

Protein sequences have to be saved in a fasta format.

```txt
>protein_id1
XXXXXXXXXXX
>protein_id2
XXXXXXXXXXXXXX
...
```

The model input also requires multiple sequence feature information, and the [generate_featrues.py](generate_features.py) script can be used to obtain the corresponding feature files.

**run inference**

First you need to set the input file and feature dir in the [config/test.yaml](./config/test.yaml) configuration file. The program will go to the feature directory to find the sequence-based feature file based on the corresponding protein ID.

- input_file: "input.fasta"
- feature_dir: "feature"

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
Then you need to set the [config/trainval.yaml](./config/trainval.yaml) to suit your needs, and call the following script to start the training.
```python
python train.py ./config/trainval.yaml
```
