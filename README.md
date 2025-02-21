<p align="center">
    <img src="assets/BioxLSTM_Overview.png" alt="xlstm"/>
</p>

# Chem-xLSTM

This repository provides the code necessary to reproduce the experiments presented in the paper [Bio-xLSTM: Generative modeling, representation and in-context learning of biological and chemical sequences](https://arxiv.org/abs/2411.04165). The code is organized across the following repositories:

- [Chem-xLSTM](https://github.com/ml-jku/Chem-xLSTM/) (current repository) 
- [DNA-xLSTM](https://github.com/ml-jku/DNA-xLSTM/)
- [Prot-xLSTM](https://github.com/ml-jku/Prot-xLSTM/)

## Quickstart

### Installation

```bash
git clone https://github.com/ml-jku/Chem-xLSTM.git
cd Chem-xLSTM
mamba env create -f chem_xlstm_env.yml # you can also use conda ;)
mamba activate chem_xlstm
pip install -e .
```

This package also supports the use of S4 using [s4dd](https://github.com/molML/s4-for-de-novo-drug-design), as well as Mamba and the Llama Transformer Model. If you want to enable flash attention for the transformer model, install `flash-attn` separately.

### Model Weights and Pre-Processed Data

Model weights and the processed dataset can be downloaded [here](https://ml.jku.at/research/Bio-xLSTM/downloads/Chem-xLSTM/data/). To reproduce the results, place the model weights in a `checkpoints/` folder and copy the dataset to the `data/` folder.

### Applications

For an easy start with Chem-xLSTM applications, we provide a sample notebooks:

- [`examples/train.ipynb`](https://github.com/ml-jku/Chem-xLSTM/blob/main/examples/train.ipynb): This notebook demonstrates how to train a Chem-xLSTM model.


### Repository Structure

- `chemxlstm/`: Implementation of Chem-xLSTM.
- `data/`: Train, validation and test splits of the dataset. 
- `scripts/`: Scripts to reproduce experiments.
- `examples/`: Example notebooks for Chem-xLSTM applications.

## Pretraining

### Data

The repository comes with both datasets: [ChEMBL v31 SMILES Dataset](https://github.com/ml-jku/Chem-xLSTM/tree/main/data/chemblv31) as well as the [In-Context Style Transfer (ICST) - Conditional Molecule Generation Dataset](https://github.com/ml-jku/Chem-xLSTM/tree/main/data/icst).

### Model Training

To train a Chem-xLSTM model, set the desired model parameters as input and run for example:

```bash
python chemxlstm/train.py
```

To train an S4 model for example run:

```bash
python chemxlstm/train.py --model_class S4 --model_dim 256 --state_dim 64 --n_layers 4 --n_ssm 1 --n_max_epochs 100 --batch_size 2048 --device cuda:0 --learning_rate 5e-3 --dropout 0.25 --vocab_size 37 --sequence_length 100 --logdir ./models/ --training_molecules_path ./datasets/chemblv31/train.zip --val_molecules_path ./datasets/chemblv31/valid.zip --patience 5 --delta 1e-5 --save_per_epoch 3 --no_denovodesign
```

## Evaluation

### Evaluation of unconditional generation

If you remove the parameter ```--no_denovodesign``` from the training script, this will result in generating 102,400 molecules from the last model-epoch at different temperatures. The resulting molecules can be evaluated calling:

```bash
python ./chemxlstm/evaluate.py
```
This evaluates all relevant files and saves a ```metrics.csv``` file in the coresponding folders.

### Conditional Generation Evaluation

To evaluate the model on the Conditional Generation ICST dataset, run the following and adjust to your parameters:

```bash
python ./chemxlstm/eval_cond_gen.py --model_path="./models/icst_v2/Mamba-14.8M-ed512_hid64_l8_he8_465/" --model_class Mamba --device cuda:0 --n_ssm 8 --batch_size 1024 --context_path=./data/icst/test.zip --n_designs 1024 --mode gen --n_context_molecules 1
```


## Acknowledgments

The underlying code was adapted from the [S4DD](https://github.com/molML/s4-for-de-novo-drug-design) repository. We further include original code from the [xLSTM](https://github.com/NX-AI/xlstm) repository. Thanks for the great work.

### Citation

If you found this work helpful in your project, please cite

```latex
@article{schmidinger2024bio-xlstm,
  title={{Bio-xLSTM}: Generative modeling, representation and in-context learning of biological and chemical  sequences},
  author={Niklas Schmidinger and Lisa Schneckenreiter and Philipp Seidl and Johannes Schimunek and Pieter-Jan Hoedt and Johannes Brandstetter and Andreas Mayr and Sohvi Luukkonen and Sepp Hochreiter and Günter Klambauer},
  journal={arXiv},
  doi = {},
  year={2024},
  url={}
}
```
