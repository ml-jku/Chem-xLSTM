<p align="center">
    <img src="assets/BioxLSTM_Overview.png" alt="xlstm"/>
</p>

# Chem-xLSTM

This repository provides the code necessary to reproduce the experiments presented in the paper [Bio-xLSTM: Generative modeling, representation and in-context learning of biological and chemical sequences](https://arxiv.org/abs/2411.04165). The code is organized across the following repositories:

- [Chem-xLSTM](https://github.com/ml-jku/Chem-xLSTM/) (current repository) 
- [DNA-xLSTM](https://github.com/ml-jku/DNA-xLSTM/)
- [Prot-xLSTM](https://github.com/ml-jku/Prot-xLSTM/)

## Quickstart [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml-jku/Chem-xLSTM/blob/main/examples/chem_xlstm_minimal_colab_example.ipynb)

### Installation

```bash
git clone https://github.com/ml-jku/Chem-xLSTM.git
cd Chem-xLSTM
mamba env create -f chem_xlstm_env.yml # you can also use conda ;)
mamba activate chem_xlstm
pip install -e .
```

This package also supports the use of S4 using [s4dd](https://github.com/molML/s4-for-de-novo-drug-design), as well as Mamba and the Llama Transformer Model. If you want to enable flash attention for the transformer model, install `flash-attn` separately.

Minor: (I've had to use a minor modification to ```xlstm/blocks/slstm/src/cuda_init.py``` where ```os.environ["CUDA_LIB"] = os.path.join(os.path.split(torch.utils.cpp_extension.include_paths(cuda=True)[-1])[0], "lib")``` the cuda argument has to be dropped due to the torch version, be aware this issue might exist)

Setting up the env was the hardest part ;)

### Applications

For an easy start with Chem-xLSTM applications, we provide a sample notebooks:

- [`examples/train.ipynb`](https://github.com/ml-jku/Chem-xLSTM/blob/main/examples/train.ipynb): This notebook demonstrates how to train a Chem-xLSTM model. (currently under construction)


### Repository Structure

- `chemxlstm/`: Implementation of Chem-xLSTM.
- `data/`: Train, validation and test splits of the dataset. 
- `scripts/`: Scripts to reproduce experiments.
- `examples/`: Example notebooks for Chem-xLSTM applications.

## Pretraining

### Data

The repository comes with both datasets: [ChEMBL v31 SMILES Dataset](https://github.com/ml-jku/Chem-xLSTM/tree/main/data/chemblv31) as well as the [In-Context Style Transfer (ICST) - Conditional Molecule Generation Dataset](https://github.com/ml-jku/Chem-xLSTM/tree/main/data/icst).

### Model Training

To train an xLSTM model for example run (adjust the batch size and dim to your GPU memory):

xLSTM 1.6M model which takes  ~8.5 GB of GPU memory and training one epoch takes ~25 minutes:
```bash
python chemxlstm/train.py --model_class xLSTM --model_dim 256 --state_dim 64 --n_layers 4 --n_heads 1 --n_max_epochs 100 --batch_size 512 --device cuda:0 --learning_rate 5e-3 --dropout 0.25 --vocab_size 37 --sequence_length 100 --logdir ./models/ --training_molecules_path ./data/chemblv31/train.zip --val_molecules_path ./data/chemblv31/valid.zip --patience 5 --delta 1e-5 --save_per_epoch 3 --no_denovodesign
```

xLSTM 14.8M model
```bash
python chemxlstm/train.py --model_class=xLSTM --device=cuda:1 --n_layers=9 --model_dim=512 --n_heads=8 --batch_size=1024 --warmup_steps=4000
```

to train a model on in-context molecular generation, you just set ```--training_molecules_path=./data/icst/train.zip``` and the ```--val_molecules_path=./data/icst/valid.zip```
```bash
python ./chemxlstm/train.py --model_class=xLSTM --device=cuda:0 --n_layers=9 --model_dim=512 --n_heads=8 --logdir=./models/icst/ --vocab_size=100 --sequence_length=2048 --training_molecules_path=./data/icst/train.zip --val_molecules_path=./data/icst/valid.zip --no_denovodesign --batch_size=8 --warmup_steps=400 --permute_augmentation --learning_rate=1e-3 --patience=10 --accumulation_steps=4
```

If you want to finetune a pretrained model, set the model path: e.g.: ```--model_path=./models/xLSTM-14.8M-ed512_hid64_l9_he8_162/``` (here you need to set the n_heads (=n_heads) explicitly). The max sequence length and vocab_size of the pretrained model is automatically overwritten if stated explicitly in the command line.

## Model weights

Pretrained model weights can be downloaded [here](https://cloud.ml.jku.at/s/qpAS9iftYCN95by). It includes a pretrained xLSTM with 15M parameters each for chemblv31 and ICST.

## Evaluation

### Evaluation of unconditional generation

If you remove the parameter ```--no_denovodesign``` from the training script, this will result in generating 102,400 molecules from the last model-epoch at different temperatures. The resulting molecules can be evaluated calling:

```bash
python chemxlstm/evaluate.py --model_path ./models/xLSTM-0.0M-ed64_hid64_l1_he1_270/epoch-006 --model_class xLSTM --n_heads=1
```
The evaluation is saved as ```metrics.csv``` in the coresponding folders (as well as the cross-entropy loss per molecule as .npy).

### Conditional Generation Evaluation

To evaluate the model on the Conditional Generation ICST dataset, run the following and adjust to your parameters:

The parameter n_context_molecules is the number of molecules to be used as context for each generated molecule. The parameter n_designs is the number of designs per context molecule.

```bash
python ./chemxlstm/evaluate_cond_gen.py --model_path="./models/icst/xLSTM-14.8M-ed512_hid64_l8_he8_465/" --model_class xLSTM --device cuda:0 --n_heads 8 --batch_size 32 --context_path=./data/icst/test.zip --n_designs 1024 --mode gen --n_context_molecules 1
```

### Conditional Generation Example

We provide a simple notebook that can be used to generate molecules given a few examples using a pretrained model on the ICST dataset:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml-jku/Chem-xLSTM/blob/main/examples/chem_xlstm_few_shot_conditional_molecule_generation.ipynb)
In this case, we show 6-shot molecule generation.

## Acknowledgments

The underlying code was adapted from the [S4DD](https://github.com/molML/s4-for-de-novo-drug-design) repository. We further include original code from the [xLSTM](https://github.com/NX-AI/xlstm) repository. Thanks for the great work.

### Citation

If you found this work helpful in your project, please cite

```latex
@inproceedings{schmidinger2025bioxlstm,
    title={{Bio-xLSTM}: Generative modeling, representation and in-context learning of biological and chemical sequences},
    author={Niklas Schmidinger and Lisa Schneckenreiter and Philipp Seidl and Johannes Schimunek and Pieter-Jan Hoedt and Johannes Brandstetter and Andreas Mayr and Sohvi Luukkonen and Sepp Hochreiter and G{\"u}nter Klambauer},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=IjbXZdugdj}
}
```



