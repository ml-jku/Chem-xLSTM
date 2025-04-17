"""
adapted from https://github.com/molML/s4-for-de-novo-drug-design/blob/main/s4dd/dataloaders.py
"""
from typing import Dict, List, Tuple

import torch

from . import smiles_utils
from . import fasta_utils

class PaddedLabelEncodedDataset(torch.utils.data.Dataset):
    """A dataset that returns a tuple of `(X, y)` where `X` and `y` are both
    torch tensors. `X` is a sequence of integers representing the SMILES
    tokens, and `y` is the same sequence shifted by one position to the
    right.

    The outputs are padded to the same length and label encoded.
    """

    def __init__(
        self,
        label_encoded_molecules: List[List[int]],
        token2label: Dict[str, int],
        smi_dataset: List[str] = None,
        permute_augmentation: bool = False,
        smiles_augmentation: bool = False,
        sequence_length: int = 100,
    ):
        """Creates a `PaddedLabelEncodedDataset`.

        Parameters
        ----------
        label_encoded_molecules : List[List[int]]
            A list of label encoded and padded molecules, where each molecule is a list of
            integers representing the SMILES tokens. The integers are the labels of the
            tokens in the token2label dictionary. All molecules must be padded to the same
            length.
        token2label : Dict[str, int]
            A dictionary mapping SMILES tokens to integer labels.
        """
        self.label_encoded_molecules = label_encoded_molecules
        self.token2label = token2label
        self.permute_augmentation = permute_augmentation
        self.smiles_augmentation = smiles_augmentation
        self.smi_dataset = smi_dataset
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        """Returns the number of molecules in the dataset.

        Returns
        -------
        int
            Number of molecules in the dataset.
        """
        return len(self.label_encoded_molecules)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a tuple of `(X, y)` where `X` and `y` are both torch tensors. `X` is a
        sequence of integers representing the SMILES tokens, and `y` is the same
        sequence shifted by one position to the right.

        Parameters
        ----------
        idx : int
            Index of the molecule to return.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple of `(X, y)` where `X` and `y` are both torch tensors. `X` is a sequence of
            integers representing the SMILES tokens, and `y` is the same sequence shifted
            by one position to the right.
        """
        molecule = self.label_encoded_molecules[idx]
        if self.smi_dataset is not None:
            smi = self.smi_dataset[idx]
            smis = smi.split(".")
            if self.smiles_augmentation:
                smis = smiles_utils.augment_smiles_batch(smis)
            if self.permute_augmentation:
                from random import shuffle
                shuffle(smis)
            if self.smiles_augmentation or self.permute_augmentation:
                smiles = ".".join(smis)
                # tokenize:
                tok = ["[BEG]"] + smiles_utils.segment_smiles(smiles) + ["[END]"]
                tok = [self.token2label.get(t, 0) for t in tok]
                # pad:
                tok = smiles_utils.pad_sequences([tok], self.sequence_length, padding_value=0)[0] # PAD token is allways 0
                molecule = tok
        elif self.permute_augmentation: # we have to perform permutation on token level for this if there is no smi_dataset
            #print("before permute augmentation")
            #label2token = {v: k for k, v in self.token2label.items()}
            #print(''.join([label2token[l] for l in molecule]))
            # IGNORES $0$ and $1$ at the beginning
            permute_idx = self.token2label["."]
            ignore_idxs = [self.token2label[t] for t in ["[PAD]", "[BEG]", "[END]"]]
            # check if it starts with $0$ or $1$
            add_dollar = None
            if molecule[1]==self.token2label["$"] and molecule[3]==self.token2label("$"):
                add_dollar = molecule[:4]
                molecule = molecule[4:]
            mols = []
            new_mol = []
            for el in molecule:
                if el == permute_idx:
                    mols.append(new_mol.copy())
                    new_mol = []
                elif el not in ignore_idxs:
                    new_mol.append(el)
            # permute mols list
            from random import shuffle
            shuffle(mols)
            new_molecule = [self.token2label["[BEG]"]]
            if add_dollar:
                new_molecule = add_dollar
            for mi, mol in enumerate(mols):
                new_molecule += mol
                # if it's not the last element, add a dot
                if mi < (len(mols)-1):
                    new_molecule.append(permute_idx)
            new_molecule.append(self.token2label["[END]"])
            
            while len(new_molecule) < len(molecule):# add pad tokens to the end 
                new_molecule.append(self.token2label["[PAD]"])
            molecule = new_molecule
            #print("after permute augmentation")
            #print(''.join([label2token[l] for l in molecule]))

        X = torch.tensor(molecule[:-1])
        y = torch.tensor(molecule[1:])
        return X, y


def create_dataloader(
    path_to_data: str,
    mode: str, # "smiles" or "fasta"
    batch_size: int,
    sequence_length: int = 100,
    num_workers: int = 8,
    shuffle: bool = True,
    token2label: Dict[str, int] = None,
    n_augmentations: int = 0,
    add_tokens: bool = False,
    permute_augmentation: bool = False,
    smiles_augmentation: bool = False
) -> torch.utils.data.DataLoader:
    """Creates a dataloader for a dataset of SMILES strings. The input sequences will be
    tokenized, pre/appended with `"[BEG]`/`"[END]"` tokens, label encoded, and padded to the same length.

    Parameters
    ----------
    path_to_data : str
        Path to the dataset. Can be a zip file or a text file. The dataset must be a
        list of SMILES strings, one per line.
    batch_size : int
        Batch size.
    sequence_length : int, optional
        Number of tokens in the tokenized SMILES sequences. If a SMILES sequence has more tokens than this limit, it will be
        pre-truncated. If a sequence has less tokens than this, it will be post-padded with the value `"[PAD]"`.
        Note that the output sequences will be shifted by one position to the right,
        and the training sequence length will be `sequence_length - 1`.
        The default is 100.
    num_workers : int, optional
        Number of workers for the dataloader.
        The default is 8.
    shuffle : bool, optional
        Whether to shuffle the dataset.
        The default is True.
    token2label : Dict[str, int], optional
        A dictionary mapping SMILES tokens to integer labels. If `None`, the labels will
        be learned from the dataset, which is useful to create the train dataloader. The validation and test dataloaders
        should use the same `token2label` learned during the creation of the train dataloader.
        The default is `None`.
    n_augmentations : int, optional
        Number of SMILES augmentations to generate per molecule in the dataset.
        The default is 0.

    Returns
    -------
    torch.utils.data.DataLoader
        A dataloader for the dataset.
    """
    if mode == "smiles":
        pass # nothing to do
    elif mode == "fasta":
        # replace functions
        
        smiles_utils.segment_smiles = fasta_utils.segment_fasta
        smiles_utils.learn_label_encoding = fasta_utils.learn_label_encoding
        smiles_utils.pad_sequences = fasta_utils.pad_sequences
        smiles_utils.augment_smiles_batch = fasta_utils.augment_fasta_batch

    if mode == "smiles":
        if path_to_data.endswith(".zip"):
            try:
                import zipfile
                with open(path_to_data, "rb") as f:
                    with zipfile.ZipFile(f) as zf:
                        fname = zf.namelist()[0]
                        with zf.open(fname) as g:
                            dataset = g.read().decode("utf-8").splitlines()
            except:
                import pandas as pd
                dataset = pd.read_csv(path_to_data, header=None)[0].tolist()
        else:
            with open(path_to_data, "r") as f:
                dataset = f.read().splitlines()
    elif mode == "fasta":
        dataset = fasta_utils.read_fastas(path_to_data)

    if n_augmentations > 0:
        print(f"Augmenting dataset with {n_augmentations} SMILES per molecule.")
        for i in range(n_augmentations):
            dataset += smiles_utils.augment_smiles_batch(dataset)

    tokenized_dataset = [
        ["[BEG]"] + smiles_utils.segment_smiles(smiles) + ["[END]"]
        for smiles in dataset
    ]
    if token2label is None:
        token2label = smiles_utils.learn_label_encoding(tokenized_dataset)

    padded_dataset = smiles_utils.pad_sequences(
        tokenized_dataset, sequence_length, padding_value="[PAD]"
    )
    #dataset = [[token2label.get(token, 0) for token in tokens] for tokens in padded_dataset]
    # throw warning if token not found
    tok_dataset = []
    for tokens in padded_dataset:
        encoded_tokens = []
        for token in tokens:
            if token in token2label:
                encoded_tokens.append(token2label[token])
            else:
                if add_tokens:
                    print(f"Token {token} not found in token2label. Adding to vocabulary.")
                    token2label[token] = len(token2label)
                    encoded_tokens.append(token2label[token])
                else:
                    print(f"Token {token} not found in token2label. Replacing with [PAD].")
                    encoded_tokens.append(0)
        tok_dataset.append(encoded_tokens)

    return torch.utils.data.DataLoader(
        PaddedLabelEncodedDataset(
            tok_dataset,
            smi_dataset=dataset,
            token2label=token2label,
            smiles_augmentation=smiles_augmentation,
            permute_augmentation=permute_augmentation,
            sequence_length=sequence_length,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    train_loader = create_dataloader(
        "./datasets/chemblv31/train.zip",
        batch_size=16,
        sequence_length=100,
        num_workers=8,
        shuffle=True,
        token2label=None,
    )
    val_loader = create_dataloader(
        "./datasets/chemblv31/valid.zip",
        batch_size=16,
        sequence_length=100,
        num_workers=8,
        shuffle=False,
        token2label=train_loader.dataset.token2label,
    )
