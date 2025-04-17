import re
import zipfile
import gzip
from typing import Dict, List, Union


def read_fastas(path_to_data):
    """Read fasta sequences from a file.
    
    Parameters
    -----------
    path_to_data: str
      Path to the dataset. Must be fasta file. Can be compressed as .zip or .gz.

    Returns
    -------
    sequences: lst
        List of protein sequences.  
    """

    sequences = []
    current_sequence = None

    if path_to_data.endswith(".fasta.zip"):
        with open(path_to_data, "rb") as f:
            with zipfile.ZipFile(f) as zf:
                fname = zf.namelist()[0]
                with zf.open(fname) as fasta_file:
                    # '>' indicated start of new sequence in the following line
                    for line in fasta_file: 
                        line = line.decode('utf-8').strip()
                        if line.startswith('>'):
                            if current_sequence is not None:
                                sequences.append(current_sequence)
                            current_sequence = ''
                        else:
                            current_sequence += line
                # Add last sequence
                sequences.append(current_sequence)

    elif path_to_data.endswith(".fasta.gz"):
        with gzip.open(path_to_data, "rt") as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_sequence is not None:
                        sequences.append(current_sequence)
                    current_sequence = ''
                else:
                    current_sequence += line     

    elif path_to_data.endswith(".fasta"):
        with open(path_to_data, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_sequence is not None:
                        sequences.append(current_sequence)
                    current_sequence = ''
                else:
                    current_sequence += line           

    return sequences

def segment_fasta(fasta: str, replace_rare_aas = True) -> List[str]:
    """Segment a FASTA sequence into tokens.

    Parameters
    ----------
    fasta : str
        A FASTA sequence.
    replace_rare_aas : bool
        Whether to replace rare amino acids (U, Z, O and A) with an X.
        The default is `True`.

    Returns
    -------
    List[str]
        A list of tokens.
    """
    if replace_rare_aas:
        "".join(list(re.sub(r"[UZOB]", "X", fasta)))
    return list(fasta)

def learn_label_encoding(tokenized_inputs: List[List[str]]) -> Dict[str, int]:
    """Learn a label encoding from a tokenized dataset. The padding token, `"[PAD]"` is always assigned the label 0.

    Parameters
    ----------
    tokenized_inputs : List[List[str]]
        FASTA of the protein sequence in the dataset, tokenized into a list of tokens.

    Returns
    -------
    Dict[str, int]
        A dictionary mapping FASTA tokens to integer labels.
    """
    token2label = dict()
    token2label["[PAD]"] = len(token2label)
    for inp in tokenized_inputs:
        for token in inp:
            if token not in token2label:
                token2label[token] = len(token2label)

    return token2label

        
def pad_sequences(
    sequences: List[List[Union[str, int]]],
    padding_length: int,
    padding_value: Union[str, int],
) -> List[List[Union[str, int]]]:
    """Pad sequences to a given length. The padding is done at the end of the sequences.
    Longer sequences are truncated from the beginning.

    # TODO: instead of tructation random sampling of starting position.

    Parameters
    ----------
    sequences : List[List[Union[str, int]]
        A list of sequences, either tokenized or label encoded SMILES.
    padding_length : int
        The length to pad the sequences to.
    padding_value : Union[str, int]
        The value to pad the sequences with.

    Returns
    -------
    List[List[Union[str, int]]
        The padded sequences.
    """
    lens = [len(seq) for seq in sequences]
    diffs = [max(padding_length - len, 0) for len in lens]
    padded_sequences = [
        seq + [padding_value] * diff for seq, diff in zip(sequences, diffs)
    ]
    truncated_sequences = [seq[-padding_length:] for seq in padded_sequences]

    return truncated_sequences


def augment_fasta(fasta: str, **kwargs) -> str:
    raise NotImplementedError("Augmentation of FASTA sequences is not yet implemented.")

def augment_fasta_batch(fasta: List[str], **kwargs) -> List[str]:
    raise NotImplementedError("Augmentation of FASTA sequences is not yet implemented.")
            