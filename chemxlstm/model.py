"""
adapted from https://github.com/molML/s4-for-de-novo-drug-design/blob/main/s4dd/s4_for_denovo_design.py
"""

import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from chemxlstm import smiles_utils, torch_callbacks
from chemxlstm.dataloaders import create_dataloader
from chemxlstm.module_library.sequence_model import SequenceModel

import torch
import tqdm

from dataclasses import dataclass, asdict, field
from typing import Any
from enum import Enum

MODES = ("smiles", "fasta")

@dataclass
class Config:
    mode: str = "smiles" # "smiles" or "fasta"
    model_dim: int = 256
    state_dim: int = 64
    n_layers: int = 4
    n_heads: int = 1
    dropout: float = 0.25
    vocab_size: int = 37
    sequence_length: int = 99
    n_max_epochs: int = 400
    learning_rate: float = 0.001
    batch_size: int = 2048
    n_augmentations: int = 0
    device: str = "cuda"

    def __post_init__(self):
        if not self.mode in MODES:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {list(MODES)}")

class AbstractModel(nn.Module):
    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        learning_rate: float,
        sequence_length: int,
        vocab_size: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.recurrent_state = None
        self.kwargs = kwargs

    def _set_model(self):
        raise NotImplementedError
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def reset_state(self, batch_size: int, device: str = None) -> None:
        raise NotImplementedError
    
    def recurrent_step(self, x_t):
        raise NotImplementedError
    

class StructuredStateSpaceSequenceModel(AbstractModel):
    """A general purpose structured state space sequence (S4) model implemented as a pytorch module."""

    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        learning_rate: float,
        sequence_length: int,
        vocab_size: int,
        **kwargs,
    ) -> None:
        """Creates a `StructuredStateSpaceSequenceModel` instance.

        Parameters
        ----------
        model_dim : int
            The dimension of the model.
        state_dim : int
            The dimension of the state in recurrent mode.
        n_layers : int
            The number of S4 layers in the model.
        n_heads : int
            The number of state space models in each layer.
        dropout : float
            The dropout rate.
        learning_rate : float
            The learning rate.
        sequence_length : int
            The length of the sequences.
        vocab_size : int
            The size of the vocabulary.
        """
        super().__init__(
            model_dim=model_dim,
            state_dim=state_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            learning_rate=learning_rate,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            **kwargs,
        )
        self._set_model()

    def _set_model(self):
        self.layer_config = [
            {
                "_name_": "s4",
                "d_state": self.state_dim,
                "n_heads": self.n_heads,
            },
            {
                "_name_": "s4",
                "d_state": self.state_dim,
                "n_heads": self.n_heads,
            },
            {"_name_": "ff"},
        ]
        self.pool_config = {"_name_": "pool", "stride": 1, "expand": None}

        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.model = SequenceModel(
            d_model=self.model_dim,
            n_layers=self.n_layers,
            transposed=True,
            dropout=self.dropout,
            layer=self.layer_config,
            pool=self.pool_config,
        )
        self.output_embedding = nn.Linear(self.model_dim, self.vocab_size)
        

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the model. The forward pass consists of embedding the
        input tokens, passing the embeddings through the S4 model (in convolutional mode), and then passing the
        output of the S4 model through a linear layer to get the logits.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of sequences of integers representing the tokens. The input shape is (batch_size, sequence_length, 1).

        Returns
        -------
        torch.Tensor
            The logits of the model.
        """
        batch = self.embedding(batch)
        batch = batch.view(batch.shape[0], self.sequence_length, self.model_dim)
        batch, state = self.model(batch, state=self.recurrent_state)
        self.recurrent_state = state
        batch = self.output_embedding(batch)
        return batch

    def reset_state(self, batch_size: int, device: str = None) -> None:
        """Resets the recurrent state of the model.
        Used in sequential mode before processing a new batch.

        Parameters
        ----------
        batch_size : int
            The batch size.
        device : str
            The device to put the state on, *e.g.,* `"cuda"` or `"cpu"`.
        """
        self.recurrent_state = self.model.default_state(batch_size, device=device)

    def recurrent_step(self, x_t):
        """Computes a single step in the recurrent mode. The internal state of the model is also updated.

        Parameters
        ----------
        x_t : torch.Tensor
            The input token. The input shape is (batch_size, 1).

        Returns
        -------
        torch.Tensor
            The logits resulting from the stepping.
        """
        x_t = self.embedding(x_t).view(x_t.shape[0], 1, self.model_dim)
        x_t = x_t.squeeze(1)
        x_t, state = self.model.step(x_t, state=self.recurrent_state)
        self.recurrent_state = state
        x_t = self.output_embedding(x_t)
        return x_t

class GPTModel(StructuredStateSpaceSequenceModel):
    def _set_model(self):
        from chemxlstm.models.gpt import GPT, GPTConfig
        self.model = GPT(GPTConfig(
            vocab_size=self.vocab_size,
            block_size=self.sequence_length,
            n_layer=self.n_layers,
            n_head=self.n_heads, 
            n_embd=self.model_dim, # NOTE state_dim is not used
            dropout=self.dropout,
            bias=True,
            upj_factor=4.0 if not hasattr(self, "gpt_upj_factor") else self.gpt_upj_factor,
        ))
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # batch is [batch_size, sequence_length, 1] or [batch_size, sequence_length]
        # if 3 dim squeeze:
        if len(batch.shape) == 3:
            batch = batch.squeeze(2)
        batch = self.model(batch)
        return batch # returns batch, seq_len, vocab_size

    def reset_state(self, batch_size: int, device: str = None) -> None:
        self.recurrent_state = None

    def recurrent_step(self, x_t):
        """
        x_t of the shape (batch_size)
        """
        B = x_t.shape[0]
        if self.recurrent_state is None:
            self.recurrent_state = x_t.view(B, 1)
        else:
            self.recurrent_state = torch.cat([self.recurrent_state, x_t.view(B, 1)], dim=1) # B, S
        
        x_t = self.forward(self.recurrent_state) # B, S, Vocab
        return x_t[:, -1, :] # B, Vocab # only return the last token prediction

class LlamaModel(GPTModel): 
    def _set_model(self):
        import transformers
        self.llama_cfg = transformers.LlamaConfig(
            vocab_size = self.vocab_size,
            hidden_size = self.model_dim, # 4096 default --> 
            intermediate_size = self.state_dim, # 11008 default # dim of NLP --> shold be larger than state_dim # 2,6*hidden_size is default
            num_hidden_layers = self.n_layers,
            num_attention_heads = self.n_heads,
            max_position_embeddings = self.sequence_length,
            return_dict=True)            
            
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.model = transformers.LlamaModel(self.llama_cfg)
        self.output_embedding = nn.Linear(self.model_dim, self.vocab_size)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        B = batch.shape[0]
        S = batch.shape[1]
        if len(batch.shape) == 3:
            batch = batch.squeeze(2)
        batch = self.model(batch)
        batch = batch.last_hidden_state
        batch = self.output_embedding(batch)
        return batch

class LATModel(GPTModel):
    """Linear Attention Transformer model.
    """
    def _set_model(self):
        from linear_attention_transformer import LinearAttentionTransformerLM
        self.model = LinearAttentionTransformerLM(
            num_tokens = self.vocab_size,
            dim = self.model_dim,
            heads = self.n_heads,
            depth = self.n_layers,
            max_seq_len = 128,
            causal = True,                  # auto-regressive or not
            ff_dropout = 0.1,               # dropout for feedforward
            attn_layer_dropout = 0.1,       # dropout right after self-attention layer
            attn_dropout = 0.1,             # dropout post-attention
            emb_dim = 128,                  # embedding factorization, to save on memory
            dim_head = 128,                 # be able to fix the dimension of each head, making it independent of the embedding dimension and the number of heads
            blindspot_size = 64,            # this gives the q(kv) attention a blindspot of 64 tokens back in the causal case, but gives back an order of magnitude return in memory savings. should be paired with local attention of at least a window size of this setting. setting this to 1 will allow for full q(kv) attention of past
            n_local_attn_heads = 4,         # number of local attention heads for (qk)v attention. this can be a tuple specifying the exact number of local attention heads at that depth
            local_attn_window_size = 1,   # receptive field of the local attention #TODO was 128
            reversible = True,              # use reversible nets, from Reformer paper
            ff_chunks = 2,                  # feedforward chunking, from Reformer paper
            ff_glu = True,                  # use GLU variant for feedforward
            attend_axially = False,         # will fold the sequence by the local attention window size, and do an extra strided attention followed by a feedforward with the cheap q(kv) attention
            shift_tokens = False             # add single token shifting, for great improved convergence
        )

class MambaModel(GPTModel):
    """
    uses the LM part of Mamba
    """
    def _set_model(self):
        from mamba_ssm import MambaLMHeadModel
        from mamba_ssm.models.config_mamba import MambaConfig
        self.config = MambaConfig(
            vocab_size=self.vocab_size-1,
            d_model=self.model_dim,
            n_layer=self.n_layers,
            ssm_cfg = {
                "d_state": self.state_dim, # state dimension 
                "d_conv": 4, # local convolutional width
                "expand": 2, # expansion factor for the blocks
            }, # TODO d_state, d_conv, expand??
            rms_norm = True,
            residual_in_fp32 = True,
            fused_add_norm = True,
            pad_vocab_size_multiple = 8,
            tie_embeddings = True,
        )

        self.model = MambaLMHeadModel(config=self.config)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # batch is [batch_size, sequence_length, 1] or [batch_size, sequence_length]
        # if 3 dim squeeze:
        if len(batch.shape) == 3:
            batch = batch.squeeze(2)
        batch = self.model(batch).logits # BS, S, V (V.. vocab size)
        return batch

class LSTMModel(StructuredStateSpaceSequenceModel):
    def _set_model(self):
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.model = nn.LSTM(
            input_size=self.model_dim,
            hidden_size=self.state_dim,
            num_layers=self.n_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.output_embedding = nn.Linear(self.state_dim, self.vocab_size)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = self.embedding(batch)
        batch = batch.view(batch.shape[0], self.sequence_length, self.model_dim)
        batch, state = self.model(batch)
        batch = self.output_embedding(batch)
        return batch
    
    def reset_state(self, batch_size: int, device: str = None) -> None:
        self.recurrent_state = None
    
    def recurrent_step(self, x_t):
        x_t = self.embedding(x_t).view(x_t.shape[0], 1, self.model_dim)
        x_t = x_t.squeeze(1)
        x_t, state = self.model(x_t.unsqueeze(1), self.recurrent_state)
        self.recurrent_state = state
        x_t = self.output_embedding(x_t)
        x_t = x_t.view(x_t.shape[0], -1)
        return x_t # shape: (batch_size, vocab_size)

class LSTMPlusModel(StructuredStateSpaceSequenceModel):
    def _set_model(self): # NO skip connections
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)

        class Block(nn.Module):
            def __init__(self, model_dim, state_dim, n_heads, dropout):
                super().__init__()
                self.ln = nn.LayerNorm(model_dim, elementwise_affine=False)
                self.lstm_heads = nn.ModuleList(
                    [
                        nn.LSTM(
                            input_size=model_dim,
                            hidden_size=state_dim,
                            num_layers=1,
                            batch_first=True,
                        )
                        for _ in range(n_heads)
                    ]
                )
                self.linear = nn.Linear(state_dim*n_heads, model_dim)
                self.out_dropout = nn.Dropout(dropout)
            
            def forward(self, x, state=None):
                """
                x of shape (batch_size, sequence_length, model_dim)
                returns x of shape (batch_size, sequence_length, model_dim), and state as list of tuples
                """
                new_state = []
                x = self.ln(x)
                xs = []
                for ii, lstm in enumerate(self.lstm_heads):
                    head, s = lstm(x, state[ii] if state is not None else None)
                    xs.append(head)
                    new_state.append(s)
                x = torch.cat(xs, dim=-1)
                x = self.linear(x)
                x = self.out_dropout(x)
                return x, new_state

        self.model = nn.ModuleList([Block(self.model_dim, self.state_dim, self.n_heads, self.dropout) for _ in range(self.n_layers)])

        self.output_embedding = nn.Linear(self.model_dim, self.vocab_size)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        batch of shape (batch_size, sequence_length)
        """
        if len(batch.shape) == 3:
            batch = batch.squeeze(2)
        batch = self.embedding(batch) # B, S, D
        for layer in self.model:
            batch, state = layer(batch) # B, S, D
        batch = self.output_embedding(batch) # B, S, V
        return batch
    
    def reset_state(self, batch_size: int, device: str = None) -> None:
        self.recurrent_state = None
    
    def recurrent_step(self, x_t):
        """
        x_t of the shape ..
        """
        x_t = self.embedding(x_t).view(x_t.shape[0], 1, self.model_dim)
        x_t = x_t.squeeze(1)
        rstates = []
        for ii, layer in enumerate(self.model):
            x_t, state = layer(x_t, self.recurrent_state[ii] if self.recurrent_state is not None else None)
            rstates.append(state)
        self.recurrent_state = rstates
        x_t = self.output_embedding(x_t)
        x_t = x_t.view(x_t.shape[0], -1)
        return x_t # shape: (batch_size, vocab_size)

class xLSTMModel(LSTMModel):
    def _set_model(self):
        from omegaconf import OmegaConf
        from dacite import from_dict
        from dacite import Config as DaciteConfig
        import sys 
        from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
        import torch
        import os

        self.xlstm_cfg = f""" 
vocab_size: {self.vocab_size}
mlstm_block:
  mlstm:
    conv1d_kernel_size: 4
    qkv_proj_blocksize: 4
    num_heads: {self.n_heads}
    dropout: {self.dropout}
slstm_block:
  slstm:
    backend: cuda
    num_heads: {self.n_heads}
    conv1d_kernel_size: 4
    bias_init: powerlaw_blockdependent
  feedforward:
    proj_factor: 1.3
    act_fn: gelu
context_length: {self.sequence_length}
num_blocks: {self.n_layers} # 7
embedding_dim: {self.model_dim} #128
slstm_at: [] #[1]
"""
        cfg = OmegaConf.create(self.xlstm_cfg)
        self.cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))
        xlstm_stack = xLSTMLMModel(self.cfg)
        self.model = xlstm_stack

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # of shape (batch_size, sequence_length)
        #batch = self.embedding(batch)
        batch = batch.view(batch.shape[0], -1)
        batch = self.model(batch) # returns batch, seq_len, vocab_size
        #batch = self.output_embedding(batch)
        return batch
    
    def reset_state(self, batch_size: int, device: str = None) -> None:
        self.recurrent_state = None

    def recurrent_step(self, x_t):
        """
        x_t of the shape (batch_size)
        """
        if hasattr(self.model, "step"):
            x_t = x_t.view(x_t.shape[0], -1) # B, S
            x_t, xlstm_state = self.model.step(x_t, state=self.recurrent_state)  # B, S, Vocab
            self.recurrent_state = xlstm_state
        else:
            if not hasattr(self, "warned"):
                self.warned = True
                print("WARNING: Using slow implementation of xLSTM: consider using the official implementation from git.")
            B = x_t.shape[0]
            if self.recurrent_state is None:
                self.recurrent_state = x_t.view(B, -1)
            else:
                self.recurrent_state = torch.cat([self.recurrent_state, x_t.view(B, -1)], dim=1) # B, S
            
            x_t = self.forward(self.recurrent_state) # B, S, Vocab
        return x_t[:, -1, :] # B, Vocab # only return the last token prediction


class sLSTMModel(xLSTMModel):
    def _set_model(self):
        from omegaconf import OmegaConf
        from dacite import from_dict
        from dacite import Config as DaciteConfig
        import sys 
        #sys.path.append('../..')
        from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
        import torch
        import os

        self.xlstm_cfg = f""" 
vocab_size: {self.vocab_size}
mlstm_block:
  mlstm:
    conv1d_kernel_size: 4
    qkv_proj_blocksize: 4
    num_heads: {self.n_heads}
    dropout: {self.dropout}
slstm_block:
  slstm:
    backend: cuda
    num_heads: {self.n_heads}
    conv1d_kernel_size: 4
    bias_init: powerlaw_blockdependent
  feedforward:
    proj_factor: 1.3
    act_fn: gelu
context_length: {self.sequence_length}
num_blocks: {self.n_layers} # 7
embedding_dim: {self.model_dim} #128
slstm_at: [1]
"""
        cfg = OmegaConf.create(self.xlstm_cfg)
        self.cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))
        xlstm_stack = xLSTMLMModel(self.cfg)
        self.model = xlstm_stack

class S4forNTP:
    """A structured state space sequence (S4) model for de novo design."""

    def __init__(
        self,
        mode: str = "smiles",
        model_dim: int = 256,
        state_dim: int = 64,
        n_layers: int = 4,
        n_heads: int = 1,
        dropout: float = 0.25,
        vocab_size: int = 37,
        sequence_length: int = 99,
        context_length: int = 99,
        n_max_epochs: int = 400,
        learning_rate: float = 0.001,
        batch_size: int = 2048,
        n_augmentations: int = 0,
        device: str = "cuda",
        gpt_upj_factor: float = 4.0,
        **kwargs,
    ) -> None:
        """Creates an `S4forNTP` instance.
        The default configurations are the ones used in the [paper](https://chemrxiv.org/engage/chemrxiv/article-details/65168004ade1178b24567cd3).

        Parameters
        ----------
        model_dim : int
            The number of dimensions used across the model.
        state_dim : int
            The dimension of the state in the recurrent mode.
        n_layers : int
            The number of S4 layers in the model.
        n_heads : int
            The number of state space models in each layer.
        dropout : float
            The dropout rate.
        vocab_size : int
            The size of the vocabulary.
        sequence_length : int
            The length of the sequences.
        n_max_epochs : int
            The maximum number of epochs to train for.
        learning_rate : float
            The learning rate.
        batch_size : int
            The batch size.
        device : str
            The device to put the model on, *e.g.,* `"cuda"` or `"cpu"`.
        """
        self.mode = mode
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.n_max_epochs = n_max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.n_augmentations = n_augmentations
        self.gpt_upj_factor = gpt_upj_factor

        self.kwargs = kwargs # for additional arguments
        self.warned_about_token = set()

        self.config = Config(
            mode=mode,
            model_dim=model_dim,
            state_dim=state_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            n_max_epochs=n_max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=device,
            n_augmentations=n_augmentations,
        )
        
        # These are set during training
        self.token2label = None
        self.label2token = None

        self.model = StructuredStateSpaceSequenceModel(
            model_dim=self.model_dim,
            state_dim=self.state_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            sequence_length=self.sequence_length,
            vocab_size=self.vocab_size,
        )

    @classmethod
    def from_file(cls, loaddir: str, **kwargs):
        """Loads an `S4forNTP` instance from a directory.

        Parameters
        ----------
        loaddir : str
            The directory to load the model from.

        Returns
        -------
        S4forNTP
            The loaded model.
        """
        with open(f"{loaddir}/init_arguments.json", "r") as f:
            properties = json.load(f)
        model_class = cls.__name__.replace("forNTP", "")
        if model_class=='S4':
            model_class = "StructuredStateSpaceSequence"
        model_class = model_class + "Model"

        print(f"Loading model from {loaddir}, using model class: {model_class}")

        properties.update(kwargs)
        print(f"Properties: {properties}")
        
        model = eval(model_class)(
            model_dim=properties["model_dim"],
            state_dim=properties["state_dim"],
            n_layers=properties["n_layers"],
            n_heads=properties["n_heads"],
            dropout=properties["dropout"],
            learning_rate=properties["learning_rate"],
            sequence_length=properties["sequence_length"],
            vocab_size=properties["vocab_size"],
            gpt_upj_factor=properties.get("gpt_upj_factor", None),
        )
        model.load_state_dict(torch.load(f"{loaddir}/model.pt"))
        token2label = properties.pop("token2label")
        label2token = properties.pop("label2token")
        instance = cls(**properties)
        instance.model = model
        #instance.model.to(instance.device)
        instance.token2label = token2label
        instance.label2token = {
            int(label): token for label, token in label2token.items()
        }
        return instance

    def _compute_loss(self, loss_fn, X, y):
        X = X.unsqueeze(2).to(self.device)
        y = y.to(self.device)
        logits = self.model(X).permute(0, 2, 1)
        return loss_fn(
            logits,
            y,
        )

    def train(
        self,
        training_molecules_path: str,
        val_molecules_path: str,
        callbacks: List[torch_callbacks.TorchCallback] = None,
        **kwargs
    ) -> Dict[str, List[float]]:
        """Trains the model. The inputs are the paths to the training and validation molecules.
        The paths should point either to a .txt file that contains one SMILES per line, or to a zip file with the same structure.
        The optional callbacks can be used to monitor or configure training.
        The training history is returned as a dictionary.

        Parameters
        ----------
        training_molecules_path : str
            The path to the training molecules. Can be a zip file or a text file. Must contain one SMILES string per line.
        val_molecules_path : str
            The path to the validation molecules. Must have the same structure as `training_molecules_path`.
        callbacks : List[torch_callbacks.TorchCallback], optional
            A list of callbacks to use during training. See the documentation of the `torch_callbacks` module for available options.

        Returns
        -------
        Dict[str, List[float]]
            A dictionary containing the training history. The keys are `train_loss` and `val_loss` and the values are lists of the metric values at each epoch.
        """
        self.model = self.model.to(self.device)
        train_dataloader = create_dataloader(
            training_molecules_path,
            mode=self.mode,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length + 1,
            num_workers=0,
            shuffle=True,
            token2label=self.token2label,
            n_augmentations=self.n_augmentations,
            add_tokens=True,
            permute_augmentation=kwargs.get("permute_augmentation", False),
            smiles_augmentation=kwargs.get("smiles_augmentation", False),
        )

        # train_dataloader.dataset.token2label = self.token2label#
        new_token2label = train_dataloader.dataset.token2label
        # change embedding if token2label has changed
        if self.token2label is not None:
            # current embedding size:
            curr_emb_dim = self.model.model.token_embedding.weight.data.shape[0]
            if len(new_token2label) > curr_emb_dim:
                print("Changing embedding and output layer to match new vocab size")
                # dependent on the model_type!!!!
                if isinstance(self.model, xLSTMModel):
                    new_emb = nn.Embedding(max(self.vocab_size, len(self.token2label)), self.model_dim).to(self.device)
                    # not all might have been used up - unly use until vocab_size
                    new_emb.weight.data[:self.vocab_size] = self.model.model.token_embedding.weight.data[:self.vocab_size]
                    self.model.model.token_embedding = new_emb
                    
                    new_output = nn.Linear(self.model_dim, self.vocab_size).to(self.device)
                    new_output.weight.data[:self.model.model.lm_head.weight.data.shape[0]] = self.model.model.lm_head.weight.data
                    self.model.model.lm_head = new_output
                else:
                    # might fail
                    new_emb = nn.Embedding(max(self.vocab_size, len(self.token2label)), self.model_dim).to(self.device)
                    new_emb.weight.data[:self.model.embedding.weight.data.shape[0]] = self.model.embedding.weight.data
                    self.model.embedding = new_emb

                    new_output = nn.Linear(self.model_dim, self.vocab_size).to(self.device)
                    new_output.weight.data[:self.model.output_embedding.weight.data.shape[0]] = self.model.output_embedding.weight.data
                    self.model.output_embedding = new_output
                self.model = self.model.to(self.device)
        
        self.token2label = train_dataloader.dataset.token2label
        self.label2token = {v: k for k, v in self.token2label.items()}


        val_dataloader = create_dataloader(
            val_molecules_path,
            mode=self.mode,
            batch_size=self.batch_size*2,
            sequence_length=self.sequence_length + 1,
            num_workers=0, # 1 -> 0 fixed segmentation fault error in Mamba
            shuffle=False,
            token2label=self.token2label,
            add_tokens=False,
            #permute_augmentation=kwargs.get("permute_augmentation", False),
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=0) # TODO PAD token should be masked

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = None
        # check if lr_scheduler is in callbacks
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, torch_callbacks.LRSchedulerCallback):
                    num_training_steps = len(train_dataloader) * self.n_max_epochs
                    # a bit ugly because it assumes the scheduler_fn has a num_training_steps argument
                    scheduler = callback.scheduler_fn(optimizer, num_training_steps=num_training_steps, **callback.scheduler_kwargs)

        history = {"train_loss": list(), "val_loss": list()}
        epoch_train_loss = 0
        batch_idx = 0
        accumulation_steps = kwargs.get("accumulation_steps", 1) # default 1
        for epoch_ix in range(self.n_max_epochs):
            self.model.recurrent_state = None
            # Training
            self.model.train()
            n_train_batches = 0
            epoch_train_loss = 0
            #for X_train, y_train in tqdm.tqdm(train_dataloader):
            for el, (X_train, y_train) in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                subsample = kwargs.get("subsample_train", 1.0)
                current_perc = el / len(train_dataloader)
                if current_perc > subsample:
                    break
                batch_train_loss = self._compute_loss(loss_fn, X_train, y_train)
                epoch_train_loss += batch_train_loss.item()
                batch_train_loss.backward()
                # add gradient clipping of 1.0
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    print(f"Gradient norm is not finite! Epoch: {epoch_ix}, batch_train_loss: {batch_train_loss}")
                # Perform optimizer step when necessary
                if ((batch_idx + 1) % accumulation_steps == 0):
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

                batch_idx += 1
                n_train_batches += 1

            epoch_train_loss = epoch_train_loss / n_train_batches
            history["train_loss"].append(epoch_train_loss)

            # Validation
            self.model.eval()
            n_val_batches = len(val_dataloader)
            epoch_val_loss = 0

            with torch.no_grad(): # less memory usage
                for X_val, y_val in val_dataloader:
                    batch_val_loss = self._compute_loss(loss_fn, X_val, y_val)
                    epoch_val_loss += batch_val_loss.item()

                epoch_val_loss = epoch_val_loss / n_val_batches
                history["val_loss"].append(epoch_val_loss)

            # Callbacks
            print(
                f"Epoch:{epoch_ix}\tLoss: {epoch_train_loss}, Val Loss: {epoch_val_loss}"
            )
            stop_training = False
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_epoch_end(epoch_ix=epoch_ix, history=history, optimizer=optimizer, model=self.model)
                stop_training_flags = [callback.stop_training for callback in callbacks]
                stop_training = stop_training | (sum(stop_training_flags) > 0)
            if stop_training:
                print("Training stopped early. Epoch:", epoch_ix)
                break

            if np.isnan(epoch_train_loss) or np.isnan(epoch_val_loss):
                print("Training diverged. Epoch:", epoch_ix)
                break

        if callbacks is not None:
            for callback in callbacks:
                callback.on_train_end(epoch_ix=epoch_ix, history=history)
        return history

    @torch.no_grad()
    def design_molecules(
        self,
        n_designs: int,
        batch_size: int,
        temperature: float,
        sequence_length: int = None,
        context: str = None,
        debug: bool = False,
    ) -> Tuple[List[str], List[float]]:
        """Designs molecules using the trained model. The number of designs to generate is specified by `n_designs`.
        The designs are generated in batches of size `batch_size`. The temperature is used to control the diversity of the generated designs.
        The designs and their log-likelihoods are returned as a tuple.

        Parameters
        ----------
        n_designs : int
            The number of designs to generate.
        batch_size : int
            The batch size to use during generation.
        temperature : float
            The temperature to use during generation.
        context : str, optional
            A context string to use during generation. If provided, the context is prepended to the generated designs.
            if context is provided, [BEG] is added before

        Returns
        -------
        Tuple[List[str], List[float]]
            A tuple containing the generated SMILES strings and their log-likelihoods.
        """
        if self.token2label is None or self.label2token is None:
            raise ValueError("This model is untrained.")

        self.model = self.model.to(self.device)
        for module in self.model.modules():
            if hasattr(module, "setup_step"):
                module.setup_step()
        self.model.eval()

        n_batches = math.ceil(n_designs / batch_size)
        designs, likelihoods = list(), list()
        for batch_idx in tqdm.tqdm(range(n_batches)):
            if batch_idx == n_batches - 1:
                batch_size = n_designs - batch_idx * batch_size
            X_test = (torch.zeros(batch_size, 1).to(torch.int)+self.token2label["[BEG]"])[:, 0].to(self.device)
            #X_test = (
            #    torch.zeros(batch_size, 1).to(torch.int) + self.token2label["[BEG]"]
            #)
            #X_test = X_test.to(self.device)
            self.model.reset_state(batch_size, device=self.device)
            #X_test = X_test[:, 0]

            batch_designs, batch_likelihoods = list(), list()

            ct = smiles_utils.segment_smiles(context) if context else []
            context_tokens = []
            for t in ct:
                token = self.token2label.get(t, -1)
                if token == -1:
                    raise ValueError(f"Token {t} not in vocabulary. of context {context}")
                context_tokens.append(token)
            if context_tokens:
               # X_context = torch.tensor(context_tokens).to(self.device).repeat(batch_size, 1).to(torch.int).to(self.device)
               # X_test = torch.concatenate([X_test, X_context], dim=1).to(self.device)
                # SOME models might support this other's want a B by 1 matrix and not B by S as recurrent_step input #TODO
                preds = self.model.recurrent_step(X_test)
                for ii, token in enumerate(context_tokens):
                    X_test = (torch.zeros(batch_size, 1).to(torch.int)+token)[:, 0].to(self.device)
                    batch_designs.append([token for _ in range(batch_size)])
                    batch_likelihoods.append([1.0 for _ in range(batch_size)])
                    # if it is the last, don't forward it:
                    if ii != len(context_tokens)-1:
                        preds = self.model.recurrent_step(X_test)

            sequence_length = self.sequence_length if sequence_length is None else sequence_length
                
            for __ in range(sequence_length-len(context_tokens)):
                preds = self.model.recurrent_step(X_test)
                softmax_preds = F.softmax(preds, dim=-1).detach().cpu().numpy().tolist()
                preds = preds.detach().cpu().numpy().tolist()
                token_labels, token_likelihoods = list(), list()
                for pred_idx, pred in enumerate(preds):
                    pred_temperature = np.exp(np.array(pred) / temperature).tolist()
                    pred_sum = sum(pred_temperature)+1e-6
                    pred_normed = [p / pred_sum for p in pred_temperature]
                    # replace nan with 0
                    pred_normed = [0 if np.isnan(p) else p for p in pred_normed]
                    probas = np.random.multinomial(1, pred_normed)
                    token_label = np.argmax(probas)
                    token_labels.append(token_label)

                    token_likelihood = softmax_preds[pred_idx][token_label]
                    token_likelihoods.append(token_likelihood)

                batch_designs.append(token_labels)
                batch_likelihoods.append(token_likelihoods)
                X_test = torch.tensor(token_labels).to(self.device)

            designs.append(np.array(batch_designs).T)
            likelihoods.append(np.array(batch_likelihoods).T)

        designs = np.concatenate(designs, axis=0).tolist()

        if debug:
            for m in designs:
                print(m)

        molecules = [
            [
                self.label2token.get(label, "[UNK]")
                for label in design
                if self.label2token.get(label, "[UNK]") not in ["[BEG]", "[PAD]"] #"[END]",
            ]
            for design in designs
        ]
        # remove [END] token
        # print if there is something after [END]
        if debug:
            for mol in molecules:
                # stuff after END:
                mol = ''.join(mol)
                sp = mol.split("[END]")
                if len(sp) > 1:
                    print("Warning: [END] token is not at the end of the molecule", mol)
                    
        molecules = [mol[: mol.index("[END]")] if "[END]" in mol else mol for mol in molecules]
        molecule_lens = [
            len(molecule) + 2 for molecule in molecules
        ]  # +2 for [BEG] and [END]
        smiles = ["".join(molecule) for molecule in molecules]
        loglikelihoods = np.log(np.concatenate(likelihoods, axis=0)).tolist()
        mean_loglikelihoods = [
            np.mean(ll[: mol_len - 1])
            for ll, mol_len in zip(loglikelihoods, molecule_lens)
        ]

        return smiles, mean_loglikelihoods

    @torch.no_grad()
    def compute_molecule_loglikelihoods(
        self, molecules: List[List[str]], batch_size: int
    ) -> List[float]:
        """Computes the log-likelihoods of a list of molecules. The molecules are processed in batches of size `batch_size`.
        The log-likelihoods are returned as a list.

        Parameters
        ----------
        molecules : List[List[str]]
            A list of SMILES strings.
            The input molecules are tokenized and padded (or truncated) internally to the sequence length used during training.
        batch_size : int
            The batch size to use during computation.

        Returns
        -------
        List[float]
            A list of log-likelihoods.
        """
        tokenized_molecules = [
            ["[BEG]"] + smiles_utils.segment_smiles(smiles) + ["[END]"]
            for smiles in molecules
        ]
        # warn if tokens are too long and get cut off
        padded_molecules = smiles_utils.pad_sequences(
            tokenized_molecules, self.sequence_length + 1, padding_value="[PAD]", warn=True
        )
        def tok2lapel_with_warning(token):
            if token not in self.token2label:
                if token not in self.warned_about_token:
                    self.warned_about_token.add(token)
                    print(f"Warning: token {token} not in token2label using PAD token for now.")
                return self.token2label["[PAD]"] 
            return self.token2label[token]
        label_encoded_molecules = [
            [tok2lapel_with_warning(token) for token in tokens] for tokens in padded_molecules
        ]

        self.model = self.model.to(self.device)
        for module in self.model.modules():
            if hasattr(module, "setup_step"):
                module.setup_step()

        self.model.eval()
        n_batches = math.ceil(len(molecules) / batch_size)
        all_sequence_loglikelihoods = list()
        for batch_idx in range(n_batches):
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = (batch_idx + 1) * batch_size
            molecule_batch = label_encoded_molecules[batch_start_idx:batch_end_idx]
            # let's do it in one pass instead of recurrent mode
            X = torch.tensor(molecule_batch, dtype=torch.int).to(self.device)
            y = X[:, 1:]
            logits = self.model(X[:, :-1])
            log_probs = F.log_softmax(logits, dim=-1)  # Log-softmax to get log probabilities
            shifted_input_ids = y.long()  # Shift input to align with labels
            log_likelihood = log_probs.gather(-1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
            # ignore padding tokens
            mask = y != self.token2label["[PAD]"]
            log_likelihood = log_likelihood * mask
            log_likelihood = log_likelihood.sum(dim=1) / mask.sum(dim=1)
            all_sequence_loglikelihoods.extend(log_likelihood.tolist())
            

            """
            self.model.reset_state(
                batch_size=len(molecule_batch), device=self.device
            )

            batch_loglikelihoods = list()
            for label_idx in range(self.sequence_length):
                labels = [molecule[label_idx] for molecule in molecule_batch]
                X_test = torch.tensor(labels, dtype=torch.int).to(self.device)

                preds = self.model.recurrent_step(X_test)
                softmax_preds = F.softmax(preds, dim=-1).detach().cpu().numpy().tolist()
                log_preds = np.log(softmax_preds)

                next_token_labels = [
                    molecule[label_idx + 1] for molecule in molecule_batch
                ]
                log_likelihoods = [
                    log_pred[nt_label]
                    for nt_label, log_pred in zip(next_token_labels, log_preds)
                ]
                batch_loglikelihoods.append(log_likelihoods)

            batch_loglikelihoods = np.array(batch_loglikelihoods).T.tolist()
            molecule_lengths = [
                len(molecule)
                for molecule in tokenized_molecules[batch_start_idx:batch_end_idx]
            ]
            batch_sequence_loglikelihoods = [
                np.mean(ll[: mol_len - 1])
                for ll, mol_len in zip(batch_loglikelihoods, molecule_lengths)
            ]
            all_sequence_loglikelihoods.extend(batch_sequence_loglikelihoods)
        """
        return all_sequence_loglikelihoods

    def save(self, path: str):
        """Saves the model to a directory. The directory will be created if it does not exist.

        Parameters
        ----------
        path : str
            The directory to save the model to.
        """
        print("Saving model to", path)
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/model.pt")
        properties = {p: v for p, v in self.__dict__.items() if p not in ("model", "embedding", "config")}
        properties["model_class"] = str(self.model.__class__)
        # if it's a set convert it to list
        for k, v in properties.items():
            if isinstance(v, set):
                properties[k] = list(v)

        # TODO currently n_heads is saved as 1 allthough it is not

        with open(f"{path}/init_arguments.json", "w") as f:
            json.dump(properties, f, indent=4)


class LSTMforNTP(S4forNTP):
    def __init__(
        self,
        mode: str = "smiles",
        model_dim: int = 256,
        state_dim: int = 64,
        n_layers: int = 4,
        n_heads: int = 1, # TODO rename model-agnoistic
        dropout: float = 0.25,
        vocab_size: int = 37,
        sequence_length: int = 99,
        n_max_epochs: int = 400,
        learning_rate: float = 0.001,
        batch_size: int = 2048,
        n_augmentations: int = 0,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(
            mode=mode,
            model_dim=model_dim,
            state_dim=state_dim,
            n_layers=n_layers,
            n_heads=1,
            dropout=dropout,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            n_max_epochs=n_max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_augmentations=n_augmentations,
            device=device,
        )
        self.model = LSTMModel(
            model_dim=self.model_dim,
            state_dim=self.state_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            sequence_length=self.sequence_length,
            vocab_size=self.vocab_size,
            n_heads=n_heads,
        )

class LSTMPlusforNTP(LSTMforNTP):
    pass

class xLSTMforNTP(LSTMforNTP):
    def __init__(
        self,
        mode: str = "smiles",
        model_dim: int = 256,
        state_dim: int = 64,
        n_layers: int = 4,
        n_heads: int = 1, # TODO rename model-agnoistic
        dropout: float = 0.25,
        vocab_size: int = 37,
        sequence_length: int = 99,
        n_max_epochs: int = 400,
        learning_rate: float = 0.001,
        batch_size: int = 2048,
        n_augmentations: int = 0,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(
            mode=mode,
            model_dim=model_dim,
            state_dim=state_dim,
            n_layers=n_layers,
            n_heads=1,
            dropout=dropout,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            n_max_epochs=n_max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_augmentations=n_augmentations,
            device=device,
            kwargs=kwargs,
        )
        self.model = xLSTMModel(
            model_dim=self.model_dim,
            state_dim=self.state_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            n_heads=n_heads, #n_heads
            learning_rate=self.learning_rate,
            sequence_length=self.sequence_length,
            vocab_size=self.vocab_size,
        )

class sLSTMforNTP(LSTMforNTP):
    def __init__(
        self,
        mode: str = "smiles",
        model_dim: int = 256,
        state_dim: int = 64,
        n_layers: int = 4,
        n_heads: int = 1, # TODO rename model-agnoistic
        dropout: float = 0.25,
        vocab_size: int = 37,
        sequence_length: int = 99,
        n_max_epochs: int = 400,
        learning_rate: float = 0.001,
        batch_size: int = 2048,
        n_augmentations: int = 0,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(
            mode=mode,
            model_dim=model_dim,
            state_dim=state_dim,
            n_layers=n_layers,
            n_heads=1,
            dropout=dropout,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            n_max_epochs=n_max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_augmentations=n_augmentations,
            device=device,
            kwargs=kwargs,
        )
        self.model = sLSTMModel(
            model_dim=self.model_dim,
            state_dim=self.state_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            n_heads=n_heads, #n_heads
            learning_rate=self.learning_rate,
            sequence_length=self.sequence_length,
            vocab_size=self.vocab_size,
        )

class GPTforNTP(S4forNTP):
    def __init__(
        self,
        mode: str = "smiles",
        model_dim: int = 256,
        state_dim: int = 64,
        n_layers: int = 4,
        n_heads: int = 1,
        dropout: float = 0.25,
        vocab_size: int = 37,
        sequence_length: int = 99,
        n_max_epochs: int = 400,
        learning_rate: float = 0.001,
        batch_size: int = 2048,
        n_augmentations: int = 0,
        device: str = "cuda",
        gpt_upj_factor: float = 4.0,
        **kwargs,
    ) -> None:
        super().__init__(
            mode=mode,
            model_dim=model_dim,
            state_dim=state_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            n_max_epochs=n_max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_augmentations=n_augmentations,
            device=device,
            gpt_upj_factor=gpt_upj_factor,
            kwargs=kwargs,
        )
        self.model = GPTModel(
            model_dim=self.model_dim,
            state_dim=self.state_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            sequence_length=self.sequence_length,
            vocab_size=self.vocab_size,
            n_heads=n_heads,
            gpt_upj_factor=gpt_upj_factor
        )

class LATforNTP(GPTforNTP):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = LATModel(
            model_dim=self.model_dim,
            state_dim=self.state_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            sequence_length=self.sequence_length,
            vocab_size=self.vocab_size,
            n_heads=n_heads,
        )

class MambaforNTP(GPTforNTP):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        config = self.config
        self.model = MambaModel(
            model_dim=config.model_dim,
            state_dim=config.state_dim,
            n_layers=config.n_layers,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            sequence_length=config.sequence_length,
            vocab_size=config.vocab_size,
            n_heads=config.n_heads,
        )

class LlamaforNTP(GPTforNTP):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = LlamaModel(
            model_dim=self.model_dim,
            state_dim=self.state_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            sequence_length=self.sequence_length,
            vocab_size=self.vocab_size,
            n_heads=self.n_heads,
        )