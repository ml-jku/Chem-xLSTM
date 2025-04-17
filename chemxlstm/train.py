"""
adapted from s4dd

example usage:
python train.py --model_class S4 --model_dim 256 --state_dim 64 --n_layers 4 --n_heads 1 --n_max_epochs 100 --batch_size 2048 --device cuda:0 --learning_rate 5e-3 --dropout 0.25 --vocab_size 37 --sequence_length 100 --logdir ./models/ --training_molecules_path ./datasets/chemblv31/train.zip --val_molecules_path ./datasets/chemblv31/valid.zip --patience 5 --delta 1e-5 --save_per_epoch 3 --no_denovodesign#

# in-context fine-tuning
python ./chemxlstm/train.py --model_class=xLSTM --device=cuda:4 --batch_size=64 --training_molecules_path=./data/icst/train.zip --val_molecules_path=./data/icst/valid.zip --sequence_length=600 --logdir=./models/icst/ --vocab_size=100 --warmup_steps=3000 --learning_rate=5e-4 --model_path="./chemxlstm/models/xLSTM-14.8M-ed512_hid64_l9_he8_162/" --model_class=xLSTM --n_heads=8
"""

import argparse
import os
import importlib
import wandb
from chemxlstm.torch_callbacks import EarlyStopping, ModelCheckpoint, HistoryLogger, DenovoDesign, WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from chemxlstm.metrics import SmilesGenEvaluator
import numpy as np

import torch
from math import cos, pi

# for sLSTM you might need to ajdust the following
#os.environ["CUDA_HOME"] = "/system/apps/mlsoft/CUDA-12.2.2/" 
#os.environ["CUDA_LIB"] = "/system/apps/mlsoft/CUDA-12.2.2/lib64" 

def count_parameters(model, exclude_no_grad=True):
    if exclude_no_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def main(args):
    # Dynamically import the model class
    model_module = importlib.import_module('chemxlstm.model')
    model_class = getattr(model_module, args.model_class if "forNTP" in args.model_class else args.model_class + "forNTP")

    if args.model_path is not None:
        model = model_class.from_file(args.model_path, n_heads=args.n_heads, device=args.device, sequence_length=args.sequence_length, 
                        batch_size=args.batch_size, learning_rate=args.learning_rate, permute_augmentation=args.permute_augmentation)
        # set args to model args if present:
        replaced = {}
        model_args = model.__dict__
        model_args['device'] = args.device
        for k in args.__dict__:
            if k in model_args and (k != "model_path") and (k != "device") and (k != "n_heads"):
                print(f"Replaced args {k} with model {k}: {model_args[k]}, prev value: {getattr(args, k)}")
                setattr(args, k, model_args[k])
                replaced[k] = model_args[k]
        print(f"Replaced args with model args: {replaced}")
        
    else:
        # Create a model with parameters from argparse
        model = model_class(
            mode=args.mode,
            model_dim=args.model_dim,
            state_dim=args.state_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            
            n_max_epochs=args.n_max_epochs,
            batch_size=args.batch_size,
            device=args.device,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            vocab_size=args.vocab_size,
            sequence_length=args.sequence_length,
            accumulation_steps=args.accumulation_steps,

            permute_augmentation=args.permute_augmentation,
            smiles_augmentation=args.smiles_augmentation,

            gpt_upj_factor=args.gpt_upj_factor,
        )

    # Log the total number of parameters
    total_params = count_parameters(model.model, exclude_no_grad=False)
    print("Total number of parameters: ", total_params / 1e6, "M")
    total_params = count_parameters(model.model, exclude_no_grad=True)
    print("Total number of (trainable) parameters: ", total_params / 1e6, "M")

    randnr = np.random.randint(0, 1000)
    run_name = f"{args.model_class}-{total_params/1e6:2.1f}M-ed{args.model_dim}_hid{args.state_dim}_l{args.n_layers}_he{args.n_heads}_{randnr}"
    run_name = args.run_name_prefix + run_name + args.run_name_suffix
    # Initialize wandb and log hyperparameters
    if not args.debug: 
        #TODO change project name
        project_wandb_name = "xSMILES" if args.mode == "smiles" else "xFASTA" #TODO@fastagroup
        wandb.init(project=project_wandb_name, config=args, name=run_name)
        wandb.log({"total_parameters": total_params})
        # log the model parameter norms over training
        wandb.watch(model.model, log="all", log_freq=100)

    # calc FLOPS
    try:
        import torch
        from fvcore.nn import FlopCountAnalysis, parameter_count
        dummy_input = torch.randint(0, args.vocab_size, (1, args.sequence_length)).to(args.device) # shape (batch_size, seq_len)
        model.model.to(args.device)
        # do one forward pass to get the flops
        model.model.eval()
        with torch.no_grad():
            # count time
            import time
            start = time.time()
            out = model.model(dummy_input)
            end = time.time()
            print("Time taken for one forward pass: ", end - start, "s")
            print("Input shape: ", dummy_input.shape)
            print("Output shape: ", out.shape)
        # TODO mamba issue: https://github.com/state-spaces/mamba/issues/110

        flops = FlopCountAnalysis(model.model, dummy_input)
        gflops = flops.total() / 1e9
        print("Total FLOPS (batch_size=2048, seq-len=): ", gflops, "GFLOPS")
        None if args.debug else wandb.log({"total_flops": gflops})
    except Exception as e:
        print("Error calculating FLOPS: ", e)
        
    logdir = args.logdir + run_name + "/"

    callbacks = [
            EarlyStopping(patience=args.patience, delta=args.delta, criterion="val_loss", mode="min"),
            ModelCheckpoint(save_fn=model.save, save_per_epoch=args.save_per_epoch, basedir=logdir),
            HistoryLogger(logdir),
        ]

    if not args.debug:
        callbacks.append(WandbLogger(savedir=logdir))
        # log logdir
        wandb.log({"logdir": logdir})

    if args.warmup_steps > 0:
        # TODO add it to params of Model
        total_steps = args.n_max_epochs * 2000 #
        if '/pc24/' in args.val_molecules_path:
            total_steps = args.n_max_epochs * 2000 * 50
        print("TODO - total_steps assumed for now to be 2000 per epoch, total_steps: ", total_steps)
        from chemxlstm.torch_callbacks import LRSchedulerCallback
        import transformers
        scheduler_fn = transformers.get_cosine_with_hard_restarts_schedule_with_warmup
        lrs = LRSchedulerCallback(scheduler_fn=scheduler_fn, num_warmup_steps=args.warmup_steps, num_cycles=1)
        callbacks.append(lrs)

    if not args.no_denovodesign:
        callbacks.append(DenovoDesign(
            design_fn=lambda t: model.design_molecules(n_designs=2048, batch_size=64, temperature=t),
            basedir=logdir,
            temperatures=[1.0, 1.5, 2.0],
            per_epoch=args.save_per_epoch,
        ))

    if args.ic_eval_path is not None:
        from chemxlstm.torch_callbacks import InContextEvaluation
        callbacks.append(InContextEvaluation(
            basedir=logdir,
            per_epoch=args.save_per_epoch,
            ds_name='fsmol'
        ))
        if 'fsmol' not in args.ic_eval_path:
            print("WARNING: IC evaluation dataset not fsmol, consider changing the ds_name in InContextEvaluation callback.")

    # Pretrain the model on ChEMBL
    if args.bf16:
        model.model.half()
    history = model.train(
        training_molecules_path=args.training_molecules_path, 
        val_molecules_path=args.val_molecules_path,
        callbacks=callbacks,
        permute_augmentation=args.permute_augmentation,
        subsample_train=args.subsample_train,
        accumulation_steps=args.accumulation_steps,
    )
    return model

if __name__ == "__main__":
    print("Current working directory: ", os.getcwd())

    parser = argparse.ArgumentParser(description="Train model for denovo design")
    parser.add_argument("--mode", type=str, default="smiles", help="Mode to use (e.g., 'smiles' or 'fasta')")
    parser.add_argument("--model_class", type=str, default="xLSTM", help="Model class to use (e.g., 'S4forNTP' or S4 or 'LSTMforNTP' or LSTM)")
    
    parser.add_argument("--model_path", type=str, default=None, help="Path to a pretrained model")
    parser.add_argument("--run_name_prefix", type=str, default="", help="Prefix for the run name")
    parser.add_argument("--run_name_suffix", type=str, default="", help="Suffix for the run name")
    
    # model_dim, state_dim, n_layers, 
    parser.add_argument("--model_dim", type=int, default=256, help="Model dimension, / embedding dimension")
    parser.add_argument("--state_dim", type=int, default=64, help="State dimension, / hidden dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=1, help="Number of SSMs/ Models in parallel")
    parser.add_argument("--gpt_upj_factor", type=float, default=4.0, help="Up-Projection Factor for the GPT model MLP")
    
    parser.add_argument("--n_max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for training")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients over")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device to use for training (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate for training")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument("--vocab_size", type=int, default=37, help="Vocabulary size")
    parser.add_argument("--sequence_length", type=int, default=100, help="Sequence length")
    parser.add_argument("--context_length", type=int, default=100, help="Context length for training") #TODO add to model paparams
    parser.add_argument("--logdir", type=str, default="./models/", help="Directory to save logs and models")
    parser.add_argument("--training_molecules_path", type=str, default="./data/chemblv31/train.zip", help="Path to training molecules dataset")
    parser.add_argument("--val_molecules_path", type=str, default="./data/chemblv31/valid.zip", help="Path to validation molecules dataset")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--delta", type=float, default=1e-5, help="Delta for early stopping")
    parser.add_argument("--save_per_epoch", type=int, default=3, help="Save model every n epochs")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--permute_augmentation", action="store_true", help="Use permute augmentation")
    parser.add_argument("--smiles_augmentation", action="store_true", help="Use SMILES augmentation during training")
    parser.add_argument("--n_augmentations", type=int, default=0, help="Number of SMILES augmentations to generate per molecule in the train dataset - in preprocessing")
    parser.add_argument("--subsample_train", type=float, default=1.0, help="Subsample the training dataset by this factor")

    parser.add_argument("--ic_eval_path", type=str, default=None, help="Path to the IC (in-context) evaluation dataset, no evaluation if not provided")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
    #parser.add_argument("--total_steps", type=int, default=10000, help="Total number of steps for learning rate scheduler")
    parser.add_argument("--no_denovodesign", action="store_true", help="Use DenovoDesign callback")
    parser.add_argument("--debug", action="store_true", help="Debug mode (no wandb logging), and smaller dataset for training")

    args = parser.parse_args()

    # SOME DEFAULT CHECKS ON THE ARGS 
    if args.debug:
        if args.mode == "fasta":
            args.training_molecules_path = "./datasets/swissprot/train_min.fasta.gz"
        elif args.training_molecules_path == "./data/chemblv31/train.zip":
            args.training_molecules_path = "./data/chemblv31/mini_train.zip"
        args.n_max_epochs = 1
        args.logdir = "./models/debug/"
        print(f"Running in debug mode, setting n_max_epochs=1, training_molecules_path={args.training_molecules_path}, logdir={args.logdir} and no denovo design.") 
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        #TORCH_USE_CUDA_DSA 
        os.environ["TORCH_USE_CUDA_DSA"] = "1"

    if args.mode=="fasta" and not args.no_denovodesign:
        #TODO@fastagroup check if you want it this way
        print("No denovo design for fasta mode. Setting no_denovodesign to False.")
        args.no_denovodesign = True
        
    model = main(args)

    # load best model #TODO@all # currently it only loads the last model

    # todo move to function
    # after training generate 102,400 molecules
    if not args.no_denovodesign:
        print("Generating 102,400 molecules from the last model-epoch different temperatures.")
        n_mols = 256 if args.debug else 10240
        temperatures = [1.0, 1.5, 2.0]
        for t in temperatures:
            mols, ll = model.design_molecules(n_designs=n_mols, batch_size=16, temperature=t)
            with open(f"{args.logdir}/generated_molecules_{t}.smi", "w") as f:
                for mol in mols:
                    f.write(mol + "\n")
            # save ll as npy
            import numpy as np
            np.save(f"{args.logdir}/generated_molecules_{t}_ll.npy", ll)
            print(f"Generated {n_mols} molecules with temperature {t}.")

            # scores
            eval_metrics = SmilesGenEvaluator().eval_designs(mols)
            print(f"Eval metrics for temperature {t}: ", eval_metrics)
            if not args.debug:
                wandb.log({f'{k}@t{t:2.2f}':v for k,v in eval_metrics.items()})
