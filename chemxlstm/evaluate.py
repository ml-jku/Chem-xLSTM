"""
given a model path, loads the model and generates 102,400 SMILES strings (if not allready generated) and evaluates them
using the SmilesGenEvaluator class.

example call:
python chemxlstm/evaluate.py --model_path ./models/xLSTM-0.0M-ed64_hid64_l1_he1_270/epoch-006 --model_class xLSTM

""" 

import argparse
import os

import torch
import wandb
import pandas as pd
import importlib
from chemxlstm import S4forNTP, xLSTMforNTP, LSTMforNTP, MambaforNTP, GPTforNTP, LlamaforNTP
from chemxlstm.metrics import SmilesGenEvaluator
import pandas as pd
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--model_class", type=str, required=False, help="Model class to use (e.g., 'S4forNTP' or S4 or 'LSTMforNTP' or LSTM)")
    parser.add_argument("--cond_generation_path", type=str, required=False, help="Path to the conditional generation model file, if any")
    parser.add_argument("--test_set_path", type=str, default="./data/chemblv31/test.zip", help="Path to the test set")

    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for generation")
    parser.add_argument("--n_designs", type=int, default=102_400, help="Number of designs to generate")

    parser.add_argument("--n_heads", type=int, default=8, help="Number of SSMs/Heads in the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--gpt_upj_factor", type=float, default=None, help="Factor to scale the GPT model UPJ")

    parser.add_argument("--recompute", action="store_true", help="Recompute the generated molecules")

    #parser.add_argument("--use_desigsn")

    args = parser.parse_args()

    save_folder = os.path.dirname(args.model_path)

    se = SmilesGenEvaluator(ground_truth_path=args.test_set_path)

    # after training generate 102,400 molecules
    #print("Generating 102,400 molecules from the last model-epoch different temperatures.")
    temperatures = [1.0] #, 1.5, 2.0
    emd = {}
    model = None
    for t in temperatures:
        import time
        # if file allready exists - load it
        if args.recompute or (not os.path.exists(f"{save_folder}/generated_molecules_{t}_{args.n_designs}.smiles")):
            print(f"Generating {(args.n_designs)} molecules with temperature {t}.")
            # load model
            model_type = args.model_class
            model = eval(model_type+"forNTP").from_file(args.model_path, n_heads=args.n_heads, device=args.device, gpt_upj_factor=args.gpt_upj_factor)
            
            start = time.time()
            
            designs, ll = model.design_molecules(n_designs=args.n_designs, batch_size=args.batch_size, temperature=t)
            time_taken = time.time() - start
            print(f"Time taken for generation: {time_taken:.2f} seconds.")

            with open(f"{save_folder}/generated_molecules_{t}_{args.n_designs}.smiles", "w") as f:
                for mol in designs:
                    f.write(mol + "\n")
            # save ll as npy
            import numpy as np
            #save_path = model_path.replace(".pth", f"_designs_T_{t}.smiles")
            
            np.save(f"{save_folder}/generated_molecules_{t}_ll.npy", ll)
        else:
            print(f"Loading generated molecules with temperature {t}.")
            designs = [line.strip() for line in open(f"{save_folder}/generated_molecules_{t}_{args.n_designs}.smiles", "r").readlines()]
            ll = np.load(f"{save_folder}/generated_molecules_{t}_ll.npy")
            time_taken = 0.0

        # scores
        eval_metrics = se.eval_designs_var(designs, n_bootstraps=2)
        eval_metrics[f"time_taken"] = time_taken
        eval_metrics[f"n_designs"] = args.n_designs
        eval_metrics[f"batch_size"] = args.batch_size
        eval_metrics[f"epoch"] = args.model_path.split("epoch-")[-1][:3]
        eval_metrics[f"model_path"] = args.model_path

        print(f"Eval metrics for temperature {t}: ", eval_metrics)
        emd[t] = eval_metrics
        #if not args.debug:
        #    wandb.log({f'{k}@t{t:2.2f}':v for k,v in eval_metrics.items()})
    # save emd to csv
    df = pd.DataFrame(emd)
    df.to_csv(f"{save_folder}/generated_molecules_metrics.csv")

    ############################################################
    # also calculate loss for the test set 
    if model is None:
        # load model
        model_type = args.model_class
        model = eval(model_type+"forNTP").from_file(args.model_path, n_heads=args.n_heads, device=args.device)
        model.model.eval()
        model.model.to(args.device)
    from chemxlstm.dataloaders import create_dataloader
    from torch import nn
    for split in ['valid', 'test']:
        val_dataloader = create_dataloader(
            f'./data/chemblv31/{split}.zip',
            mode="smiles",
            batch_size=256,
            sequence_length=model.sequence_length + 1,
            num_workers=0, # 1 -> 0 fixed segmentation fault error in Mamba
            shuffle=False,
            token2label=model.token2label,
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduce=False) # TODO PAD token should be masked
        history = {"train_loss": list(), "val_loss": list()}

        epoch_train_loss = 0
        batch_idx = 0

        import tqdm
        sample_losses_np = []
        for epoch_ix in range(1):
            # Validation
            model.model.eval()
            n_val_batches = len(val_dataloader)
            epoch_val_loss = 0

            with torch.no_grad(): # less memory usage
                for X_val, y_val in tqdm.tqdm(val_dataloader, total=n_val_batches):
                    X_val, y_val = X_val.to(args.device), y_val.to(args.device)
                    res = model._compute_loss(loss_fn, X_val, y_val)
                    per_sample_loss = (res.sum(dim=1) / (y_val != 0).sum(dim=1)).detach()

                    sample_losses_np.append(per_sample_loss.cpu().numpy().tolist())
                    batch_val_loss = per_sample_loss.mean()

        sample_losses_np = np.concatenate(sample_losses_np)
        # save as npy
        #np.save(args.model_path+"/test_losses.npy", sample_losses_np)
        # epoch-024 drop that from model_path
        # main model_path_folder without epoch
        model_fldr = args.model_path.split("/")[:-1]
        fn = f"{split}_losses.npy"
        fn = "/".join(model_fldr) + "/" + fn
        np.save(fn, sample_losses_np)
        print("saved to", fn)