"""
example call:
python ./chemxlstm/evaluate.py --model_path="chemxlstm/models/icst/Mamba-14.8M-ed512_hid64_l8_he8_465/" --model_class Mamba --device cuda:2 --n_heads 8 --batch_size 1024 --context_path=chemxlstm/datasets/icst/test_v2.zip --n_designs 1024 --mode gen --n_context_molecules 1

""" 

import argparse
import os
import torch
import wandb
import pandas as pd
import importlib
from model import S4forNTP, xLSTMforNTP, LSTMforNTP, MambaforNTP, GPTforNTP, LlamaforNTP
from chemxlstm.metrics import SmilesGenEvaluator
import pandas as pd
import numpy as np

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def load_model(model_path, model_type='xLSTM', args=None):
    if "epoch" not in model_path:
        # get folder with latest epoch
        import glob
        # list all folders within the model_path
        mps = glob.glob(model_path + "/epoch*")
        model_path = sorted(mps)[-1]
        print(f"Loading model from {model_path}, (taking last epoch, since none has been provided)")

    n_heads = model_path.split("he")[-1].split("_")[0]
    if args.n_heads is None:
        print(f"Setting n_heads to {n_heads}")
        n_heads = args.n_heads
    
    if args.sequence_length is not None:
        model = eval(model_type+"forNTP").from_file(model_path, n_heads=args.n_heads, device=args.device, sequence_length=args.sequence_length, gpt_upj_factor=args.gpt_upj_factor)
    elif args.model_class != "GPT":
        model = eval(model_type+"forNTP").from_file(model_path, n_heads=args.n_heads, device=args.device)
    else:
        model = eval(model_type+"forNTP").from_file(model_path, n_heads=args.n_heads, device=args.device, gpt_upj_factor=args.gpt_upj_factor)

    model.model.to(args.device).eval()
    return model

def load_file(file_path):
    import zipfile
    with open(file_path, "rb") as f:
        with zipfile.ZipFile(f) as zf:
            fname = zf.namelist()[0]
            with zf.open(fname) as g:
                dataset = g.read().decode("utf-8").splitlines()
    return dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--model_class", type=str, required=False, help="Model class to use (e.g., 'S4forNTP' or S4 or 'LSTMforNTP' or LSTM)")
    parser.add_argument("--cond_generation_path", type=str, required=False, help="Path to the conditional generation model file, if any")
    parser.add_argument("--test_set_path", type=str, default='./data/chemblv31/test.zip', help="Path to the test set file")
    
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for generation")
    parser.add_argument("--n_designs", type=int, default=102_400, help="Number of designs to generate")

    parser.add_argument("--n_heads", type=int, default=8, help="Number of SSMs/Heads in the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--gpt_upj_factor", type=float, default=None, help="Factor to scale the GPT model UPJ")

    parser.add_argument("--sequence_length", type=int, default=None, help="Sequence length for the model, will be adapte if possible")

    parser.add_argument("--sampling", type=str, default="only_actives", help="Sampling scheme to use for sampling support set, available: only_actives, random, stratified (default: only_actives)")
    parser.add_argument("--recompute", action="store_true", help="Recompute the generated molecules")

    parser.add_argument("--mode", type=str, default="vs", help="Mode to run the script in")

    parser.add_argument("--context_path", type=str, default=None, help="Path to the context file for conditional generation")
    parser.add_argument("--n_context_molecules", type=int, default=32, help="Number of context molecules to use for conditional generation")

    #parser.add_argument("--use_desigsn")

    args = parser.parse_args()

    save_folder = os.path.dirname(args.model_path)
    model_fldr = args.model_path.split("/")[:-1]

    model = load_model(args.model_path, model_type=args.model_class, args=args)

    if args.mode == "gen":
        # after training generate 102,400 molecules
        #print("Generating 102,400 molecules from the last model-epoch different temperatures.")
        temperatures = [1.0] #, 1.5, 2.0
        emd = {}
        se = SmilesGenEvaluator(ground_truth_path=args.test_set_path)

        context_list = [None]
        if args.context_path is not None:
            pass #context = open(args.context_path, "r").readlines
            context_list = load_file(args.context_path)
        
        for t in temperatures:
            import time
            # if file allready exists - load it
            for ii, context in enumerate(context_list):
                take_x_molecules = args.n_context_molecules
                print(f"cutting down the context from {len(context.split('.'))} to {take_x_molecules} molecules")
                small_context = '.'.join(context.split(".")[:take_x_molecules])+'.' if context is not None else None
                # make sure it's not longer than context_length
                seq_len = args.sequence_length if args.sequence_length is not None else model.sequence_length
                if len(small_context) > seq_len:
                    print(f"Warning: context is longer than sequence length ({len(small_context)} > {seq_len}). --> set a higher sequence length.")

                # set seq len to estimated sequence length:, take 75 percentile of molecule length
                mol_len = np.percentile([len(m) for m in context.split(".")], 75)
                new_seq_len = len(small_context) + mol_len*5
                seq_len = int(min(seq_len, new_seq_len))
                print(f"Setting sequence length to {seq_len}")
                

                save_path = f"{save_folder}/generated_molecules{'_'+small_context[:10] if small_context else ''}_{t}_{args.n_designs}.smiles"
                if args.recompute or (not os.path.exists(save_path)):
                    print(f"Generating {(args.n_designs)} molecules with temperature {t}.")
                    
                    start = time.time()

                    designs, ll = model.design_molecules(n_designs=args.n_designs, context=small_context, batch_size=args.batch_size, temperature=t, sequence_length=seq_len)
                    time_taken = time.time() - start
                    print(f"Time taken for generation: {time_taken:.2f} seconds.")

                    with open(save_path, "w") as f:
                        for mol in designs:
                            f.write(mol + "\n")
                    # save ll as npy
                    import numpy as np
                    
                    np.save(save_path.replace('.smiles','.npy'), ll)
                else:
                    print(f"Loading generated molecules with temperature {t}.")
                    designs = [line.strip() for line in open(save_path, "r").readlines()]
                    ll = np.load(save_path.replace('.smiles','.npy'))
                    time_taken = 0.0

                # take n after context as designs
                take_x_molecules = 3
                c_mols = len(small_context.split("."))-1 if context is not None else 0
                designs = [d.split(".")[c_mols:c_mols+take_x_molecules] for d in designs]
                # flatten designs
                designs = [item for sublist in designs for item in sublist]

                # scores
                if context is not None:
                    se = SmilesGenEvaluator(ground_truth_smiles=context.split(".") if context is not None else None)
                print(f"got {len(designs)} designs")
                eval_metrics = se.eval_designs_var(designs, n_bootstraps=1)
                eval_metrics["context"] = small_context#
                eval_metrics["n_context_molecules"] = take_x_molecules
                eval_metrics["temperature"] = t
                eval_metrics["context_i"] = ii
                eval_metrics[f"time_taken"] = time_taken
                eval_metrics[f"n_designs"] = args.n_designs
                eval_metrics[f"batch_size"] = args.batch_size
                eval_metrics[f"epoch"] = args.model_path.split("epoch-")[-1][:3]
                eval_metrics[f"model_path"] = args.model_path

                print(f"Eval metrics for temperature {t}: ", eval_metrics)
                emd[f"{t}_{ii}"] = eval_metrics
                #if not args.debug:
                #    wandb.log({f'{k}@t{t:2.2f}':v for k,v in eval_metrics.items()})
            # save emd to csv
            df = pd.DataFrame(emd)
            df.to_csv(f"{save_folder}/generated_molecules_metrics.csv")

    ############################################################
    # calculate loss to rank molecules from fsmol dataset

    if args.mode == "vs": # virtual screening evaluation
        print("Virtual screening evaluation - MODE")

        results = []

        ####################
        # load the dataset:
        ds_name = 'fsmol'
        df_ds = pd.read_parquet(f"/system/user/user/user/projects/clamp/data/{ds_name}/activity.parquet")
        cid2smi = pd.read_parquet(f"/system/user/user/user/projects/clamp/data/{ds_name}/compound_smiles.parquet").CanonicalSMILES.to_dict()
        df_ds['SMILES'] = df_ds['compound_idx'].map(cid2smi)

        df_ds = df_ds[df_ds.FSMOL_split=='test']
        print("aid_len",len(df_ds.assay_idx.unique()))
        #df_ds = df_ds[df_ds.assay_idx<5017]
        #print("aid_len",len(df_ds.assay_idx.unique()))
        ####################
        # run the stuff
        from tqdm import tqdm
        import numpy as np
        from sklearn import metrics

        ks = [0, 1, 2, 5, 12, 16, 32, 64, 128, 256] #[0,1,2,3,5,8,16]
        ns = [1] # number of draws of k actives

        max_assay_size = 1024
        debug = False
        df_res = pd.DataFrame()
        # let's load the old df_res if present
        df_res_fn = "/".join(model_fldr) + "/" + f"conditional_generation_kactives_{ds_name}.parquet"
        if os.path.exists(df_res_fn):
            print("Loading old results")
            df_res = pd.read_parquet(df_res_fn)

        df_ds['input'] = df_ds.apply(lambda k: f"{k.activity}${k.SMILES}", axis=1)

        for ni in ns:
            for aid, df_aid in tqdm(df_ds.groupby('assay_idx')): # [df_ds.AID==1845206]
                for k in ks:
                    # check if we have allread
                    # y computed this
                    # if it doesnt have those attributes skip
                    for att in ['dataset', 'assay_idx', 'k','n']:
                        if not hasattr(df_res, att):
                            df_res[att] = np.nan

                    tmp = df_res.loc[(df_res.dataset==ds_name) & (df_res.assay_idx==aid) & (df_res.k==k) & (df_res.n==ni)]
                    if hasattr(tmp, 'sampling'):
                        tmp.sampling = tmp.sampling.fillna('only_actives_1') # default value for old data
                        tmp = tmp[tmp.sampling==f'{args.sampling}_{ni}']
                    if len(tmp) and not args.recompute:
                        #print(f"{aid}\t{k}\t{roc_auc:.3f}\t{avgp-df_aid2.activity.mean():.3f}\t{avgp:.3f}\t{n_active}\t{n_inactive}")
                        print(f"{aid}\t{k}\t{tmp.roc_auc.values[0]:.3f}\t{tmp.davgp.values[0]:.3f}\t{tmp.avgp.values[0]:.3f}\t{np.nan}\t{np.nan} - from cache")
                    else:
                        # check sampling scheme
                        if args.sampling == "only_actives":
                            # get k actives
                            actives = df_aid[df_aid.activity==1].input.to_list()
                            if ni>1:
                                np.random.shuffle(actives, random_state=ni)
                            actives = actives[:min(k, len(actives))]
                        elif args.sampling == "random":
                            actives = df_aid.input.sample(min(k, len(df_aid)), random_state=ni).tolist()
                        elif args.sampling == "stratified":
                            # groupby activity
                            actives = df_aid[df_aid.activity==1].input.sample(min(k, len(df_aid[df_aid.activity==1])), random_state=ni).tolist()
                            inactiv = df_aid[df_aid.activity==0].input.sample(min(k, len(df_aid[df_aid.activity==0])), random_state=ni).tolist()
                            actives.extend(inactiv)
                            # permute
                            np.random.shuffle(actives, random_state=ni)
                            # print warning if not enough actives
                            if len(actives)<(k*2):
                                print(f"Warning: assay {aid} has only {len(actives)} actives.")
                        
                        df_aid2 = df_aid[~df_aid.input.isin(actives)]
                        if len(df_aid2.activity.unique())!=2:
                            print(f"Warning: assay {aid} has only one class; Cannot be evaluated.")
                            continue
                        else:
                            actives_str = '.'.join(actives)
                            jst = '.' if len(actives)>0 else ''

                            # get k actives
                            #actives = df_aid[df_aid.activity==1].SMILES.tolist()

                            # icfsmol: syntax:  $1$CC.CN.CCC $0$CC.CN.CCC for active/inactive
                            # icfsmol2: syntax: 1$CC.0$CN.1$N.1$CC with act/inavt in one sequence

                            # if $ is in vocab:
                            # input = ('$1$' + actives_str + jst + df_aid2.SMILES).values.tolist()
                            # input = ('$1$' + actives_str + jst + df_aid2.SMILES).values.tolist()

                            input = actives_str + jst + "1$" + df_aid2.SMILES

                            # some vocab changes:
                            #input = [ipt.replace('c','C') for ipt in input]

                            # compute ll
                            ll = model.compute_molecule_loglikelihoods(input, batch_size=args.batch_size)
                            if debug:
                                import matplotlib.pyplot as plt
                                plt.scatter(ll, [len(i) for i in input])
                                sorted_ll_idx = np.argsort(ll)[::-1]
                                #Error: only integer scalar arrays can be converted to a scalar index
                                #print(np.array(ll)[sorted_ll_idx[:10]], np.array(input)[sorted_ll_idx[:10]])
                            # compute metrics
                            roc_auc = metrics.roc_auc_score(df_aid2.activity, ll) if len(set(df_aid2.activity))>1 else np.nan
                            avgp =  metrics.average_precision_score(df_aid2.activity, ll) if len(set(df_aid2.activity))>1 else np.nan
                            df_res = pd.concat([df_res, pd.DataFrame(
                                {
                                    'dataset': ds_name, 'assay_idx': aid, 'k': k, 'n': ni,
                                    'roc_auc': roc_auc,
                                    'davgp': avgp-df_aid2.activity.mean(),
                                    'avgp': avgp,
                                    'model_path': args.model_path,
                                    'model_class': args.model_class,
                                    'sequence_length': args.sequence_length,
                                    'sampling': f'{args.sampling}_{ni}',
                                    #'ll': np.array(ll),
                                # 'actives_str': actives_str,
                                # 'input': input,
                                #  'activity': df_aid2.activity.tolist(),
                                }, index=[0]
                                )])
                            
                            n_active = df_aid2.activity.sum()
                            n_inactive = len(df_aid2) - n_active
                            print(f"{aid}\t{k}\t{roc_auc:.3f}\t{avgp-df_aid2.activity.mean():.3f}\t{avgp:.3f}\t{n_active}\t{n_inactive}")
                            
                            
                            df_res.to_parquet(df_res_fn)