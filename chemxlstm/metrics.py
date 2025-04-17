from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from fcd_torch import FCD
import pandas as pd
import zipfile
import os
import unittest
import pandas as pd
import numpy as np

# turn off RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def load_smiles_from_zip(zip_path):
    """
    load zip file via pandas
    :param zip_path: Path to the zip file.
    :return: List of SMILES strings.
    """
    df = pd.read_csv(zip_path, compression='zip', header=None)
    return df[0].tolist()

class SmilesGenEvaluator():
    def __init__(self, ground_truth_path='./data/chemblv31/test.zip', ground_truth_smiles=None):
        self.ground_truth_path = ground_truth_path
        self.val_li_of_smi = load_smiles_from_zip(ground_truth_path)
        if ground_truth_smiles is not None:
            self.val_li_of_smi = ground_truth_smiles
        self.fcd = FCD(device='cpu', n_jobs=8)

    def calculate_sa_score(self, mol):
        """
        Calculate the Synthetic Accessibility score for a molecule.
        :param mol: RDKit molecule object
        :return: SA score
        """
        from rdkit import Chem
        from rdkit.Chem import RDConfig
        import os
        import sys
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        # now you can import sascore!
        import sascorer
        try:
            return sascorer.calculateScore(mol)
        except Exception as e:
            print(f"Error calculating SA score: {e}")
            return None

    def calculate_circle_diversity(self, list_of_smiles, distance_threshold=0.7, algorithm="maxmin"):
        # https://openreview.net/pdf?id=YO3d6e0ahp
        # calculate circle-diversity, or IntDiv internal diversity or #Circles
        # from Renz et al. code currently not publicly available
        import os
        # add path to the python module to the system path
        os.sys.path.append(os.path.join(os.getcwd(), '/system/user/user/user/projects/bio-x-lstm/diverse-efficiency'))
        from divopt.evaluation.process_results import compute_diverse_solutions

        idx_divers = compute_diverse_solutions(list_of_smiles, distance_threshold=distance_threshold, algorithm=algorithm)
        return len(idx_divers)

    def eval_designs(self, list_of_smiles):
        """
        Evaluate the designs using RDKit.
        :param list_of_smiles: List of SMILES strings to evaluate.
        :param ground_truth_list_of_smiles: List of ground truth SMILES strings for FCD calculation.
        :return: Dictionary with evaluation metrics.
        """
        metrics = {}

        mols = [Chem.MolFromSmiles(smiles) for smiles in list_of_smiles]
        # add try-catch over it
        inchii_keys = []
        for m in mols:
            try:
                if m is not None:
                    inchii_keys.append(Chem.MolToInchiKey(m))
            except Exception as e:
                print(f"Error calculating InChI key: {e}")
        #inchii_keys = [Chem.MolToInchiKey(m) if m is not None else '' for m in mols]
        valid_mols = [m for m in mols if m is not None]
        valid_smiles = [smi for smi, m in zip(list_of_smiles, mols) if m is not None]

        metrics["valid"] = len(valid_mols)
        metrics["unique"] = len(set(list_of_smiles))
        metrics["valid_unique"] = len(set(valid_smiles))

        metrics["unique_inchii_keys"] = len(set(inchii_keys)) # InChI key normalizes tautomers 

        metrics["novel"] = len(set(valid_smiles) - set(self.val_li_of_smi))

        # calc nr. of murcko scaffolds
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit.Chem import AllChem
        scaffolds = []
        for m in valid_mols:
            try:
                scaffolds.append(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)))
            except Exception as e:
                print(f"Error calculating Murcko scaffold: {e}")
        metrics["unique_murcko_scaffolds"] = len(set([(scaffold) for scaffold in scaffolds]))

        # calc nr. of morgan fingerprints
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in valid_mols]
        metrics["unique_morgan_fps_1024_r2"] = len(set([fp.ToBitString() for fp in fps]))
        
        if len(valid_smiles) <= 10:
            print("Not enough valid molecules to calculate FCD or SA")
            metrics["FCD"] = None
            metrics["SA"] = None
        else:
            try:
                metrics["FCD"] = self.fcd(valid_smiles, self.val_li_of_smi)
            except Exception as e:
                print(f"Error calculating FCD: {e}")
                metrics["FCD"] = None
            try:
                sa_scores = [self.calculate_sa_score(m) for m in valid_mols if m is not None]
                sa_scores = np.array([s for s in sa_scores if s is not None])
                #print(np.nanmean(sa_scores))
                metrics["SA"] = np.nanmean(sa_scores)
            except Exception as e:
                print(f"Error calculating SA score: {e}")
                metrics["SA"] = None
            try:
                # circle-diversity, #Circles # not to be confused with IntDiv which means Internal Diversity and is the mean of the distance-matrix
                metrics["DS@0.7"] = self.calculate_circle_diversity(valid_smiles, distance_threshold=0.7, algorithm="maxmin")
            except Exception as e:
                print(f"Error calculating circle diversity: {e}")
                metrics["DS@0.7"] = None
        
        num_smiles = len(list_of_smiles)
        for k, v in list(metrics.items()):
            if isinstance(v, (int, float)) and k not in ["SA", "FCD"]:
                metrics[f"{k}%"] = (v / num_smiles) * 100

        return metrics


    def eval_designs_var(self, list_of_smiles, n_bootstraps=5):
        """
        Evaluate the designs  including variance estimates.
        """
        metrics = []
        metrics_var = {}

        m = self.eval_designs(list_of_smiles)
        
        for i in range(n_bootstraps):
            bootstrapped_smiles = np.random.choice(list_of_smiles, len(list_of_smiles), replace=True)

            m = self.eval_designs(bootstrapped_smiles)
            metrics.append(m)
        
        for k in metrics[0].keys():
            for i, m in enumerate(metrics):
                metrics_var[f"{k}_{i}"] = m[k]
            metrics_var[k] = m[k]

            if isinstance(metrics[0][k], (int, float)):
                valid_values = [m[k] for m in metrics if m[k] is not None]
                metrics_var[f"{k}_var"] = np.nanvar(valid_values)
                metrics_var[f"{k}_mean"] = np.nanmean(valid_values)
    
        return metrics_var

if __name__ == "__main__":
    test_smiles = ["CCO", "O=C=O", "invalid_smiles"]
    evaluator = SmilesGenEvaluator()
    metrics = evaluator.eval_designs(test_smiles)
    print(metrics)