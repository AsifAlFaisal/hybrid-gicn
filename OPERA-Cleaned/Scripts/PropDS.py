#%% Custom Data
## This script is inspired from https://github.com/deepfindr/gnn-project/blob/main/dataset.py

from torch_geometric.data import Data, Dataset
from rdkit import Chem
import numpy as np
import torch
from torch_geometric.transforms import RemoveIsolatedNodes
from dgllife.utils import CanonicalBondFeaturizer, CanonicalAtomFeaturizer
import pandas as pd
import os
from tqdm import tqdm

class PhysPropData(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        self.test = test
        self.filename = filename
        super(PhysPropData, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        if self.test:
            return [f'tst_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'tr_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        #self.rmv = RemoveIsolatedNodes()
        self.data = pd.read_csv(self.raw_paths[0])
        target_col = self.data.columns[-2]
        try:
            for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
                mol_obj = Chem.MolFromInchi(mol['InChI'])
                node_feat_mat = self._get_node_feature_matrix(mol_obj)
                edge_attr_mat = self._get_edge_attributes(mol_obj)
                adjacency_mat = self._get_edge_index(mol_obj)
                target= self._get_classes(mol[target_col])
                data = Data(x=node_feat_mat, edge_index=adjacency_mat, edge_attr=edge_attr_mat, y = target)
                #data = self.rmv(data)
                if self.test:
                    torch.save(data, os.path.join(self.processed_dir, f'tst_{index}.pt'))
                else:
                    torch.save(data, os.path.join(self.processed_dir, f'tr_{index}.pt'))
        except AttributeError:
            print(mol['SMILES'])
    
    
    def _get_node_feature_matrix(self, mol):
        # shape [num_nodes, num_node_features]
        node_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')
        node_feat = node_featurizer(mol)['feat']
        return node_feat

    def _get_edge_attributes(self, mol):
        edge_featurizer = CanonicalBondFeaturizer(bond_data_field='feat')
        edge_feat = edge_featurizer(mol)['feat']
        return edge_feat

    def _get_edge_index(self, mol):
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index += [[i,j],[j,i]]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = edge_index.t().view(2,-1)
        return edge_index

    def _get_classes(self, classes):
        classes = np.asarray([classes])
        return torch.tensor(classes, dtype=torch.float32)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'tst_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, f'tr_{idx}.pt'))
        return data