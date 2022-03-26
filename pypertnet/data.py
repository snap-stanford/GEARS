from torch_geometric.data import Data
import torch
import numpy as np
import pickle
from torch_geometric.data import DataLoader
import os
from random import shuffle
import pandas as pd
import scanpy as sc
import networkx as nx
from tqdm import tqdm

import sys
sys.path.append('/dfs/user/yhr/cell_reprogram/model/')

from utils import parse_single_pert, parse_combo_pert, parse_any_pert


class GeneSimNetwork():
    def __init__(self, edge_list, gene_list, node_map):
        self.edge_list = edge_list
        self.G = nx.from_pandas_edgelist(self.edge_list, source='source',
                        target='target', edge_attr=['importance'],
                        create_using=nx.DiGraph())    
        self.gene_list = gene_list
        for n in self.gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)
        
        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in
                      self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        #self.edge_weight = torch.Tensor(self.edge_list['importance'].values)
        
        edge_attr = nx.get_edge_attributes(self.G, 'importance') 
        importance = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(importance)

class GeneCoexpressNetwork():
    def __init__(self, fname, gene_list, node_map):
        self.edge_list = pd.read_csv(fname)
        self.G = nx.from_pandas_edgelist(self.edge_list, source='source',
                        target='target', edge_attr=['importance'],
                        create_using=nx.DiGraph())    
        self.gene_list = gene_list
        for n in self.gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)
        
        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in
                      self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        self.edge_weight = torch.Tensor(self.edge_list['importance'].values)
                
        
class PertDataloader():
    def __init__(self, adata, args, binary_pert=True):
        self.args = args
        self.adata = adata
        self.ctrl_adata = adata[adata.obs[args['perturbation_key']] == 'ctrl']
        self.node_map = {x: it for it, x in enumerate(adata.var.gene_symbols)}
        self.binary_pert=binary_pert
        self.gene_names = self.adata.var.gene_symbols
        self.loaders = self.create_dataloaders()

    def create_dataloaders(self):
        """
        Main routine for setting up dataloaders
        """
        print("Creating pyg object for each cell in the data...")
        
        # create dataset processed pyg objects
        dataset_fname = './data_pyg/' + self.args['dataset'] + '.pkl'
                
        if os.path.isfile(dataset_fname):
            print("Local copy of pyg dataset is detected. Loading...")
            dataset_processed = pickle.load(open(dataset_fname, "rb"))
        else:
            print("Processing dataset...")
            
            dataset_processed = self.create_dataset_file()
            print("Saving new dataset pyg object at " + dataset_fname) 
            pickle.dump(dataset_processed, open(dataset_fname, "wb"))
            
        print("Loading splits...")
        
        split_path = './splits/' + self.args['dataset'] + '_' + self.args['split'] + '_' + str(self.args['seed']) + '_' + str(self.args['test_set_fraction']) + '.pkl'
        
        if self.args['test_perts'] != 'N/A':
            split_path = split_path[:-4] + '_' + self.args['test_perts'] + '.pkl'

        if self.args['test_pert_genes'] != 'N/A':
            split_path = split_path[:-4] + '_' + self.args['test_pert_genes']\
                         + '.pkl'
        
        if (self.args['split'] == 'custom') and (self.args['split_path'] != 'N/A'):
            split_path = self.args['split_path']
        elif self.args['split_path'] != 'N/A':
            if self.args['split'] != 'custom':
                raise ValueError('To use custom split path, you have to turn the split mode into custom!')
            else:
                pass
        print(split_path)
        if os.path.exists(split_path):
            print("Local copy of split is detected. Loading...")
            set2conditions = pickle.load(open(split_path, "rb"))
            if self.args['split'] == 'simulation':
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                subgroup = pickle.load(open(subgroup_path, "rb"))
                self.subgroup = subgroup
        else:
            print("Creating new splits....")
            if self.args['split'] == 'simulation':
                if self.args['test_perts'] != 'N/A':
                    test_perts = self.args['test_perts'].split('_')
                else:
                    test_perts = None
                    
                DS = DataSplitter(self.adata, split_type='simulation')
                
                adata, subgroup = DS.split_data(train_gene_set_size = self.args['train_gene_set_size'], 
                                                combo_seen2_train_frac = self.args['combo_seen2_train_frac'],
                                                seed=self.args['seed'],
                                                test_perts = test_perts,
                                                only_test_set_perts = self.args['only_test_set_perts']
                                               )
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                pickle.dump(subgroup, open(subgroup_path, "wb"))
                self.subgroup = subgroup
            elif self.args['split'] == 'simulation_single':
                if self.args['test_perts'] != 'N/A':
                    test_perts = self.args['test_perts'].split('_')
                else:
                    test_perts = None
                    
                DS = DataSplitter(self.adata, split_type='simulation_single')
                
                adata, subgroup = DS.split_data(train_gene_set_size = self.args['train_gene_set_size'], 
                                                seed=self.args['seed'],
                                                test_perts = test_perts,
                                                only_test_set_perts = self.args['only_test_set_perts']
                                               )
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                pickle.dump(subgroup, open(subgroup_path, "wb"))
                self.subgroup = subgroup    
                
            elif self.args['split'][:5] == 'combo':
                split_type = 'combo'
                seen = int(self.args['split'][-1])

                if self.args['test_perts'] != 'N/A':
                    test_perts = self.args['test_perts'].split('_')
                else:
                    test_perts = None

                if self.args['test_pert_genes'] != 'N/A':
                    test_pert_genes = self.args['test_pert_genes'].split('_')
                else:
                    test_pert_genes = None

                DS = DataSplitter(self.adata, split_type=split_type,
                                  seen=int(seen))
                adata = DS.split_data(test_size=self.args['test_set_fraction'],
                                      split_name='split',
                                      test_perts=test_perts,
                                      test_pert_genes=test_pert_genes,
                                      seed=self.args['seed'])
            
            elif self.args['split'] == 'no_split':          
                adata = self.adata
                adata.obs['split'] = 'test'
            else:
                DS = DataSplitter(self.adata, split_type=self.args['split'])
            
                adata = DS.split_data(test_size=self.args['test_set_fraction'], split_name='split',
                                       seed=self.args['seed'])
            
            set2conditions = dict(adata.obs.groupby('split').agg({'condition': lambda x: x}).condition)
            set2conditions = {i: j.unique().tolist() for i,j in set2conditions.items()} 
            pickle.dump(set2conditions, open(split_path, "wb"))
            print("Saving new splits at " + split_path) 
            
        self.set2conditions = set2conditions
        
        if self.args['split'] == 'simulation':
            print('Simulation split test composition:')
            for i,j in subgroup['test_subgroup'].items():
                print(i + ':' + str(len(j)))
        
        # Create cell graphs
        cell_graphs = {}
        
        if self.args['split'] == 'no_split':
            i = 'test'
            cell_graphs[i] = []
            for p in set2conditions[i]:
                if p != 'ctrl':
                    cell_graphs[i].extend(dataset_processed[p])
                
            print("Creating dataloaders....")
            # Set up dataloaders
            test_loader = DataLoader(cell_graphs['test'],
                                batch_size=self.args['batch_size'], shuffle=False)

            print("Dataloaders created...")
            return {'test_loader': test_loader}
        else:
            for i in ['train', 'val', 'test']:
                cell_graphs[i] = []
                for p in set2conditions[i]:
                    cell_graphs[i].extend(dataset_processed[p])

            print("Creating dataloaders....")
            # Set up dataloaders
            train_loader = DataLoader(cell_graphs['train'],
                                batch_size=self.args['batch_size'], shuffle=True, drop_last = True)
            val_loader = DataLoader(cell_graphs['val'],
                                batch_size=self.args['batch_size'], shuffle=True)
            test_loader = DataLoader(cell_graphs['test'],
                                batch_size=self.args['batch_size'], shuffle=False)

            print("Dataloaders created...")
            return {'train_loader': train_loader,
                    'val_loader': val_loader,
                    'test_loader': test_loader}
    
    
    def create_dataset_file(self):
        """
        Creates a dataloader for adata dataset
        """
        dl = {}

        for p in tqdm(self.adata.obs[self.args['perturbation_key']].unique()):
            cell_graph_dataset = self.create_cell_graph_dataset(
                self.adata, p, num_samples=self.args['num_ctrl_samples'])
            dl[p] = cell_graph_dataset
        return dl
    
    def get_pert_idx(self, pert_category, adata_):
        """
        Get indices (and signs) of perturbations
        """

        pert_idx = [np.where(p == self.gene_names)[0][0]
                    for p in pert_category.split('+')
                    if p != 'ctrl']

        # In case of binary perturbations, attach a sign to index value
        for i, p in enumerate(pert_idx):
            if self.binary_pert:
                sign = np.sign(adata_.X[0, p] - self.ctrl_adata.X[0, p])
                if sign == 0:
                    sign = 1
                pert_idx[i] = sign * pert_idx[i]

        return pert_idx

    # Set up feature matrix and output
    def create_cell_graph(self, X, y, de_idx, pert, pert_idx=None):
        """
        Uses the gene expression information for a cell and an underlying
        graph (e.g coexpression) to create a graph for each cell
        """

        # If perturbations will be represented as node features
        pert_feats = np.zeros(len(X[0]))
        if pert_idx is not None:
            for p in pert_idx:
                pert_feats[int(np.abs(p))] = np.sign(p)
        pert_feats = np.expand_dims(pert_feats, 0)
        feature_mat = torch.Tensor(np.concatenate([X, pert_feats])).T

        save_graph = None
        save_attr = None

        return Data(x=feature_mat, edge_index=save_graph, edge_attr=save_attr,
                    y=torch.Tensor(y), de_idx=de_idx, pert=pert)

    def create_cell_graph_dataset(self, split_adata, pert_category,
                                  num_samples=1):
        """
        Combine cell graphs to create a dataset of cell graphs
        """

        print(pert_category)
        num_de_genes = 20
        adata_ = split_adata[split_adata.obs['condition'] == pert_category]
        de_genes = adata_.uns['rank_genes_groups_cov']
        Xs = []
        ys = []

        # When considering a non-control perturbation
        if pert_category != 'ctrl':
            # Get the indices (and signs) of applied perturbation
            pert_idx = self.get_pert_idx(pert_category, adata_)

            # Store list of genes that are most differentially expressed for testing
            pert_de_category = adata_.obs['cov_drug_dose_name'][0]
            de_idx = np.where(adata_.var_names.isin(
                np.array(de_genes[pert_de_category])))[0]

            for cell_z in adata_.X:
                # Use samples from control for input to the GNN_AE model
                ctrl_samples = self.ctrl_adata[np.random.randint(0,
                                        len(self.ctrl_adata), num_samples), :]
                for c in ctrl_samples.X:
                    Xs.append(c)
                    ys.append(cell_z)

        # When considering a control perturbation
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            for cell_z in adata_.X:
                Xs.append(cell_z)
                ys.append(cell_z)

        # Create cell graphs
        cell_graphs = []
        for X, y in zip(Xs, ys):
            cell_graphs.append(self.create_cell_graph(X.toarray(),
                                y.toarray(), de_idx, pert_category, pert_idx))

        return cell_graphs


    def get_train_test_split(self):
        """
        Get train, validation and test set split from input data
        """

        adata = sc.read(self.args['fname'])
        train_adata = adata[adata.obs[self.args['split_key']] == 'train']
        val_adata = adata[adata.obs[self.args['split_key']] == 'val']
        test_adata = adata[adata.obs[self.args['split_key']] == 'test']

        train_split = list(train_adata.obs['condition'].unique())
        val_split = list(val_adata.obs['condition'].unique())
        test_split = list(test_adata.obs['condition'].unique())

        return train_split, val_split, test_split

    
class DataSplitter():
    """
    Class for handling data splitting. This class is able to generate new
    data splits and assign them as a new attribute to the data file.
    """
    def __init__(self, adata, split_type='single', seen=0):
        self.adata = adata
        self.split_type = split_type
        self.seen = seen

    def split_data(self, test_size=0.1, test_pert_genes=None,
                   test_perts=None, split_name='split', seed=None, val_size = 0.1,
                   train_gene_set_size = 0.75, combo_seen2_train_frac = 0.75, only_test_set_perts = False):
        """
        Split dataset and adds split as a column to the dataframe
        Note: split categories are train, val, test
        """
        np.random.seed(seed=seed)
        unique_perts = [p for p in self.adata.obs['condition'].unique() if
                        p != 'ctrl']
        
        if self.split_type == 'simulation':
            train, test, test_subgroup = self.get_simulation_split(unique_perts,
                                                                  train_gene_set_size,
                                                                  combo_seen2_train_frac, 
                                                                  seed, test_perts, only_test_set_perts)
            train, val, val_subgroup = self.get_simulation_split(train,
                                                                  0.9,
                                                                  0.9,
                                                                  seed)
            ## adding back ctrl to train...
            train.append('ctrl')
        elif self.split_type == 'simulation_single':
            train, test, test_subgroup = self.get_simulation_split_single(unique_perts,
                                                                  train_gene_set_size,
                                                                  seed, test_perts, only_test_set_perts)
            train, val, val_subgroup = self.get_simulation_split_single(train,
                                                                  0.9,
                                                                  seed)
        else:
            train, test = self.get_split_list(unique_perts,
                                          test_pert_genes=test_pert_genes,
                                          test_perts=test_perts,
                                          test_size=test_size)
            
            train, val = self.get_split_list(train, test_size=val_size)

        map_dict = {x: 'train' for x in train}
        map_dict.update({x: 'val' for x in val})
        map_dict.update({x: 'test' for x in test})
        map_dict.update({'ctrl': 'train'})

        self.adata.obs[split_name] = self.adata.obs['condition'].map(map_dict)

        # Add some control to the validation set
        #ctrl_idx = self.adata.obs_names[self.adata.obs['condition'] == 'ctrl']
        #val_ctrl = np.random.choice(ctrl_idx, int(len(ctrl_idx) * test_size))
        #self.adata.obs.at[val_ctrl, 'split'] = 'val'
        if self.split_type == 'simulation':
            return self.adata, {'test_subgroup': test_subgroup, 
                                'val_subgroup': val_subgroup
                               }
        else:
            return self.adata
    
    def get_simulation_split_single(self, pert_list, train_gene_set_size = 0.85, seed = 1, test_set_perts = None, only_test_set_perts = False):
        unique_pert_genes = self.get_genes_from_perts(pert_list)
        
        pert_train = []
        pert_test = []
        np.random.seed(seed=seed)
        
        if only_test_set_perts and (test_set_perts is not None):
            ood_genes = np.array(test_set_perts)
            train_gene_candidates = np.setdiff1d(unique_pert_genes, ood_genes)
        else:
            ## a pre-specified list of genes
            train_gene_candidates = np.random.choice(unique_pert_genes,
                                                    int(len(unique_pert_genes) * train_gene_set_size), replace = False)

            if test_set_perts is not None:
                num_overlap = len(np.intersect1d(train_gene_candidates, test_set_perts))
                train_gene_candidates = train_gene_candidates[~np.isin(train_gene_candidates, test_set_perts)]
                ood_genes_exclude_test_set = np.setdiff1d(unique_pert_genes, np.union1d(train_gene_candidates, test_set_perts))
                train_set_addition = np.random.choice(ood_genes_exclude_test_set, num_overlap, replace = False)
                train_gene_candidates = np.concatenate((train_gene_candidates, train_set_addition))
                
            ## ood genes
            ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)  
        
        pert_single_train = self.get_perts_from_genes(train_gene_candidates, pert_list,'single')
        unseen_single = self.get_perts_from_genes(ood_genes, pert_list, 'single')
        assert len(unseen_single) + len(pert_single_train) == len(pert_list)
        
        return pert_single_train, unseen_single, {'unseen_single': unseen_single}
    
    def get_simulation_split(self, pert_list, train_gene_set_size = 0.85, combo_seen2_train_frac = 0.85, seed = 1, test_set_perts = None, only_test_set_perts = False):
        
        unique_pert_genes = self.get_genes_from_perts(pert_list)
        
        pert_train = []
        pert_test = []
        np.random.seed(seed=seed)
        
        if only_test_set_perts and (test_set_perts is not None):
            ood_genes = np.array(test_set_perts)
            train_gene_candidates = np.setdiff1d(unique_pert_genes, ood_genes)
        else:
            ## a pre-specified list of genes
            train_gene_candidates = np.random.choice(unique_pert_genes,
                                                    int(len(unique_pert_genes) * train_gene_set_size), replace = False)

            if test_set_perts is not None:
                num_overlap = len(np.intersect1d(train_gene_candidates, test_set_perts))
                train_gene_candidates = train_gene_candidates[~np.isin(train_gene_candidates, test_set_perts)]
                ood_genes_exclude_test_set = np.setdiff1d(unique_pert_genes, np.union1d(train_gene_candidates, test_set_perts))
                train_set_addition = np.random.choice(ood_genes_exclude_test_set, num_overlap, replace = False)
                train_gene_candidates = np.concatenate((train_gene_candidates, train_set_addition))
                
            ## ood genes
            ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)                
        
        pert_single_train = self.get_perts_from_genes(train_gene_candidates, pert_list,'single')
        pert_combo = self.get_perts_from_genes(train_gene_candidates, pert_list,'combo')
        pert_train.extend(pert_single_train)
        
        ## the combo set with one of them in OOD
        combo_seen1 = [x for x in pert_combo if len([t for t in x.split('+') if
                                     t in train_gene_candidates]) == 1]
        pert_test.extend(combo_seen1)
        
        pert_combo = np.setdiff1d(pert_combo, combo_seen1)
        ## randomly sample the combo seen 2 as a test set, the rest in training set
        np.random.seed(seed=seed)
        pert_combo_train = np.random.choice(pert_combo, int(len(pert_combo) * combo_seen2_train_frac), replace = False)
       
        combo_seen2 = np.setdiff1d(pert_combo, pert_combo_train).tolist()
        pert_test.extend(combo_seen2)
        pert_train.extend(pert_combo_train)
        
        ## unseen single
        unseen_single = self.get_perts_from_genes(ood_genes, pert_list, 'single')
        combo_ood = self.get_perts_from_genes(ood_genes, pert_list, 'combo')
        pert_test.extend(unseen_single)
        
        ## here only keeps the seen 0, since seen 1 is tackled above
        combo_seen0 = [x for x in combo_ood if len([t for t in x.split('+') if
                                     t in train_gene_candidates]) == 0]
        pert_test.extend(combo_seen0)
        assert len(combo_seen1) + len(combo_seen0) + len(unseen_single) + len(pert_train) + len(combo_seen2) == len(pert_list)

        return pert_train, pert_test, {'combo_seen0': combo_seen0,
                                       'combo_seen1': combo_seen1,
                                       'combo_seen2': combo_seen2,
                                       'unseen_single': unseen_single}
        
    def get_split_list(self, pert_list, test_size=0.1,
                       test_pert_genes=None, test_perts=None,
                       hold_outs=True):
        """
        Splits a given perturbation list into train and test with no shared
        perturbations
        """

        single_perts = [p for p in pert_list if 'ctrl' in p and p != 'ctrl']
        combo_perts = [p for p in pert_list if 'ctrl' not in p]
        unique_pert_genes = self.get_genes_from_perts(pert_list)
        hold_out = []

        if test_pert_genes is None:
            test_pert_genes = np.random.choice(unique_pert_genes,
                                        int(len(single_perts) * test_size))

        # Only single unseen genes (in test set)
        # Train contains both single and combos
        if self.split_type == 'single' or self.split_type == 'single_only':
            test_perts = self.get_perts_from_genes(test_pert_genes, pert_list,
                                                   'single')
            if self.split_type == 'single_only':
                # Discard all combos
                hold_out = combo_perts
            else:
                # Discard only those combos which contain test genes
                hold_out = self.get_perts_from_genes(test_pert_genes, pert_list,
                                                     'combo')

        elif self.split_type == 'combo':
            if self.seen == 0:
                # NOTE: This can reduce the dataset size!
                # To prevent this set 'holdouts' to False, this will cause
                # the test set to have some perturbations with 1 gene seen
                single_perts = self.get_perts_from_genes(test_pert_genes,
                                                         pert_list, 'single')
                combo_perts = self.get_perts_from_genes(test_pert_genes,
                                                        pert_list, 'combo')

                if hold_outs:
                    # This just checks that none of the combos have 2 seen genes
                    hold_out = [t for t in combo_perts if
                                len([t for t in t.split('+') if
                                     t not in test_pert_genes]) > 0]
                combo_perts = [c for c in combo_perts if c not in hold_out]
                test_perts = single_perts + combo_perts

            elif self.seen == 1:
                # NOTE: This can reduce the dataset size!
                # To prevent this set 'holdouts' to False, this will cause
                # the test set to have some perturbations with 2 genes seen
                single_perts = self.get_perts_from_genes(test_pert_genes,
                                                         pert_list, 'single')
                combo_perts = self.get_perts_from_genes(test_pert_genes,
                                                        pert_list, 'combo')

                if hold_outs:
                    # This just checks that none of the combos have 2 seen genes
                    hold_out = [t for t in combo_perts if
                                len([t for t in t.split('+') if
                                     t not in test_pert_genes]) > 1]
                combo_perts = [c for c in combo_perts if c not in hold_out]
                test_perts = single_perts + combo_perts

            elif self.seen == 2:
                if test_perts is None:
                    test_perts = np.random.choice(combo_perts,
                                     int(len(combo_perts) * test_size))       
                else:
                    test_perts = np.array(test_perts)
        train_perts = [p for p in pert_list if (p not in test_perts)
                                        and (p not in hold_out)]
        return train_perts, test_perts

    def get_perts_from_genes(self, genes, pert_list, type_='both'):
        """
        Returns all single/combo/both perturbations that include a gene
        """

        single_perts = [p for p in pert_list if ('ctrl' in p) and (p != 'ctrl')]
        combo_perts = [p for p in pert_list if 'ctrl' not in p]
        
        perts = []
        
        if type_ == 'single':
            pert_candidate_list = single_perts
        elif type_ == 'combo':
            pert_candidate_list = combo_perts
        elif type_ == 'both':
            pert_candidate_list = pert_list
            
        for p in pert_candidate_list:
            for g in genes:
                if g in parse_any_pert(p):
                    perts.append(p)
                    break
        '''
        perts = []
        for gene in genes:
            if type_ == 'single':
                perts.extend([p for p in single_perts if gene in parse_any_pert(p)])

            if type_ == 'combo':
                perts.extend([p for p in combo_perts if gene in parse_any_pert(p)])

            if type_ == 'both':
                perts.extend([p for p in pert_list if gene in parse_any_pert(p)])
        '''
        return perts

    def get_genes_from_perts(self, perts):
        """
        Returns list of genes involved in a given perturbation list
        """

        if type(perts) is str:
            perts = [perts]
        gene_list = [p.split('+') for p in np.unique(perts)]
        gene_list = [item for sublist in gene_list for item in sublist]
        gene_list = [g for g in gene_list if g != 'ctrl']
        return np.unique(gene_list)