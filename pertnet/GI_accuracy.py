import glob
import numpy as np
import torch
import scanpy as sc
import pandas as pd 
import copy
import sys
import os
from data import PertDataloader
from inference import evaluate, compute_metrics
from inference import GIs
import matplotlib.patches as mpatches

# Linear model fitting functions
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from dcor import distance_correlation, partial_distance_correlation
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
%matplotlib inline

device = 'cuda:1'
home_dir = '/dfs/user/yhr/perturb_GNN/pertnet/'
sys.path.append(home_dir)


## Read in output from leave one out models
# Plot generation functions
def get_t_p_seen1(metric):
    res_seen1 = {}
    res_p_seen1 = {}
    res_t_seen1 = {}

    # Set up output dictionaries
    for GI in GI_names:
        res_p_seen1[GI] = []
        res_t_seen1[GI] = []

    for GI_sel in GI_names:

        # For a given GI what are all the relevant perturbations
        all_perts_gi = GIs[GI_sel.upper()]

        # What are all the relevant single gene perturbations
        all_perts_gi = [v.split('+') for v in all_perts_gi] 
        seen1_perts_gi = np.unique([item for sublist in all_perts_gi for item in sublist])

        # Iterate over all models trained with these single genes held out
        for GI in seen1_perts_gi:
            for d in dict_names:
                if GI in d:
                    res_seen1[GI] = np.load(d, allow_pickle=True).item()

                    # Get all keys for single pert model predictions that are relevant
                    keys_ = [k for k in res_seen1[GI].keys() if k in GIs[GI_sel.upper()]]

                    p_vals = [res_seen1[GI][k]['pred'][metric] for k in keys_]
                    t_vals = [res_seen1[GI][k]['truth'][metric] for k in keys_]

                    res_p_seen1[GI_sel].extend(p_vals)
                    res_t_seen1[GI_sel].extend(t_vals)
                
    return res_p_seen1, res_t_seen1

def get_t_p_seen2(metric):

    # Seen 2
    res_p = {}
    res_t = {}

    for GI in GI_names:
        res_p[GI] = []
        res_t[GI] = []

    for GI in GI_names:
        for d in dict_names:
            if GI in d:
                loaded = list(np.load(d, allow_pickle=True).item().values())[0]
                res_p[GI].append(loaded['pred'][metric])
                res_t[GI].append(loaded['truth'][metric])

    return res_p, res_t


## Compute accuracy

def synergy_similar_pheno(dict_):
    return np.sum(np.array(dict_['mag']['synergy_similar_pheno'])>1)/len(dict_['mag']['synergy_similar_pheno'])

def synergy_dissimilar_pheno(dict_):
    return np.sum(np.array(dict_['mag']['synergy_dissimilar_pheno'])>1)/len(dict_['mag']['synergy_dissimilar_pheno'])

def potentiation(dict_):
    cond1 = np.sum(np.array(dict_['mag']['potentiation'])>1)/len(dict_['mag']['potentiation'])
    return cond1
    #cond2 = 

# TODO check this condition
def additive(dict_, thresh=0.3):
    cond = np.abs(np.array(dict_['mag']['additive'])-1)<=thresh
    return np.sum(cond)/len(dict_['mag']['additive'])

def suppressor(dict_):
    return np.sum(np.array(dict_['mag']['suppressor'])<1)/len(dict_['mag']['suppressor'])

def neomorphic(dict_):
    return np.sum(np.array(dict_['corr_fit']['neomorphic'])<0.85)/len(dict_['corr_fit']['neomorphic'])

def redundant(dict_):
    return np.sum(np.array(dict_['dcor']['redundant'])>0.8)/len(dict_['dcor']['redundant'])

def epistasis(dict_):
    return np.sum(np.array(dict_['dominance']['epistasis'])>0.25)/len(dict_['dominance']['epistasis'])


res_p_seen1_dict = {}
res_t_seen1_dict = {}

res_p_seen2_dict = {}
res_t_seen2_dict = {}


# Set up data dictionaries
for metric in ['mag', 'corr_fit', 'dcor', 'dominance', 'dcor_singles']:
    res_p_seen1_dict[metric], res_t_seen1_dict[metric] = get_t_p_seen1(metric)
    res_p_seen2_dict[metric], res_t_seen2_dict[metric] = get_t_p_seen2(metric)
    

accuracy_seen2 = {}
accuracy_seen1 = {}

accuracy_seen2['synergy_similar_pheno'] = synergy_similar_pheno(res_p_seen2_dict)
accuracy_seen2['synergy_dissimilar_pheno'] = synergy_dissimilar_pheno(res_p_seen2_dict)
accuracy_seen2['potentiation'] = potentiation(res_p_seen2_dict)
accuracy_seen2['additive'] = additive(res_p_seen2_dict)
accuracy_seen2['suppressor'] = suppressor(res_p_seen2_dict)
accuracy_seen2['neomorphic'] = neomorphic(res_p_seen2_dict)
accuracy_seen2['redundant'] = redundant(res_p_seen2_dict)
accuracy_seen2['epistasis'] = epistasis(res_p_seen2_dict)

accuracy_seen1['synergy_similar_pheno'] = synergy_similar_pheno(res_p_seen1_dict)
accuracy_seen1['synergy_dissimilar_pheno'] = synergy_dissimilar_pheno(res_p_seen1_dict)
accuracy_seen1['potentiation'] = potentiation(res_p_seen1_dict)
accuracy_seen1['additive'] = additive(res_p_seen1_dict)
accuracy_seen1['suppressor'] = suppressor(res_p_seen1_dict)
accuracy_seen1['neomorphic'] = neomorphic(res_p_seen1_dict)
accuracy_seen1['redundant'] = redundant(res_p_seen1_dict)
accuracy_seen1['epistasis'] = epistasis(res_p_seen1_dict)
