from copy import deepcopy
import argparse
from time import time
import sys

import scanpy as sc
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from model import No_Perturb, PertNet
from data import PertDataloader, GeneSimNetwork, GeneCoexpressNetwork
from inference import evaluate, compute_metrics, deeper_analysis, GI_subgroup, non_dropout_analysis, non_zero_analysis
from utils import loss_fct, uncertainty_loss_fct, parse_any_pert, get_coexpression_network_from_train, get_similarity_network

torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")



def trainer(args):
 
    wandb_status = args['wandb']         
    device = args['device']
    project_name = args['project_name']
    entity_name = args['entity_name']
    exp_name = args['exp_name']
    
    model_name = args['model_name']
    args = np.load('./saved_args/'+model_name+'.npy', allow_pickle = True).item()
    args['device'] = device

    ## set up wandb
    if wandb_status:
        import wandb 
        wandb.init(project=project_name, entity=entity_name, name=exp_name)
        wandb.config.update(args)

    if args['dataset'] == 'Norman2019':
        data_path = '/dfs/project/perturb-gnn/datasets/Norman2019/Norman2019_hvg+perts_more_de.h5ad'
    elif args['dataset'] == 'Adamson2016':
        data_path = '/dfs/project/perturb-gnn/datasets/Adamson2016_hvg+perts_more_de_in_genes.h5ad'
    elif args['dataset'] == 'Dixit2016':
        data_path = '/dfs/project/perturb-gnn/datasets/Dixit2016_hvg+perts_more_de.h5ad'
    elif args['dataset'] == 'Norman2019_Adamson2016':
        data_path = '/dfs/project/perturb-gnn/datasets/trans_norman_adamson/norman2019.h5ad'
    
    s = time()
    adata = sc.read_h5ad(data_path)
    if 'gene_symbols' not in adata.var.columns.values:
        adata.var['gene_symbols'] = adata.var['gene_name']
    gene_list = [f for f in adata.var.gene_symbols.values]
 
    # Pertrubation dataloader
    pertdl = PertDataloader(adata, args)
         
    model = torch.load('./saved_models/' + model_name)
    model.args = args
    
    if 'G_sim' in vars(model):
        if isinstance(model.G_sim, dict):
            for i,j in model.G_sim.items():
                model.G_sim[i] = j.to(model.args['device'])

            for i,j in model.G_sim_weight.items():
                model.G_sim_weight[i] = j.to(model.args['device'])
        else:
            model.G_sim = model.G_sim.to(model.args['device'])
            model.G_sim_weight = model.G_sim_weight.to(model.args['device'])

    best_model = model
        
    print('Start testing....')
    test_res = evaluate(pertdl.loaders['test_loader'],best_model, args)
    
    test_metrics, test_pert_res = compute_metrics(test_res)
    
    if wandb_status:
        metrics = ['mse', 'mae', 'spearman', 'pearson', 'r2']
        for m in metrics:
            wandb.log({'test_' + m: test_metrics[m],
                       'test_de_'+m: test_metrics[m + '_de']                     
                      })
    
    out = deeper_analysis(adata, test_res)
    out_non_dropout = non_dropout_analysis(adata, test_res)
    out_non_zero = non_zero_analysis(adata, test_res)
    GI_out = GI_subgroup(out)
    GI_out_non_dropout = GI_subgroup(out_non_dropout)
    GI_out_non_zero = GI_subgroup(out_non_zero)
    
    metrics = ['frac_in_range', 'frac_in_range_45_55', 'frac_in_range_40_60', 'frac_in_range_25_75', 'mean_sigma', 'std_sigma', 'frac_sigma_below_1', 'frac_sigma_below_2', 'pearson_delta',
               'pearson_delta_de', 'fold_change_gap_all', 'pearson_delta_top200_hvg', 'fold_change_gap_upreg_3', 
               'fold_change_gap_downreg_0.33', 'fold_change_gap_downreg_0.1', 'fold_change_gap_upreg_10', 
               'pearson_top200_hvg', 'pearson_top200_de', 'pearson_top20_de', 'pearson_delta_top200_de', 
               'pearson_top100_de', 'pearson_delta_top100_de', 'pearson_delta_top50_de', 'pearson_top50_de', 'pearson_delta_top20_de',
               'mse_top200_hvg', 'mse_top100_de', 'mse_top200_de', 'mse_top50_de', 'mse_top20_de', 'frac_correct_direction_all', 'frac_correct_direction_20', 'frac_correct_direction_50', 'frac_correct_direction_100', 'frac_correct_direction_200', 'frac_correct_direction_20_nonzero']
    
    metrics_non_dropout = ['frac_correct_direction_top20_non_dropout', 'frac_opposite_direction_top20_non_dropout', 'frac_0/1_direction_top20_non_dropout', 'frac_correct_direction_non_zero', 'frac_correct_direction_non_dropout', 'frac_in_range_non_dropout', 'frac_in_range_45_55_non_dropout', 'frac_in_range_40_60_non_dropout', 'frac_in_range_25_75_non_dropout', 'mean_sigma_non_dropout', 'std_sigma_non_dropout', 'frac_sigma_below_1_non_dropout', 'frac_sigma_below_2_non_dropout', 'pearson_delta_top20_de_non_dropout', 'pearson_top20_de_non_dropout', 'mse_top20_de_non_dropout', 'frac_opposite_direction_non_dropout', 'frac_0/1_direction_non_dropout', 'frac_opposite_direction_non_zero', 'frac_0/1_direction_non_zero']
    
    
    metrics_non_zero = ['frac_correct_direction_top20_non_zero', 'frac_opposite_direction_top20_non_zero', 'frac_0/1_direction_top20_non_zero', 'frac_in_range_non_zero', 'frac_in_range_45_55_non_zero', 'frac_in_range_40_60_non_zero', 'frac_in_range_25_75_non_zero', 'mean_sigma_non_zero', 'std_sigma_non_zero', 'frac_sigma_below_1_non_zero', 'frac_sigma_below_2_non_zero', 'pearson_delta_top20_de_non_zero', 'pearson_top20_de_non_zero', 'mse_top20_de_non_zero']
    
    if args['wandb']:
        for m in metrics:
            wandb.log({'test_' + m: np.mean([j[m] for i,j in out.items() if m in j])})
        
        for m in metrics_non_dropout:
            wandb.log({'test_' + m: np.mean([j[m] for i,j in out_non_dropout.items() if m in j])})        
        
        for m in metrics_non_zero:
            wandb.log({'test_' + m: np.mean([j[m] for i,j in out_non_zero.items() if m in j])})        

            
    if args['split'] == 'simulation':
        subgroup = pertdl.subgroup
        subgroup_analysis = {}
        for name in subgroup['test_subgroup'].keys():
            subgroup_analysis[name] = {}
            for m in list(list(test_pert_res.values())[0].keys()):
                subgroup_analysis[name][m] = []

        for name, pert_list in subgroup['test_subgroup'].items():
            for pert in pert_list:
                for m, res in test_pert_res[pert].items():
                    subgroup_analysis[name][m].append(res)

        for name, result in subgroup_analysis.items():
            for m in result.keys():
                subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                if args['wandb']:
                    wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                print('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))
        
        ## deeper analysis
        subgroup_analysis = {}
        for name in subgroup['test_subgroup'].keys():
            subgroup_analysis[name] = {}
            for m in metrics:
                subgroup_analysis[name][m] = []
                
            for m in metrics_non_dropout:
                subgroup_analysis[name][m] = []
                
            for m in metrics_non_zero:
                subgroup_analysis[name][m] = []

        for name, pert_list in subgroup['test_subgroup'].items():
            for pert in pert_list:
                for m, res in out[pert].items():
                    subgroup_analysis[name][m].append(res)
                
                for m, res in out_non_dropout[pert].items():
                    subgroup_analysis[name][m].append(res)
                
                for m, res in out_non_zero[pert].items():
                    subgroup_analysis[name][m].append(res)

                    
        for name, result in subgroup_analysis.items():
            for m in result.keys():
                subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                if args['wandb']:
                    wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                print('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))
    
    for i,j in GI_out.items():
        for m in  ['mean_sigma', 'frac_in_range_45_55', 'frac_in_range_40_60', 'frac_in_range_25_75', 
               'fold_change_gap_all', 'pearson_delta_top200_de', 'pearson_delta_top100_de',  'pearson_delta_top50_de',
               'mse_top200_de', 'mse_top100_de', 'mse_top50_de', 'mse_top20_de', 'pearson_delta_top20_de']:
            if args['wandb']:
                wandb.log({'test_' + i + '_' + m: j[m]})
                
                
    for i,j in GI_out_non_dropout.items():
        for m in  ['frac_correct_direction_top20_non_dropout', 'mse_top20_de_non_dropout', 'pearson_delta_top20_de_non_dropout', 'frac_in_range_25_75_non_dropout', 'frac_sigma_below_1_non_dropout']:
            if args['wandb']:
                wandb.log({'test_' + i + '_' + m: j[m]})
                
                
    for i,j in GI_out_non_zero.items():
        for m in  ['frac_correct_direction_top20_non_zero', 'mse_top20_de_non_zero', 'pearson_delta_top20_de_non_zero', 'frac_in_range_25_75_non_zero', 'frac_sigma_below_1_non_zero']:
            if args['wandb']:
                wandb.log({'test_' + i + '_' + m: j[m]})
    

    print('Done!')


def parse_arguments():
    """
    Argument parser
    """

    # dataset arguments
    parser = argparse.ArgumentParser(description='Perturbation response')
    
    # wandb related
    parser.add_argument('--wandb', default=False, action='store_true',
                    help='Use wandb or not')
    parser.add_argument('--project_name', type=str, default='pert_gnn',
                        help='project name')
    parser.add_argument('--entity_name', type=str, default='kexinhuang',
                        help='entity name')
    parser.add_argument('--exp_name', type=str, default='N/A',
                        help='entity name')
    
    # misc
    parser.add_argument('--model_name', type=str, default='pert_gnn')
    parser.add_argument('--device', type=str, default='cuda')
    
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    
    #python evaluate.py --project_name pert_gnn_simulation_norman2019 \
    #                --exp_name no_perturb \
    #                --model_name no_perturb \
    #                --device cuda:7 \
    #                --wandb
        
    trainer(parse_arguments())
