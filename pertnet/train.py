from copy import deepcopy
import argparse
from time import time
import sys, os

import scanpy as sc
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from model import No_Perturb, PertNet, GNN, DNN
from data import PertDataloader, GeneSimNetwork, GeneCoexpressNetwork
from inference import evaluate, compute_metrics, deeper_analysis, \
    GI_subgroup, non_dropout_analysis, non_zero_analysis, compute_synergy_loss
from utils import loss_fct, uncertainty_loss_fct, parse_any_pert, \
                  get_coexpression_network_from_train, get_similarity_network, \
                  get_high_umi_idx, get_mean_ctrl, combine_res

torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")


def train(model, train_loader, val_loader, args, ctrl_expression, dict_filter, device="cpu"):
    best_model = deepcopy(model)
    if args['wandb']:
        import wandb
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = StepLR(optimizer, step_size=args['lr_decay_step_size'], gamma=args['lr_decay_factor'])

    min_val = np.inf
    adata = sc.read_h5ad(args['data_path'])

    print('Start Training...')

    for epoch in range(args["max_epochs"]):
        total_loss = 0
        model.train()
        num_graphs = 0

        for step, batch in enumerate(train_loader):

            batch.to(device)            
            model.to(device)
            optimizer.zero_grad()

            y = batch.y
            if args['func_readout']:
                y_func = batch.func_readout
            else:
                y_func = None

            if args['uncertainty']:
                pred, logvar, pred_func = model(batch)
                loss = uncertainty_loss_fct(pred, logvar, y, batch.pert,
                                  pred_func=pred_func,
                                  y_func=batch.func_readout,
                                  loss_mode = args['loss_mode'], 
                                  gamma = args['focal_gamma'],
                                  reg = args['uncertainty_reg'],
                                  reg_core = args['uncertainty_reg_core'],
                                  loss_direction = args['loss_direction'], 
                                  ctrl = ctrl_expression, 
                                  filter_status = args['filter_status'],
                                  dict_filter = dict_filter,
                                  direction_lambda = args['direction_lambda'])
            else:
                pred, pred_func = model(batch)
                # Compute loss
                loss = loss_fct(pred, y, batch.pert, args['pert_loss_wt'],
                                pred_func=pred_func,
                                y_func=y_func,
                                loss_mode = args['loss_mode'],
                                gamma = args['focal_gamma'],
                                loss_type = args['loss_type'],
                                loss_direction = args['loss_direction'],
                                ctrl = ctrl_expression, 
                                filter_status = args['filter_status'],
                                dict_filter = dict_filter,
                                direction_lambda = args['direction_lambda'])
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
            
            if args['wandb']:
                wandb.log({'training_loss': loss.item()})
                
            if step % args["print_progress_steps"] == 0:
                log = "Epoch {} Step {} Train Loss: {:.4f}" 
                print(log.format(epoch + 1, step + 1, loss.item()))

        scheduler.step()
        # Evaluate model performance on train and val set
        total_loss /= num_graphs
        train_res = evaluate(train_loader, model, args)
        val_res = evaluate(val_loader, model, args)
        train_metrics, _ = compute_metrics(train_res)
        val_metrics, _ = compute_metrics(val_res)
        
        # Print epoch performance
        log = "Epoch {}: Train: {:.4f} " \
              "Validation: {:.4f}. " \
              "Loss: {:.4f}"
        print(log.format(epoch + 1, train_metrics['mse'], 
                         val_metrics['mse'], 
                         total_loss))
        
        if args['wandb']:
            metrics = ['mse', 'mae', 'spearman', 'pearson']
            for m in metrics:
                wandb.log({'train_' + m: train_metrics[m],
                           'val_'+m: val_metrics[m],
                           'train_de_' + m: train_metrics[m + '_de'],
                           'val_de_'+m: val_metrics[m + '_de']})
            
            if args['func_readout']:
                m = 'func_mse'
                wandb.log({'train_' + m: train_metrics[m],
                           'val_' + m: val_metrics[m]})
        
        
        # Print epoch performance for DE genes
        log = "DE_Train: {:.4f} " \
              "DE_Validation: {:.4f}. "
        print(log.format(train_metrics['mse_de'],
                         val_metrics['mse_de']))
    
            
        # Select best model
        if args['func_readout'] != None:
            if val_metrics['func_mse'] < min_val:
                min_val = val_metrics['func_mse']
                best_model = deepcopy(model)

        else:
            if val_metrics['mse_de'] < min_val:
                min_val = val_metrics['mse_de']
                best_model = deepcopy(model)

    return best_model


def trainer(args):
    print('---- Printing Arguments ----')
    for i, j in args.items():
        print(i + ': ' + str(j))
    print('----------------------------')
    
    ## set up wandb
    if args['wandb']:
        import wandb 
        if not args['wandb_sweep']:
            wandb.init(project=args['project_name'] + '_' + args['split'] + '_' + args['dataset'], entity=args['entity_name'], name=args['exp_name'])
            wandb.config.update(args)

    if args['dataset'] == 'Norman2019':
        data_path = '/dfs/project/perturb-gnn/datasets/Norman2019/Norman2019_hvg+perts_more_de.h5ad'
    elif args['dataset'] == 'Norman2019_umi':
        data_path = '/lfs/local/0/kexinh/dataset/perturb_gnn/Norman2019_hi_umi+hvg.h5ad'
        if not os.path.exists(data_path):
            data_path = '/dfs/project/perturb-gnn/datasets/Norman2019/Norman2019_hi_umi+hvg.h5ad'
    elif args['dataset'] == 'Norman2019_umi_all_poss':
        data_path = '/dfs/project/perturb-gnn/datasets/Norman2019/Norman2019_hi_umi+hvg_all_poss.h5ad'
    elif args['dataset'] == 'Norman2019_GI':
        data_path = '/dfs/project/perturb-gnn/datasets/Norman2019/Norman2019_all_possible_train1.h5ad'
    elif args['dataset'] == 'Adamson2016':
        data_path = '/dfs/project/perturb-gnn/datasets/Adamson2016_hvg+perts_more_de_in_genes.h5ad'
    elif args['dataset'] == 'Dixit2016':
        data_path = '/dfs/project/perturb-gnn/datasets/Dixit2016_hvg+perts_more_de_in_perts.h5ad'
    elif args['dataset'] == 'Norman2019_Adamson2016':
        data_path = '/dfs/project/perturb-gnn/datasets/trans_norman_adamson/norman2019.h5ad'
    elif args['dataset'] == 'Frangieh2020':
        data_path = '/dfs/project/perturb-gnn/Frangieh2020_coculture_hvg+perts_more_de_in_perts.h5ad'
    args['data_path'] = data_path

    s = time()
    adata = sc.read_h5ad(data_path)
    if 'gene_symbols' not in adata.var.columns.values:
        adata.var['gene_symbols'] = adata.var['gene_name']
    gene_list = [f for f in adata.var.gene_symbols.values]
    args['gene_list'] = gene_list
    args['num_genes'] = len(gene_list)
    ctrl_expression = torch.tensor(np.mean(adata.X[adata.obs.condition == 'ctrl'], axis = 0)).reshape(-1,).to(args['device'])
    
    if args['mean_control']:
        from utils import get_mean_ctrl
        args['ctrl'] = get_mean_ctrl(adata).values
        
    try:
        args['num_ctrl_samples'] = adata.uns['num_ctrl_samples']
    except:
        args['num_ctrl_samples'] = 1

    print('Training '+ args['exp_name'])

    # Pertrubation dataloader
    pertdl = PertDataloader(adata, args)
    
    if args['network_type'] == 'all':    
        
        for i in ['string_ppi', 'co-expression_train', 'gene_ontology']:
            
            edge_list = get_similarity_network(i, args['dataset'], adata, pertdl, args, args['sim_gnn_gene_threshold'], args['sim_gnn_gene_k'],  args['network_randomize_pert'])        
            sim_network = GeneSimNetwork(edge_list, args['gene_list'], node_map = pertdl.node_map)
            args['G_sim_' + i] = sim_network.edge_index
            args['G_sim_weight_' + i] = sim_network.edge_weight
    
    else:
        edge_list = get_similarity_network(args['network_type'], args['dataset'], adata, pertdl, args, args['sim_gnn_gene_threshold'], args['sim_gnn_gene_k'],  args['network_randomize_pert'])
    
        sim_network = GeneSimNetwork(edge_list, args['gene_list'], node_map = pertdl.node_map)
        args['G_sim'] = sim_network.edge_index
        args['G_sim_weight'] = sim_network.edge_weight
        
    if args['gene_sim_pos_emb'] or args['model'] == 'GNN':
        edge_list = get_similarity_network(args['network_type_gene'], args['dataset'], adata, pertdl, args, args['sim_gnn_gene_threshold'], args['sim_gnn_gene_k'],  args['network_randomize_pert'])
        sim_network = GeneSimNetwork(edge_list, args['gene_list'], node_map = pertdl.node_map)

        args['G_coexpress'] = sim_network.edge_index
        args['G_coexpress_weight'] = sim_network.edge_weight
    
    if args['filter'] != 'N/A':
        pert_full_id2pert = dict(adata.obs[['cov_drug_dose_name', 'condition']].values)
        args['filter_status'] = True
        if args['filter'] == 'non_pert_zero':
            dict_filter = adata.uns['non_zeros_gene_idx']
        elif args['filter'] == 'non_dropout':
            dict_filter = adata.uns['non_dropout_gene_idx']
        dict_filter = {pert_full_id2pert[i]: j for i,j in dict_filter.items()}
    else:
        args['filter_status'] = False
        dict_filter = None
        
    print('Finished data setup, in total takes ' + str((time() - s)/60)[:5] + ' min')
    
    print('Initializing model... ')
    
    if args['model'] == 'PertNet':
        model = PertNet(args)
    elif args['model'] == 'No_Perturb':
        model = No_Perturb()
    elif args['model'] == 'GNN':
        model = GNN(args)
    elif args['model'] == 'DNN':
        model = DNN(args)
    
    if args['model'] == 'No_Perturb': 
        best_model = model
    else:
        best_model = train(model, pertdl.loaders['train_loader'],
                              pertdl.loaders['val_loader'],
                              args, ctrl_expression, dict_filter, device=args["device"])
            
    # Save model outputs and best model
    if args['save_model']:
        print('Saving model....')
        np.save('./saved_args/'+ args['exp_name'], args)
        torch.save(best_model, './saved_models/' +args['exp_name'])    
    
    print('Start testing....')    
    
    test_res = evaluate(pertdl.loaders['test_loader'],best_model, args)    
    test_metrics, test_pert_res = compute_metrics(test_res)    
    
    log = "Final best performing model: Test_DE: {:.4f}"
    print(log.format(test_metrics['mse_de']))
    if args['wandb']:
        metrics = ['mse', 'mae', 'spearman', 'pearson', 'r2']
        for m in metrics:
            wandb.log({'test_' + m: test_metrics[m],
                       'test_de_'+m: test_metrics[m + '_de']                     
                      })
        if args['func_readout']:
            wandb.log({'test_func_mse': test_metrics['func_mse']})
    
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
    
    
    if 'umi' in args['dataset']:
        
        train_res = evaluate(pertdl.loaders['train_loader'], best_model, args)
        val_res = evaluate(pertdl.loaders['val_loader'], best_model, args)
        
        high_umi_idx = get_high_umi_idx(args['gene_list'])
        from utils import get_mean_ctrl
        mean_control = get_mean_ctrl(adata)
        test_synergy_loss = {}
        
        for subtype in ['POTENTIATION', 'SYNERGY_SIMILAR_PHENO', 'SYNERGY_DISSIMILAR_PHENO', 'SUPPRESSOR', 'ADDITIVE']:

            train_loss, train_mag = compute_synergy_loss(train_res, mean_control,
                                                high_umi_idx, subtype = subtype)
            
            val_train_loss, val_train_mag = compute_synergy_loss(combine_res(train_res, val_res), mean_control,
                                                high_umi_idx, subtype = subtype)
            
            test_train_loss, test_train_mag = compute_synergy_loss(combine_res(train_res, test_res), mean_control,
                                                high_umi_idx, subtype = subtype) 
            
            if args['wandb']:
                wandb.log({'test_' + subtype + '_loss': test_train_loss - train_loss})
                wandb.log({'train_' + subtype + '_loss': train_loss})
                wandb.log({'val_' + subtype + '_loss': val_train_loss - train_loss})

                wandb.log({'test_' + subtype + '_pred_mag': test_train_mag - train_mag})
                wandb.log({'val_' + subtype + '_pred_mag': val_train_mag - train_mag})
                wandb.log({'train_' + subtype + '_pred_mag': train_mag})
                
    if ('_' in args['dataset']) and (args['dataset'].split('_')[1] == 'Adamson2016'):
        print('Starting Testing on Cross Dataset....')
        ## cross dataset evaluation
        if args['dataset'].split('_')[1] == 'Adamson2016':
            adata_cross = sc.read_h5ad('/dfs/project/perturb-gnn/datasets/trans_norman_adamson/adamson2016.h5ad')
            args['dataset'] = 'Norman2019_Adamson2016_Target'
        
        args['split'] = 'no_split'
        
        if 'gene_symbols' not in adata_cross.var.columns.values:
            adata_cross.var['gene_symbols'] = adata_cross.var['gene_name']
        
        pertdl_cross_dataset = PertDataloader(adata_cross, network.G, network.weights, args)
        
        test_res = evaluate(pertdl_cross_dataset.loaders['test_loader'],
                            pertdl_cross_dataset.loaders['edge_index'],
                            pertdl_cross_dataset.loaders['edge_attr'], best_model, args)
    
        test_metrics, test_pert_res = compute_metrics(test_res)

        log = "Final best performing model: Test_DE on New Dataset: {:.4f}, R2 {:.4f} "
        print(log.format(test_metrics['mse_de'], test_metrics['r2_de']))
        if args['wandb']:
            metrics = ['mse', 'mae', 'spearman', 'pearson', 'r2']
            for m in metrics:
                wandb.log({'cross_dataset_test_' + m: test_metrics[m],
                           'cross_dataset_test_de_'+m: test_metrics[m + '_de']
                           #'test_de_macro_'+m: test_metrics[m + '_de_macro'],
                           #'test_macro_'+m: test_metrics[m + '_macro'],                       
                          })
        out = deeper_analysis(adata_cross, test_res)

        metrics = ['frac_in_range', 'frac_in_range_45_55', 'frac_in_range_40_60', 'frac_in_range_25_75', 'mean_sigma', 'std_sigma', 'frac_sigma_below_1', 'frac_sigma_below_2', 'pearson_delta',
                   'pearson_delta_de', 'fold_change_gap_all', 'pearson_delta_top200_hvg', 'fold_change_gap_upreg_3', 
                   'fold_change_gap_downreg_0.33', 'fold_change_gap_downreg_0.1', 'fold_change_gap_upreg_10', 
                   'pearson_top200_hvg', 'pearson_top200_de', 'pearson_top20_de', 'pearson_delta_top200_de', 
                   'pearson_top100_de', 'pearson_delta_top100_de', 'pearson_delta_top50_de', 'pearson_top50_de', 'pearson_delta_top20_de',
                   'mse_top200_hvg', 'mse_top100_de', 'mse_top200_de', 'mse_top50_de', 'mse_top20_de']


        if args['wandb']:
            for m in metrics:
                wandb.log({'cross_dataset_test_' + m: np.mean([j[m] for i,j in out.items() if m in j])})



    print('Done!')


def parse_arguments():
    """
    Argument parser
    """

    # dataset arguments
    parser = argparse.ArgumentParser(description='Perturbation response')
    
    parser.add_argument('--dataset', type=str, choices = ['Norman2019',
                                                          'Adamson2016',
                                                          'Dixit2016',
                                                          'Norman2019_GI',
                                                          'Norman2019_umi',
                                                          'Frangieh2020' 
                                                          'Norman2019_umi_all_poss',
                                                          'Norman2019_Adamson2016'], default="Norman2019_umi")
    parser.add_argument('--split', type=str, choices = ['simulation', 'simulation_single', 'combo_seen0', 'combo_seen1', 'combo_seen2', 'single', 'single_only', 'custom'],
                                                        default="combo_seen2")
    parser.add_argument('--split_path', type=str, default='N/A')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test_set_fraction', type=float, default=0.1)
    parser.add_argument('--train_gene_set_size', type=float, default=0.75)
    parser.add_argument('--combo_seen2_train_frac', type=float, default=0.75)
    parser.add_argument('--test_perts', type=str, default='N/A')
    parser.add_argument('--test_pert_genes', type=str, default='N/A')
    parser.add_argument('--more_samples', type=int, default=0)
    parser.add_argument('--only_test_set_perts', default=False, action='store_true')
    
    # Dataloader related
    parser.add_argument('--pert_feats', default=True, action='store_false',
                        help='Separate feature to indicate perturbation')
    parser.add_argument('--pert_delta', default=False, action='store_true',
                        help='Represent perturbed cells using delta gene '
                             'expression')
    parser.add_argument('--perturbation_key', type=str, default="condition")
    parser.add_argument('--binary_pert', default=True, action='store_false')
    parser.add_argument('--ctrl_remove_train', default=False, action='store_true')
    parser.add_argument('--func_readout', type=str, default=None)

    
    # training arguments
    parser.add_argument('--device', type=str, default='cuda:5')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_step_size', type=int, default=1)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--print_progress_steps', type=int, default=50)
                        
    # model arguments
    parser.add_argument('--node_hidden_size', type=int, default=64,
                        help='hidden dimension for GNN')    
    parser.add_argument('--model', choices = ['No_Perturb', 'PertNet', 'GNN', 'DNN'], 
                        type = str, default = 'PertNet', help='model name')
    parser.add_argument('--num_of_gnn_layers', type=int, default=1)    
    
    parser.add_argument('--gene_pert_agg', default='sum', choices = ['sum', 'concat+w','sum_trans'], type = str)
    parser.add_argument('--pert_emb_lambda', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--uncertainty', default=False, action='store_true')
    parser.add_argument('--uncertainty_reg', type=float, default=1)
    parser.add_argument('--uncertainty_reg_core', type=float, default=1)

    parser.add_argument('--gene_sim_pos_emb', default=False, action='store_true')
    parser.add_argument('--gene_sim_pos_emb_num_layers', type=int, default=1)    
    parser.add_argument('--model_backend', choices = ['GCN', 'GAT', 'SGC'], 
                        type = str, default = 'SGC', help='model name')  
    parser.add_argument('--indv_out_hidden_size', type=int, default=4)    
    parser.add_argument('--num_mlp_layers', type=int, default=3)    
    parser.add_argument('--network_randomize_pert', default=False, action='store_true')	
    parser.add_argument('--network_randomize_gene', default=False, action='store_true')

    parser.add_argument('--sim_gnn_gene_k', type=int, default=5)    
    parser.add_argument('--sim_gnn_gene_threshold', type=float, default=0.4)  
    parser.add_argument('--network_type', default = 'gene_ontology', type=str,
                        choices = ['co-expression_train', 'gene_ontology', 'string_ppi', 'all'])
    parser.add_argument('--indv_out_layer', default=True, action='store_true')
    parser.add_argument('--cross_gene_MLP', default=False, action='store_true')
    parser.add_argument('--cross_gene_decoder', default='na', choices = ['mlp', 'skip-connect', 'skip-connect-mlp'], type=str)
    
    parser.add_argument('--network_type_gene', default = 'co-expression_train', type=str, choices = ['co-expression_train', 'gene_ontology', 'string_ppi', 'all'])
    parser.add_argument('--indv_out_layer_uncertainty', default=False, action='store_true')
    parser.add_argument('--add_gene_expression', default=False, action='store_true')
    parser.add_argument('--add_gene_expression_back', default=False, action='store_true')
    parser.add_argument('--post_coexpress', default=False, action='store_true')
    parser.add_argument('--mlp_pert_fuse', default=False, action='store_true')
    parser.add_argument('--set_self_attention', default=False, action='store_true')
    parser.add_argument('--set_self_attention_num_head', default=4, type = int)
    parser.add_argument('--set_self_attention_layernorm', default=False, action='store_true')
    parser.add_argument('--set_self_attention_agg', choices=['sum', 'weight_ori_emb', 'weight_post_emb'], type = str)
    parser.add_argument('--set_self_attention_post_mlp', default=False, action='store_true')
    parser.add_argument('--cg_mlp', choices=['baseline', 'small', 'deep', 'wide'], default = 'baseline', type = str)
    parser.add_argument('--pert_fuse_linear_to_mlp', default=False, action='store_true')
    parser.add_argument('--de_drop', default=False, action='store_true')
    parser.add_argument('--mean_control', default=False, action='store_true')
    parser.add_argument('--add_gene_expression_before_cross_gene', default=False, action='store_true')
    parser.add_argument('--add_gene_expression_gene_specific', default=False, action='store_true')
    
    parser.add_argument('--expression_concat', default=False, action='store_true')
    parser.add_argument('--expression_no_bn', default=False, action='store_true')
    
    parser.add_argument('--emb_trans_v2_mlp', default=False, action='store_true')
    parser.add_argument('--emb_trans_mlp', default=False, action='store_true')
    parser.add_argument('--pert_base_trans_w_mlp', default=False, action='store_true')
    parser.add_argument('--gene_base_trans_w_mlp', default=False, action='store_true')
    parser.add_argument('--transform_mlp', default=False, action='store_true')
    parser.add_argument('--pert_fuse_mlp', default=False, action='store_true')
    
    # loss
    parser.add_argument('--pert_loss_wt', type=int, default=1,
                        help='weights for perturbed cells compared to control cells')
    parser.add_argument('--loss_type', type=str, default='macro', choices = ['macro', 'micro'],
                        help='micro averaged or not')
    parser.add_argument('--loss_mode', choices = ['l1', 'l2', 'l3', 'weight_y'], type = str, default = 'l3')
    parser.add_argument('--focal_gamma', type=int, default=2)    
    parser.add_argument('--loss_direction', default=True, action='store_true')
    parser.add_argument('--direction_lambda', type=float, default=1e-1)    
    parser.add_argument('--filter', type=str, default='N/A', choices = ['non_pert_zero', 'non_dropout'])    

    # wandb related
    parser.add_argument('--wandb', default=False, action='store_true',
                    help='Use wandb or not')
    parser.add_argument('--wandb_sweep', default=False, action='store_true',
                help='Use wandb or not')
    parser.add_argument('--project_name', type=str, default='pert_gnn',
                        help='project name')
    parser.add_argument('--entity_name', type=str, default='kexinhuang',
                        help='entity name')
    parser.add_argument('--exp_name', type=str, default='testing',
                        help='entity name')
    
    # misc
    parser.add_argument('--save_model', default=True, action='store_true')
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    trainer(parse_arguments())
