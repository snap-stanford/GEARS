import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import TheilSenRegressor
from inference import GIs
import torch.nn as nn

## helper function
def parse_single_pert(i):
    a = i.split('+')[0]
    b = i.split('+')[1]
    if a == 'ctrl':
        pert = b
    else:
        pert = a
    return pert

def parse_combo_pert(i):
    return i.split('+')[0], i.split('+')[1]

def combine_res(res_1, res_2):
    res_out = {}
    for key in res_1:
        res_out[key] = np.concatenate([res_1[key], res_2[key]])
    return res_out

def parse_any_pert(p):
    if ('ctrl' in p) and (p != 'ctrl'):
        return [parse_single_pert(p)]
    elif 'ctrl' not in p:
        out = parse_combo_pert(p)
        return [out[0], out[1]]

def np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)


def get_similarity_network(network_type, dataset, adata, pertdl, args, threshold = 0.4, k = 10, network_randomize = False):
    
    if network_type == 'co-expression_train':
        df_out = get_coexpression_network_from_train(adata, pertdl, args, threshold, k)
    elif network_type == 'gene_ontology':
        if dataset == 'Norman2019' or dataset == 'Norman2019_GI':
            df_jaccard = pd.read_csv('/dfs/user/kexinh/perturb_GNN/kexin/gene_sim_jc_filter_norman.csv')
        elif dataset == 'Norman2019_umi':
            df_jaccard = pd.read_csv('/dfs/user/kexinh/perturb_GNN/kexin/gene_sim_jc_filter_norman_new.csv')
        elif dataset == 'Norman2019_umi_all_poss':
            df_jaccard = pd.read_csv('/dfs/user/kexinh/perturb_GNN/kexin/gene_sim_jc_filter_norman_new.csv')
        elif dataset == 'Adamson2016':
            df_jaccard = pd.read_csv('/dfs/user/kexinh/perturb_GNN/kexin/gene_sim_jc_filter_adamson.csv')
        elif dataset == 'Frangieh2020':
            df_jaccard = pd.read_csv('/dfs/user/kexinh/perturb_GNN/kexin/gene_sim_jc_filter_frangieh.csv')
        elif dataset == 'Dixit2016':
            df_jaccard = pd.read_csv('/dfs/user/kexinh/perturb_GNN/kexin/gene_sim_jc_filter_dixit.csv')
        df_out = df_jaccard.groupby('target').apply(lambda x: x.nlargest(k + 1,['importance'])).reset_index(drop = True)
    elif network_type == 'string_ppi':
        df_string =  pd.read_csv('/dfs/project/perturb-gnn/graphs/STRING_full_9606.csv')
        gene_list = adata.var.gene_name.values
        df_out = get_string_ppi(df_string, gene_list, k)
        
    if network_randomize:
        df_out['source'] = df_out['source'].sample(frac = 1, random_state=1).values
    return df_out
        
def get_string_ppi(df_string, gene_list, k):        
    df_string = df_string[df_string.source.isin(gene_list)]
    df_string = df_string[df_string.target.isin(gene_list)]
    df_string = df_string.sort_values('importance', ascending=False)
    df_string = df_string.groupby('target').apply(lambda x: x.nlargest(k + 1,['importance'])).reset_index(drop = True)
    return df_string

def get_coexpression_network_from_train(adata, pertdl, args, threshold = 0.4, k = 10):
    import os
    import pandas as pd
    
    fname = './saved_networks/' + args['dataset'] + '_' + args['split'] + '_' + str(args['seed']) + '_' + str(args['test_set_fraction']) + '_' + str(threshold) + '_' + str(k) + '_co_expression_network.csv'
    
    if os.path.exists(fname):
        return pd.read_csv(fname)
    else:
        gene_list = [f for f in adata.var.gene_symbols.values]
        idx2gene = dict(zip(range(len(gene_list)), gene_list)) 
        X = adata.X
        train_perts = pertdl.set2conditions['train']
        X_tr = X[np.isin(adata.obs.condition, [i for i in train_perts if 'ctrl' in i])]
        gene_list = adata.var['gene_name'].values

        X_tr = X_tr.toarray()
        out = np_pearson_cor(X_tr, X_tr)
        out[np.isnan(out)] = 0
        out = np.abs(out)

        out_sort_idx = np.argsort(out)[:, -(k + 1):]
        out_sort_val = np.sort(out)[:, -(k + 1):]

        df_g = []
        for i in range(out_sort_idx.shape[0]):
            target = idx2gene[i]
            for j in range(out_sort_idx.shape[1]):
                df_g.append((idx2gene[out_sort_idx[i, j]], target, out_sort_val[i, j]))

        df_g = [i for i in df_g if i[2] > threshold]
        df_co_expression = pd.DataFrame(df_g).rename(columns = {0: 'source', 1: 'target', 2: 'importance'})
        df_co_expression.to_csv(fname, index = False)
        return df_co_expression
    
def weighted_mse_loss(input, target, weight):
    """
    Weighted MSE implementation
    """
    sample_mean = torch.mean((input - target) ** 2, 1)
    return torch.mean(weight * sample_mean)

def uncertainty_loss_fct(pred, logvar, y, perts, loss_mode = 'l2', gamma = 1, reg = 0.1, 
                        reg_core = 1, loss_direction = False, ctrl = None, direction_lambda = 1e-3,
                        filter_status = False, dict_filter = None, pred_func=None, y_func=None):
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
    if pred_func is not None:
        count_func = torch.logical_not(torch.isnan(y_func)).sum()

    for p in set(perts):
        if (p!= 'ctrl') and filter_status:
            retain_idx = dict_filter[p]
            pred_p = pred[np.where(perts==p)[0]][:, retain_idx]
            y_p = y[np.where(perts==p)[0]][:, retain_idx]
            logvar_p = logvar[np.where(perts==p)[0]][:, retain_idx]
        else:
            pred_p = pred[np.where(perts==p)[0]]
            y_p = y[np.where(perts==p)[0]]
            logvar_p = logvar[np.where(perts==p)[0]]

        pert_idx = np.where(perts == p)[0]
        pred_p = pred[pert_idx]
        y_p = y[pert_idx]
        logvar_p = logvar[pert_idx]

        # GI loss
        if pred_func is not None:
            pred_p_func = pred_func[pert_idx]
            y_p_func = y_func[pert_idx]
            if not torch.isnan(y_p_func)[0]:
                losses += func_beta * torch.sum((pred_p_func -
                                                 y_p_func) ** (2 + gamma)) / \
                          pred_p_func.shape[0] / count_func

        if torch.count_nonzero(y_p, 1)[0] > 2:
            if loss_mode == 'l2':
                losses += torch.sum(0.5 * torch.exp(-logvar_p) * (pred_p - y_p)**2 + 0.5 * logvar_p)/pred_p.shape[0]/pred_p.shape[1]
            elif loss_mode == 'l3':
                losses += reg_core * torch.sum((pred_p - y_p)**(2 + gamma) + reg * torch.exp(-logvar_p) * (pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]
        
        if loss_mode == 'l2':
            losses += torch.sum(0.5 * torch.exp(-logvar_p) * (pred_p - y_p)**2 + 0.5 * logvar_p)/pred_p.shape[0]/pred_p.shape[1]
        elif loss_mode == 'l3':
            #losses += torch.sum(0.5 * torch.exp(-logvar_p) * (pred_p - y_p)**(2 + gamma) + 0.01 * logvar_p)/pred_p.shape[0]/pred_p.shape[1]
            #losses += torch.sum((pred_p - y_p)**(2 + gamma) + 0.1 * torch.exp(-logvar_p) * (pred_p - y_p)**(2 + gamma) + 0.1 * logvar_p)/pred_p.shape[0]/pred_p.shape[1]
            losses += reg_core * torch.sum((pred_p - y_p)**(2 + gamma) + reg * torch.exp(-logvar_p) * (pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]
        if loss_direction:
            if (p!= 'ctrl') and filter_status:
                losses += torch.sum(direction_lambda * (torch.sign(y_p - ctrl[retain_idx]) - torch.sign(pred_p - ctrl[retain_idx]))**2)/pred_p.shape[0]/pred_p.shape[1]

            else:
                losses += torch.sum(direction_lambda * (torch.sign(y_p - ctrl) - torch.sign(pred_p - ctrl))**2)/pred_p.shape[0]/pred_p.shape[1]
            
    return losses/(len(set(perts)))


def loss_fct(pred, y, perts, weight=1, loss_type = 'macro', loss_mode = 'l2', gamma = 1, 
            loss_direction = False, ctrl = None, direction_lambda = 1e-3, 
            filter_status = False, dict_filter = None, pred_func=None, y_func=None):

        # Micro average MSE
        if loss_type == 'macro':
            mse_p = torch.nn.MSELoss()
            perts = np.array(perts)
            losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
            if pred_func is not None:
                count_func = torch.logical_not(torch.isnan(y_func)).sum()
            for p in set(perts):
                pert_idx = np.where(perts == p)[0]
                if (p!= 'ctrl') and filter_status:
                    retain_idx = dict_filter[p]
                    pred_p = pred[pert_idx][:, retain_idx]
                    y_p = y[pert_idx][:, retain_idx]
                else:
                    pred_p = pred[pert_idx]
                    y_p = y[pert_idx]

                # GI loss
                if pred_func is not None:
                    pred_p_func = pred_func[pert_idx]
                    y_p_func = y_func[pert_idx]
                    if not torch.isnan(y_p_func)[0]:
                        losses += func_beta*torch.sum((pred_p_func -
                            y_p_func)**(2+gamma))/pred_p_func.shape[0]/count_func

                # Transcription loss
                if torch.count_nonzero(y_p,1)[0]>2:
                    if loss_mode == 'l2':
                        losses += torch.sum((pred_p - y_p)**2)/pred_p.shape[0]/pred_p.shape[1]
                    elif loss_mode == 'l3':
                        losses += torch.sum((pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]
                    elif loss_mode == 'l1':
                        loss_fct = nn.L1Loss()
                        losses += loss_fct(pred_p, y_p)
                        
                    if loss_direction:
                        if (p!= 'ctrl') and filter_status:
                            losses += torch.sum(direction_lambda * (torch.sign(y_p - ctrl[retain_idx]) - torch.sign(pred_p - ctrl[retain_idx]))**2)/pred_p.shape[0]/pred_p.shape[1]

                        else:
                            losses += torch.sum(direction_lambda * (torch.sign(y_p - ctrl) - torch.sign(pred_p - ctrl))**2)/pred_p.shape[0]/pred_p.shape[1]
                        #losses += torch.sum(direction_lambda * (pred_p - y_p)**(2 + gamma) * (torch.sign(y_p - ctrl) - torch.sign(pred_p - ctrl))**2)/pred_p.shape[0]/pred_p.shape[1]
            return losses/(len(set(perts)))

        else:
            # Weigh the loss for perturbations (unweighted by default)
            #weights = np.ones(len(pred))
            #non_ctrl_idx = np.where([('ctrl' != p) for p in perts])[0]
            #weights[non_ctrl_idx] = weight
            #loss = weighted_mse_loss(pred, y, torch.Tensor(weights).to(pred.device))
            if loss_mode == 'l2':
                loss = torch.sum((pred - y)**2)/pred.shape[0]/pred.shape[1]
            elif loss_mode == 'l3':
                loss = torch.sum((pred - y)**(2 + gamma))/pred.shape[0]/pred.shape[1]

            return loss

def get_high_umi_idx(gene_list):
    # Genes used for linear model fitting
    try:
        high_umi = np.load('../genes_with_hi_mean.npy', allow_pickle=True)
    except:
        high_umi = np.load('./genes_with_hi_mean.npy', allow_pickle=True)
    high_umi_idx = np.where([g in high_umi for g in gene_list])[0]
    return high_umi_idx

def get_mean_ctrl(adata):
    return adata[adata.obs['condition'] == 'ctrl'].to_df().mean().reset_index(
        drop=True)

def get_single_name(g, all_perts):
    name = g+'+ctrl'
    if name in all_perts:
        return name
    else:
        return 'ctrl+'+g


def get_test_set_results_seen2(res, sel_GI_type):
    # Get relevant test set results
    test_pert_cats = [p for p in np.unique(res['pert_cat']) if
                      p in GIs[sel_GI_type] or 'ctrl' in p]
    pred_idx = np.where([t in test_pert_cats for t in res['pert_cat']])
    out = {}
    for key in res:
        out[key] = res[key][pred_idx]
    return out

## Synergy loss calculation functions
def get_all_vectors(all_res, mean_control, double,
                    single1, single2, high_umi_idx):
    # Pred
    pred_df = pd.DataFrame(all_res['pred'])
    pred_df['condition'] = all_res['pert_cat']
    subset_df = pred_df[pred_df['condition'] == double].iloc[:, :-1]
    delta_double_pred = subset_df.mean(0) - mean_control
    single_df_1_pred = pred_df[pred_df['condition'] == single1].iloc[:, :-1]
    single_df_2_pred = pred_df[pred_df['condition'] == single2].iloc[:, :-1]

    # True
    truth_df = pd.DataFrame(all_res['truth'])
    truth_df['condition'] = all_res['pert_cat']
    subset_df = truth_df[truth_df['condition'] == double].iloc[:, :-1]
    delta_double_truth = subset_df.mean(0) - mean_control
    single_df_1_truth = truth_df[truth_df['condition'] == single1].iloc[:, :-1]
    single_df_2_truth = truth_df[truth_df['condition'] == single2].iloc[:, :-1]

    delta_single_truth_1 = single_df_1_truth.mean(0) - mean_control
    delta_single_truth_2 = single_df_2_truth.mean(0) - mean_control
    delta_single_pred_1 = single_df_1_pred.mean(0) - mean_control
    delta_single_pred_2 = single_df_2_pred.mean(0) - mean_control

    return {'single_pred_1': delta_single_pred_1.values[high_umi_idx],
            'single_pred_2': delta_single_pred_2.values[high_umi_idx],
            'double_pred': delta_double_pred.values[high_umi_idx],
            'single_truth_1': delta_single_truth_1.values[high_umi_idx],
            'single_truth_2': delta_single_truth_2.values[high_umi_idx],
            'double_truth': delta_double_truth.values[high_umi_idx]}


def get_coeffs_synergy(singles_expr, double_expr):
    results = {}
    results['ts'] = TheilSenRegressor(fit_intercept=False,
                                      max_subpopulation=1e5,
                                      max_iter=1000,
                                      random_state=1000)
    X = singles_expr
    y = double_expr
    try:
        results['ts'].fit(X, y.ravel())
    except:
        print(X)
        print(y)
    results['c1'] = results['ts'].coef_[0]
    results['c2'] = results['ts'].coef_[1]
    results['mag'] = np.sqrt((results['c1'] ** 2 + results['c2'] ** 2))
    return results


def Fit(all_vectors, type_='pertnet'):
    if type_ == 'pertnet':
        singles_expr = np.array(
            [all_vectors['single_pred_1'], all_vectors['single_pred_2']]).T
        first_expr = np.array([all_vectors['single_pred_1']]).T
        second_expr = np.array([all_vectors['single_pred_2']]).T
        double_expr = np.array(all_vectors['double_pred']).T

    elif type_ == 'truth':
        singles_expr = np.array(
            [all_vectors['single_truth_1'], all_vectors['single_truth_2']]).T
        first_expr = np.array([all_vectors['single_truth_1']]).T
        second_expr = np.array([all_vectors['single_truth_2']]).T
        double_expr = np.array(all_vectors['double_truth']).T

    return get_coeffs_synergy(singles_expr, double_expr)


def get_linear_params(pred_res, high_umi_idx, mean_control, all_perts):
    results = {}
    for c in set(pred_res['pert_cat']):
        if 'ctrl' in c:
            continue
        double = c
        single1 = get_single_name(double.split('+')[0], all_perts)
        single2 = get_single_name(double.split('+')[1], all_perts)
        all_vectors = get_all_vectors(pred_res, mean_control, double,
                                      single1, single2, high_umi_idx)

        pertnet_results = Fit(all_vectors, type_='pertnet')
        truth_results = Fit(all_vectors, type_='truth')

        results[c] = {
            'truth': truth_results,
            'pred': pertnet_results}

    return results
