import torch
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import pickle
import sys, os
import requests
from torch_geometric.data import Data
from zipfile import ZipFile
import tarfile
from sklearn.linear_model import TheilSenRegressor
from dcor import distance_correlation
from multiprocessing import Pool


# Measured as "percent relative to control", e.g. 100 is the same as control,
# 200 is a 2X overexpression, 20 is 5X underexpression, 0 is knockout.
DEFAULT_MAGNITUDE = 100

def rm_magnitude(gene_with_magnitude: str) -> str:
    return gene_with_magnitude.split('*')[0]

def get_pert_genes(pert: list[str]) -> list[str]:
    # Remove perturbation mangnitude.
    # TODO: can be combined with `get_genes_from_perts`
    return [rm_magnitude(g) for g in pert.split('+')]

def get_pert_magnitude(pert: str) -> list[int]:
    # Returns magnitudes for genes in `pert`.
    return [
        int(gene.split('*')[-1]) if '*' in gene else DEFAULT_MAGNITUDE
        for gene in pert.split('+')
        if gene != 'ctrl'
    ]

def parse_single_pert(i):
    a = i.split('+')[0]
    b = i.split('+')[1]
    if a == 'ctrl':
        pert = b
    else:
        pert = a
    return rm_magnitude(pert)

def parse_combo_pert(i):
    return get_pert_genes(i)[:2]

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


def dataverse_download(url, save_path):
    """dataverse download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """

    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        print_sys("Downloading...")
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()


def zip_data_download_wrapper(url, save_path, data_path):

    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        dataverse_download(url, save_path + '.zip')
        print_sys('Extracting zip file...')
        with ZipFile((save_path + '.zip'), 'r') as zip:
            zip.extractall(path = data_path)
        print_sys("Done!")

def tar_data_download_wrapper(url, save_path, data_path):

    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        dataverse_download(url, save_path + '.tar.gz')
        print_sys('Extracting tar file...')
        with tarfile.open(save_path  + '.tar.gz') as tar:
            tar.extractall(path= data_path)
        print_sys("Done!")

def get_go_auto(gene_list, data_path, data_name):
    go_path = os.path.join(data_path, data_name, 'go.csv')

    if os.path.exists(go_path):
        return pd.read_csv(go_path)
    else:
        ## download gene2go.pkl
        if not os.path.exists(os.path.join(data_path, 'gene2go.pkl')):
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
            dataverse_download(server_path, os.path.join(data_path, 'gene2go.pkl'))
        with open(os.path.join(data_path, 'gene2go.pkl'), 'rb') as f:
            gene2go = pickle.load(f)

        gene2go = {i: list(gene2go[i]) for i in gene_list if i in gene2go}
        edge_list = []
        for g1 in tqdm(gene2go.keys()):
            for g2 in gene2go.keys():
                edge_list.append((g1, g2, len(np.intersect1d(gene2go[g1],
                   gene2go[g2]))/len(np.union1d(gene2go[g1], gene2go[g2]))))

        edge_list_filter = [i for i in edge_list if i[2] > 0]
        further_filter = [i for i in edge_list if i[2] > 0.1]
        df_edge_list = pd.DataFrame(further_filter).rename(columns = {0: 'gene1',
                                                                      1: 'gene2',
                                                                      2: 'score'})

        df_edge_list = df_edge_list.rename(columns = {'gene1': 'source',
                                                      'gene2': 'target',
                                                      'score': 'importance'})
        df_edge_list.to_csv(go_path, index = False)
        return df_edge_list

def get_go(df_gene2go):
    df_gene2go['Entry name'] = df_gene2go['Entry name'].apply(lambda x: x.split('_')[0])
    df_gene2go = df_gene2go[df_gene2go['Gene ontology IDs'].notnull()]
    df_gene2go = df_gene2go.rename(columns = {[i for i in df_gene2go.columns.values
                                            if 'yourlist' in i][0]: 'gene_id'})
    geneid2go = dict(df_gene2go[['gene_id', 'Gene ontology IDs']].values)

    gene2go = {}
    for i,j in geneid2go.items():
        j = [k.strip() for k in j.split(';')]
        for k in i.split(','):
            gene2go[ensembl2genename[k]] = j

    from tqdm import tqdm
    edge_list = []
    for g1 in tqdm(gene2go.keys()):
        for g2 in gene2go.keys():
            edge_list.append((g1, g2, len(np.intersect1d(gene2go[g1],
                    gene2go[g2]))/len(np.union1d(gene2go[g1], gene2go[g2]))))

    edge_list_filter = [i for i in edge_list if i[2] > 0]
    further_filter = [i for i in edge_list if i[2] > 0.1]
    df_edge_list = pd.DataFrame(further_filter).rename(columns = {0: 'gene1',
                                                                  1: 'gene2',
                                                                  2: 'score'})

    df_edge_list = df_edge_list.rename(columns = {'gene1': 'source',
                                                  'gene2': 'target',
                                                  'score': 'importance'})
    return df_edge_list

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

def get_GO_edge_list(args):
    g1, gene2go = args
    edge_list = []
    for g2 in gene2go.keys():
        score = len(gene2go[g1].intersection(gene2go[g2])) / len(
            gene2go[g1].union(gene2go[g2]))
        if score > 0.1:
            edge_list.append((g1, g2, score))
    return edge_list

def make_GO(data_path, pert_list, data_name, num_workers=25, save=True):
    """
    Creates Gene Ontology graph from a custom set of genes
    """

    fname = './data/go_essential_' + data_name + '.csv'
    if os.path.exists(fname):
        return pd.read_csv(fname)

    with open(os.path.join(data_path, 'gene2go_all.pkl'), 'rb') as f:
        gene2go = pickle.load(f)
    gene2go = {i: gene2go[i] for i in pert_list}

    print('Creating custom GO graph, this can take a few minutes')
    with Pool(num_workers) as p:
        all_edge_list = list(
            tqdm(p.imap(get_GO_edge_list, ((g, gene2go) for g in gene2go.keys())),
                      total=len(gene2go.keys())))
    edge_list = []
    for i in all_edge_list:
        edge_list = edge_list + i

    df_edge_list = pd.DataFrame(edge_list).rename(
        columns={0: 'source', 1: 'target', 2: 'importance'})

    if save:
        print('Saving edge_list to file')
        df_edge_list.to_csv(fname, index=False)

    return df_edge_list

def get_similarity_network(network_type, adata, threshold, k,
                           data_path, data_name, split, seed, train_gene_set_size,
                           set2conditions, default_pert_graph=True, pert_list=None,
                           go_path: str = None):

    if network_type == 'co-express':
        df_out = get_coexpression_network_from_train(adata, threshold, k,
                                                     data_path, data_name, split,
                                                     seed, train_gene_set_size,
                                                     set2conditions)
    elif network_type == 'go':
        if go_path is not None:
            df_jaccard = pd.read_csv(go_path)
            print_sys(f'Loaded {len(df_jaccard)} similarities from {go_path}')
        else:
            if default_pert_graph:
                server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934319'
                tar_data_download_wrapper(server_path,
                                        os.path.join(data_path, 'go_essential_all'),
                                        data_path)
                df_jaccard = pd.read_csv(os.path.join(data_path,
                                        'go_essential_all/go_essential_all.csv'))
            else:
                df_jaccard = make_GO(data_path, pert_list, data_name)

        df_out = df_jaccard.groupby('target').apply(lambda x: x.nlargest(k + 1,
                                    ['importance'])).reset_index(drop = True)

    return df_out

def get_coexpression_network_from_train(adata, threshold, k, data_path,
                                        data_name, split, seed, train_gene_set_size,
                                        set2conditions):

    fname = os.path.join(os.path.join(data_path, data_name), split + '_'  +
                         str(seed) + '_' + str(train_gene_set_size) + '_' +
                         str(threshold) + '_' + str(k) +
                         '_co_expression_network.csv')

    if os.path.exists(fname):
        return pd.read_csv(fname)
    else:
        gene_list = [f for f in adata.var.gene_name.values]
        idx2gene = dict(zip(range(len(gene_list)), gene_list))
        X = adata.X
        train_perts = set2conditions['train']
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
        df_co_expression = pd.DataFrame(df_g).rename(columns = {0: 'source',
                                                                1: 'target',
                                                                2: 'importance'})
        df_co_expression.to_csv(fname, index = False)
        return df_co_expression

def filter_pert_in_go(condition, pert_names):
    if condition == 'ctrl':
        return True
    else:
        cond1, cond2 = get_pert_genes(condition)[:2]
        num_ctrl = (cond1 == 'ctrl') + (cond2 == 'ctrl')
        num_in_perts = (cond1 in pert_names) + (cond2 in pert_names)
        if num_ctrl + num_in_perts == 2:
            return True
        else:
            return False

def uncertainty_loss_fct(pred, logvar, y, perts, reg = 0.1, ctrl = None,
                         direction_lambda = 1e-3, dict_filter = None):
    gamma = 2
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
    for p in set(perts):
        if p!= 'ctrl':
            retain_idx = dict_filter[p]
            pred_p = pred[np.where(perts==p)[0]][:, retain_idx]
            y_p = y[np.where(perts==p)[0]][:, retain_idx]
            logvar_p = logvar[np.where(perts==p)[0]][:, retain_idx]
        else:
            pred_p = pred[np.where(perts==p)[0]]
            y_p = y[np.where(perts==p)[0]]
            logvar_p = logvar[np.where(perts==p)[0]]

        # uncertainty based loss
        losses += torch.sum((pred_p - y_p)**(2 + gamma) + reg * torch.exp(
            -logvar_p)  * (pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]

        # direction loss
        if p!= 'ctrl':
            losses += torch.sum(direction_lambda *
                                (torch.sign(y_p - ctrl[retain_idx]) -
                                 torch.sign(pred_p - ctrl[retain_idx]))**2)/\
                                 pred_p.shape[0]/pred_p.shape[1]
        else:
            losses += torch.sum(direction_lambda *
                                (torch.sign(y_p - ctrl) -
                                 torch.sign(pred_p - ctrl))**2)/\
                                 pred_p.shape[0]/pred_p.shape[1]

    return losses/(len(set(perts)))


def loss_fct(pred, y, perts, ctrl = None, direction_lambda = 1e-3, dict_filter = None):
    gamma = 2
    mse_p = torch.nn.MSELoss()
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)

    for p in set(perts):
        pert_idx = np.where(perts == p)[0]

        # during training, we remove the all zero genes into calculation of loss.
        # this gives a cleaner direction loss. empirically, the performance stays the same.
        if p!= 'ctrl':
            retain_idx = dict_filter[p]
            pred_p = pred[pert_idx][:, retain_idx]
            y_p = y[pert_idx][:, retain_idx]
        else:
            pred_p = pred[pert_idx]
            y_p = y[pert_idx]
        #losses += torch.sum((torch.mean(pred_p, 0) - y_p)**(2+gamma))/pred_p.shape[1]
        losses = losses + torch.sum((pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]

        ## direction loss
        if (p!= 'ctrl'):
            losses = losses + torch.sum(direction_lambda *
                                (torch.sign(y_p - ctrl[retain_idx]) -
                                 torch.sign(pred_p - ctrl[retain_idx]))**2)/\
                                 pred_p.shape[0]/pred_p.shape[1]
        else:
            losses = losses + torch.sum(direction_lambda * (torch.sign(y_p - ctrl) -
                                                torch.sign(pred_p - ctrl))**2)/\
                                                pred_p.shape[0]/pred_p.shape[1]
    return losses/(len(set(perts)))


def print_sys(s):
    """system print

    Args:
        s (str): the string to print
    """
    print(s, flush = True, file = sys.stderr)

def create_cell_graph_for_prediction(X, pert_idx, pert_gene):

    if pert_idx is None:
        pert_idx = [-1]
    pert_magnitude = [get_pert_magnitude(g)[0] for g in pert_gene]
    return Data(x=torch.Tensor(X).T, pert_idx=pert_idx, pert=pert_gene, pert_magnitude=pert_magnitude)


def create_cell_graph_dataset_for_prediction(pert_gene, ctrl_adata, gene_names,
                                             device, num_samples = 300):

    # Get the indices (and signs) of applied perturbation
    pert_gene_no_magnitude = [rm_magnitude(p) for p in pert_gene]
    pert_idx = [np.where(p == np.array(gene_names))[0][0] for p in pert_gene_no_magnitude]
    Xs = ctrl_adata[np.random.randint(0, len(ctrl_adata), num_samples), :].X.toarray()
    # Create cell graphs
    cell_graphs = [create_cell_graph_for_prediction(X, pert_idx, pert_gene).to(device) for X in Xs]
    return cell_graphs

##
##GI related utils
##

def get_coeffs(singles_expr, first_expr, second_expr, double_expr):
    results = {}
    results['ts'] = TheilSenRegressor(fit_intercept=False,
                          max_subpopulation=1e5,
                          max_iter=1000,
                          random_state=1000)
    X = singles_expr
    y = double_expr
    results['ts'].fit(X, y.ravel())
    Zts = results['ts'].predict(X)
    results['c1'] = results['ts'].coef_[0]
    results['c2'] = results['ts'].coef_[1]
    results['mag'] = np.sqrt((results['c1']**2 + results['c2']**2))

    results['dcor'] = distance_correlation(singles_expr, double_expr)
    results['dcor_singles'] = distance_correlation(first_expr, second_expr)
    results['dcor_first'] = distance_correlation(first_expr, double_expr)
    results['dcor_second'] = distance_correlation(second_expr, double_expr)
    results['corr_fit'] = np.corrcoef(Zts.flatten(), double_expr.flatten())[0,1]
    results['dominance'] = np.abs(np.log10(results['c1']/results['c2']))
    results['eq_contr'] = np.min([results['dcor_first'], results['dcor_second']])/\
                        np.max([results['dcor_first'], results['dcor_second']])

    return results

def get_GI_params(preds, combo):

    singles_expr = np.array([preds[combo[0]], preds[combo[1]]]).T
    first_expr = np.array(preds[combo[0]]).T
    second_expr = np.array(preds[combo[1]]).T
    double_expr = np.array(preds[combo[0]+'_'+combo[1]]).T

    return get_coeffs(singles_expr, first_expr, second_expr, double_expr)

def get_GI_genes_idx(adata, GI_gene_file):
    # Genes used for linear model fitting
    GI_genes = np.load(GI_gene_file, allow_pickle=True)
    GI_genes_idx = np.where([g in GI_genes for g in adata.var.gene_name.values])[0]

    return GI_genes_idx

def get_mean_control(adata):
    cols = adata.var.gene_name.values.astype('str')
    mean_ctrl_exp = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()
    return mean_ctrl_exp

def get_genes_from_perts(perts):
    """
    Returns list of genes involved in a given perturbation list
    """

    if type(perts) is str:
        perts = [perts]
    gene_list = [p.split('+') for p in np.unique(perts)]
    gene_list = [item for sublist in gene_list for item in sublist]
    gene_list = [g for g in gene_list if g != 'ctrl']
    return list(np.unique(gene_list))
