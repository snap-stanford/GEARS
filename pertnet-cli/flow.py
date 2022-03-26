import pandas as pd
import numpy as np
import tqdm
from multiprocessing import Pool
from sklearn import linear_model

# -------------------------
# Flow method
# -------------------------

def get_expression_data(filename, graph_df, p_thresh, TF_only=False,
                        zero_TFs=False, TFs=None):
    # Read in expression data
    expression_data = pd.read_csv(filename, index_col=0)
    if expression_data.shape[1] <= 1:
        expression_data = pd.read_csv(filename, index_col=0, delimiter='\t')

    expression_genes = np.unique(np.array(graph_df.iloc[:,:2].values).flatten())
    expression = expression_data[expression_data['gene_name'].isin(expression_genes)]
    try:
        expression = expression[expression['p_val_adj'] < p_thresh]  # Seurat
        col = 'avg_log2FC'
    except:
        expression = expression[expression['adj.P.Val'] < p_thresh]  # limma
        col = 'logFC'

    if zero_TFs:
        missed_TFs = [t for t in TFs if t not in expression[
                            'gene_name'].values]
        missed_TFs = pd.DataFrame({'gene_name': missed_TFs, col: 0})
        expression = expression.append(missed_TFs, ignore_index=True)

    return expression


def get_expression_lambda(modelname):
    try:
        df = pd.read_csv('../Data/'+modelname, index_col=0).drop(columns=[
            'index'])
    except:
        df = pd.read_csv('../Data/'+modelname, index_col=0)
    return df.rename(columns={'avg_log2FC':'logFC'})


def get_graph(name, TF_only=False, top=None):
    # Read in TF network
    if (TF_only):
        df = pd.read_csv('/dfs/user/yhr/cell_reprogram/Data/transcription_networks/TF_only_'+name,
                         header=None)
    elif top is not None:
        df = pd.read_csv('/dfs/user/yhr/cell_reprogram/Data/transcription_networks/G_all_edges_top'
                         +str(top)+'_'+name, header=None)
    else:
        df = pd.read_csv('/dfs/user/yhr/cell_reprogram/Data/transcription_networks/G_all_edges_'+name,
                         header=None)
    return df


def add_weight(G, u, v, weight):
    try:
        G.remove_edge(u, v)
    except:
        # If the edge doesn't exist don't add a weighted version
        # return
        pass
    G.add_edge(u, v, weight=weight)


# Set diagonal elements to 1 only for TFs
def get_TFs(species):
    if species =='mouse':
         TFs = pd.read_csv('/dfs/user/yhr/cell_reprogram/Data/TF_names/mouse_tf_gene_names.txt',
                       delimiter='\t',header=None).iloc[:,0].values
    elif species=='human':
         TFs = pd.read_csv('/dfs/user/yhr/cell_reprogram/Data/TF_names/TF_names_v_1.01_human.txt',
                       delimiter='\t',header=None).iloc[:,0].values
    return np.unique(TFs, return_counts=True)[0]


def I_TF(A, expression, lamb, TFs):
    TF_idx = np.where(expression['gene_name'].isin(TFs).values)
    res = np.array(A).copy()

    # If A is a matrix
    if len(res.shape) > 1:
        if res.shape[0] == res.shape[1]:
            for i in TF_idx:
                res[i, i] += lamb
            return res

    # If A is a vetor
    for i in TF_idx:
        res[i] *= lamb
    return res


def get_model(A, y, expression, lamb=1):
    n_col = A.shape[1]
    try:
        sol = np.linalg.lstsq(A + I_TF(A, expression, lamb), I_TF(y, expression, 2))
        sol = np.array([float(i) for i in sol[0]])
    except:
        sol = np.zeros(len(expression))
    return sol

def solve_parallel2(A,B, expression, lambdas, threads):
    pool = Pool()
    for l in lambdas:
        print('Lambda: ' + str(l))
        iter = list([(A, B, expression, l)] * threads)

        results = pool.starmap(get_model, iter)

        for j in range(threads):
            expression[str(l)+'_'+str(j)] = results[j]
    return expression

def Map(F, x, args, workers):
    """
    wrapper for map()
    Spawn workers for parallel processing
    
    """
    iter_ = ((xi, args) for xi in x) 
    with Pool(workers) as pool:
        ret = pool.starmap(F, iter_)
        #ret = list(tqdm.tqdm(pool.starmap(F, iter_), total=len(x)))
    return ret


def mapper(l, args):
    A, B, positive = args
    return solve_lasso(A, B, lamb=l, positive=positive)


def solve_parallel(A,B, expression, lambdas, positive=False,
                    workers=10):
    args = (A, B, positive)
    exp_df_list = Map(mapper, lambdas, args, workers=workers)

    dict_ = {l: exp for l, exp in zip(lambdas, exp_df_list)}
    exp_df =pd.DataFrame(dict_)
    for c in expression.columns:
        exp_df[c] = expression.reset_index()[c]

    return exp_df


def solve_lasso(A,B,lamb, positive):
    print('Lambda: '+ str(lamb))
    clf = linear_model.Lasso(alpha=lamb, positive=positive)
    clf.fit(A,B)
    return clf.coef_

def solve(A,B, expression, lambdas, positive=False):
    exp_df = expression
    for l in lambdas:
        exp_df[str(l)] = solve_lasso(A,B, lamb=l, positive=positive)
    return exp_df
