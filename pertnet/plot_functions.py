import matplotlib.pyplot as plt
import numpy as np

def get_de(res, query):
    """
    Given a query perturbation and model output file,
    return predicted and true post-perturbation expression
    """
    query_idx = np.where(res['pert_cat'] == query)[0]
    de = {"pred_de": res['pred_de'][query_idx],
           "truth_de": res['truth_de'][query_idx]}
    return de

def get_de_ctrl(pert, adata):
    """
    Get ctrl expression for DE genes for a given perturbation
    """
    mean_ctrl_exp = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()
    de_genes = get_covar_genes(pert, adata)
    return mean_ctrl_exp[de_genes]

def get_covar_genes(p, adata):
    """
    Get genes that are differentially expressed
    """
    gene_name_dict = adata.var.loc[:,'gene_name'].to_dict()
    pert_name = 'A549_'+p+'_1+1'
    de_genes = adata.uns['rank_genes_groups_cov'][pert_name]
    return de_genes

def create_boxplot(res, adata, query, genes=None):
    """
    Create a boxplot showing true, predicted and control expression
    for a given perturbation
    """
    
    plt.figure(figsize=[10,3])
    plt.title(query)
    pert_de_res = get_de(res, query)['pred_de']
    truth_de_res = get_de(res, query)['truth_de']
    plt.boxplot(truth_de_res, showfliers=False,
                medianprops = dict(linewidth=0))
    ctrl_means = get_de_ctrl(query, adata).values

    for i in range(pert_de_res.shape[1]):
        _ = plt.scatter(i+1, np.mean(pert_de_res[:,i]), color='red')
        _ = plt.scatter(i+1, ctrl_means[i], color='forestgreen', marker='*')
    
    ax = plt.gca()
    if genes is not None:
        ax.xaxis.set_ticklabels(genes)
    else:
        ax.xaxis.set_ticklabels(['G1','G2','G3','G4','G5','G6','G7','G8','G9', 'G10',
                                'G11','G12','G13','G14','G15','G16','G17','G18','G19', 'G20'])
