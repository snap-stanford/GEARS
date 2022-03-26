from flow import get_graph, get_expression_data,\
            add_weight, get_TFs, solve,\
            solve_parallel, get_expression_lambda
import networkx as nx
import numpy as np
import pandas as pd


# Linear model for simulating linear perturbation effects
class linear_model():
    def __init__(self, graph_path, weights_path, gene_list,
                 binary=False, pos_edges=False, hops=3,
                 species='human'):
        self.TFs = get_TFs(species)
        self.gene_list = gene_list

        # Set up graph structure
        G_df = get_graph(name = graph_path, TF_only=False)
        print('Edges: '+str(len(G_df)))
        self.G = nx.from_pandas_edgelist(G_df, source=0,
                            target=1, create_using=nx.DiGraph())

        # Add edge weights
        self.read_weights = pd.read_csv(weights_path, index_col=0)
        try:
            self.read_weights = self.read_weights.set_index('TF')
        except:
            pass

        # Get adjacency matrix
        self.adj_mat = self.create_adj_mat()

        A = self.adj_mat.T
        if binary and pos_edges:
                A = np.array(A != 0).astype('float')

        # Set the diagonal elements to zero everywhere except the TFs
        np.fill_diagonal(A, 0)
        each_hop = A.copy()
        last_hop = A.copy()
        for k in range(hops-1):
            last_hop = last_hop @ each_hop
            if binary:
                A += last_hop/(k+2)
            else:
                A += last_hop
        self.A = A
    
    
    def create_adj_mat(self):
        # Create a df version of the graph for merging
        G_df = pd.DataFrame(self.G.edges(), columns=['TF', 'target'])

        # Merge it with the weights DF
        weighted_G_df = self.read_weights.merge(G_df, on=['TF', 'target'])
        for w in weighted_G_df.iterrows():
            add_weight(self.G, w[1]['TF'], w[1]['target'], w[1]['importance'])

        # Get an adjacency matrix based on the gene ordering from the DE list
        return nx.linalg.graphmatrix.adjacency_matrix(
            self.G, nodelist=self.gene_list).todense()
            

    def simulate_pert(self, pert_genes, pert_mags=None):
        """
        Returns predicted differential expression (delta) upon perturbing
        a list of genes 'pert_genes'
        """
        
        # Create perturbation vector
        pert_idx = np.where([(g in pert_genes) for g in self.gene_list])[0]
        theta = np.zeros([len(self.gene_list),1])
        
        # Set up the input vector 
        if pert_mags is None:
            pert_mags = np.ones(len(pert_genes))
        for idx, pert_mag in zip(pert_idx, pert_mags):
            theta[pert_idx] = pert_mag

        # Compute differential expression vector
        delta = np.dot(self.A, theta)
        delta = np.squeeze(np.array(delta))
        
        # Add the perturbation magnitude directly for the TF
        delta = delta + np.squeeze(theta)
        
        return delta
