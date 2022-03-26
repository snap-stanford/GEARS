import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import math
import pdb

from torch_geometric.nn import GINConv, GCNConv, GATConv, GraphConv, SGConv
from torch.nn import Sequential, Linear, ReLU, LayerNorm, PReLU
import pandas as pd

import sys

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
        
        for n in self.gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)
        
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
    
class DNN(torch.nn.Module):
    """
    DNN
    """

    def __init__(self, args):
        super(DNN, self).__init__()
        self.num_genes = args['num_genes']
        hidden_size = args['node_hidden_size']
        self.pert_w = nn.Linear(self.num_genes, hidden_size)
        self.gene_w = nn.Linear(self.num_genes, hidden_size)
        
        #self.pert_w = MLP([self.num_genes, hidden_size, hidden_size])
        #self.gene_w = MLP([self.num_genes, hidden_size, hidden_size])
        self.NN = MLP([hidden_size, self.num_genes], last_layer_act='linear')
        
    def forward(self, data):
        x, batch = data.x, data.batch
        num_graphs = len(data.batch.unique())
        
        gene_base = x[:, 0].reshape(num_graphs, self.num_genes)
        gene_emb = self.gene_w(gene_base)
            
        pert = x[:, 1].reshape(num_graphs, self.num_genes)
        pert_emb = self.pert_w(pert)
        base_emb = pert_emb+gene_emb
        
        out = self.NN(base_emb) + x[:, 0].reshape(*data.y.shape)
        #out = self.NN(base_emb)
        
        return out
    
class GNN_Diffusion(torch.nn.Module):
    """
    GNN_Diffusion
    """

    def __init__(self, args):
        super(GNN, self).__init__()

        self.num_genes = args['num_genes']
        hidden_size = args['node_hidden_size']
        self.num_layers = args['num_of_gnn_layers']
        self.network_type = args['network_type']
        
        self.args = args        

        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)

        # gene basal embedding - encoding gene expression
        self.gene_basal_w = nn.Linear(1, hidden_size)
        
        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])
        
        if self.model_backend == 'GAT':
            self.layers_emb_pos = GATConv(hidden_size, hidden_size, heads = 1)
        elif self.model_backend == 'GCN':
            self.layers_emb_pos = GCNConv(hidden_size, hidden_size)
        elif self.model_backend == 'SGC':
            self.layers_emb_pos = SGConv(hidden_size, hidden_size, 1)

        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
           

    def forward(self, data):
        x, batch = data.x, data.batch
        num_graphs = len(data.batch.unique())
        
        ## add the gene expression positional embedding
        gene_base = x[:, 0].reshape(-1,1)
        gene_emb = self.gene_basal_w(gene_base)
            
        pert = x[:, 1].reshape(-1,1)
        pert_emb = self.pert_w(pert)
        base_emb = pert_emb+gene_emb
        
        for idx in range(self.num_layers):
            base_emb = self.layers_emb_pos(base_emb, self.G_coexpress, self.G_coexpress_weight)
            if idx < len(self.layers_emb_pos) - 1:
                base_emb = base_emb.relu()
        
        #out = self.recovery_w(base_emb) + x[:, 0].reshape(-1,1)
        out = self.recovery_w(base_emb) + x[:, 0].reshape(-1,1)
        out = torch.split(torch.flatten(out), self.num_genes)

        return torch.stack(out)
    
class GNN(torch.nn.Module):
    """
    GNN
    """

    def __init__(self, args):
        super(GNN, self).__init__()

        self.num_genes = args['num_genes']
        hidden_size = args['node_hidden_size']
        self.num_layers = args['num_of_gnn_layers']
        self.network_type = args['network_type']
        self.num_layers_gene_pos = args['gene_sim_pos_emb_num_layers']
        self.model_backend = args['model_backend']
        
        self.args = args        

        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)

        # gene basal embedding - encoding gene expression
        self.gene_basal_w = nn.Linear(1, hidden_size)
        
        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

        #self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.emb_pos = nn.Parameter(torch.Tensor(self.num_genes, hidden_size), requires_grad = False)
        nn.init.xavier_uniform_(self.emb_pos)
        
        self.layers_emb_pos = torch.nn.ModuleList()
        
        for i in range(1, self.num_layers_gene_pos + 1):
            if self.model_backend == 'GAT':
                self.layers_emb_pos.append(GATConv(hidden_size, hidden_size, heads = 1))
            elif self.model_backend == 'GCN':
                self.layers_emb_pos.append(GCNConv(hidden_size, hidden_size))
            elif self.model_backend == 'SGC':
                self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))
        
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
           

    def forward(self, data):
        x, batch = data.x, data.batch
        num_graphs = len(data.batch.unique())
        
        ## add the gene expression positional embedding
        #pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
        
        gene_base = x[:, 0].reshape(-1,1)
        gene_emb = self.gene_basal_w(gene_base) + self.emb_pos.repeat(num_graphs, 1).to(self.args['device'])
            
        pert = x[:, 1].reshape(-1,1)
        pert_emb = self.pert_w(pert)
        base_emb = pert_emb+gene_emb
        
        for idx, layer in enumerate(self.layers_emb_pos):
            base_emb = layer(base_emb, self.G_coexpress, self.G_coexpress_weight)
            if idx < len(self.layers_emb_pos) - 1:
                base_emb = base_emb.relu()
        
        #out = self.recovery_w(base_emb) + x[:, 0].reshape(-1,1)
        out = self.recovery_w(base_emb) + x[:, 0].reshape(-1,1)
        out = torch.split(torch.flatten(out), self.num_genes)

        return torch.stack(out)         
        

class No_Perturb(torch.nn.Module):
    """
    No Perturbation
    """

    def __init__(self):
        super(No_Perturb, self).__init__()        

    def forward(self, data):
        
        x = data.x
        x = x[:, 0].reshape(*data.y.shape)
        
        return x, None

class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        if self.activation == "ReLU":
            return self.relu(self.network(x))
        else:
            return self.network(x)

class Set_Self_Attention(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(Set_Self_Attention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O        
        

class PertNet(torch.nn.Module):
    """
    PertNet
    """

    def __init__(self, args):
        super(PertNet, self).__init__()

        self.num_genes = args['num_genes']
        hidden_size = args['node_hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_of_gnn_layers']
        self.network_type = args['network_type']
        self.indv_out_layer = args['indv_out_layer']
        self.indv_out_hidden_size = args['indv_out_hidden_size']
        self.num_mlp_layers = args['num_mlp_layers']
        self.indv_out_layer_uncertainty = args['indv_out_layer_uncertainty']
        self.add_gene_expression = args['add_gene_expression']
        self.add_gene_expression_back = args['add_gene_expression_back']
        self.expression_concat = args['expression_concat']
        self.expression_no_bn = args['expression_no_bn']
        self.post_coexpress = args['post_coexpress']
        # gene structure
        self.gene_sim_pos_emb = args['gene_sim_pos_emb']
        self.num_layers_gene_pos = args['gene_sim_pos_emb_num_layers']
        self.model_backend = args['model_backend']
        self.add_gene_expression_gene_specific = args['add_gene_expression_gene_specific']
        
        self.args = args        
        # lambda for aggregation between global perturbation emb + gene embedding
        self.pert_emb_lambda = args['pert_emb_lambda']
        
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)

        # gene basal embedding - encoding gene expression
        if not self.expression_concat:
            self.gene_basal_w = nn.Linear(1, hidden_size)
        
        # gene/globel perturbation embedding dictionary lookup            
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        
        # transformation layer
        
        if args['emb_trans_mlp']:
            self.emb_trans = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        else:
            self.emb_trans = nn.ReLU()
            
        if args['pert_base_trans_w_mlp']:
            self.pert_base_trans_w = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        else:
            self.pert_base_trans_w = nn.ReLU()
            
        if self.expression_concat:
            self.gene_base_trans_w = MLP([hidden_size + 1, hidden_size, hidden_size], last_layer_act='linear')
        else:
            if args['gene_base_trans_w_mlp']:
                self.gene_base_trans_w = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
            else:
                self.gene_base_trans_w = nn.ReLU()
           
        if args['transform_mlp']:
            self.transform = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        else:
            self.transform = nn.ReLU()
        
        self.cross_gene_MLP = args['cross_gene_MLP']
        self.cross_gene_decoder = args['cross_gene_decoder']
        self.de_drop = args['de_drop']
        self.add_gene_expression_before_cross_gene = args['add_gene_expression_before_cross_gene']
        
        self.mean_control = args['mean_control']
        if self.mean_control:
            self.ctrl = args['ctrl']
            
        if self.gene_sim_pos_emb:
            self.G_coexpress = args['G_coexpress'].to(args['device'])
            self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])
            
            if args['emb_trans_v2_mlp']:
                self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
            else:
                self.emb_trans_v2 = nn.ReLU()

            self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
            self.layers_emb_pos = torch.nn.ModuleList()
            for i in range(1, self.num_layers_gene_pos + 1):
                if self.model_backend == 'GAT':
                    self.layers_emb_pos.append(GATConv(hidden_size, hidden_size, heads = 1))
                elif self.model_backend == 'GCN':
                    self.layers_emb_pos.append(GCNConv(hidden_size, hidden_size))
                elif self.model_backend == 'SGC':
                    self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))


        if self.indv_out_layer:
            
            if self.cross_gene_MLP:
                self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')
                #self.recovery_w = MLP([hidden_size, hidden_size], last_layer_act='ReLU')
                self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                                       hidden_size, 1))
                self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
                self.act = nn.ReLU()
                
                # Cross gene state encoder
                self.cross_gene_state = MLP([self.num_genes, hidden_size,
                                             hidden_size])

                # First layer parameters
                if self.add_gene_expression_gene_specific:
                    self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                                       hidden_size+2))
                else:
                    self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                                       hidden_size+1))
                    
                self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
                nn.init.xavier_normal_(self.indv_w2)
                nn.init.xavier_normal_(self.indv_b2)
                
                nn.init.xavier_normal_(self.indv_w1)
                nn.init.xavier_normal_(self.indv_b1)
            elif self.cross_gene_decoder in ['mlp', 'skip-connect', 'skip-connect-mlp']:
                self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size, self.indv_out_hidden_size], last_layer_act='linear')
                self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                                       self.indv_out_hidden_size, 1))
                self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
                self.act = nn.ReLU()
                
                self.cg_mlp = args['cg_mlp']
                if self.cg_mlp == 'baseline':                
                    self.cross_gene_state = MLP([self.num_genes, hidden_size*2,
                                             hidden_size*2, self.num_genes])
                elif self.cg_mlp == 'small':
                    self.cross_gene_state = MLP([self.num_genes, hidden_size, self.num_genes])
                    
                elif self.cg_mlp == 'deep':
                    self.cross_gene_state = MLP([self.num_genes, hidden_size, hidden_size, hidden_size, hidden_size, self.num_genes])                
                elif self.cg_mlp == 'wide':
                    self.cross_gene_state = MLP([self.num_genes, hidden_size * 4, hidden_size * 4, self.num_genes])             
                
                if self.cross_gene_decoder == 'skip-connect-mlp':
                    self.cross_gene_state_2 = MLP([self.num_genes,
                                             hidden_size, self.num_genes])
                
                
            else:
                self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size, self.indv_out_hidden_size], last_layer_act='linear')
                self.indv_w1 = nn.Parameter(torch.rand(self.num_genes, self.indv_out_hidden_size, 1))
                self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))

            nn.init.kaiming_uniform_(self.indv_w1, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.indv_w1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.indv_b1, -bound, bound)

        else:
            self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')

        if self.args['func_readout']:
            self.func_w1 = MLP([hidden_size, hidden_size * 2, hidden_size, 1],
                last_layer_act='linear')
            self.func_w2 = MLP([self.num_genes, self.num_genes * 2, self.num_genes, 1],
                last_layer_act='linear')
        
        ### perturbation embedding similarity
        if self.network_type == 'all':
            self.sim_layers_networks = {}
            self.G_sim = {}
            self.G_sim_weight = {}
            
            for i in ['string_ppi', 'co-expression_train', 'gene_ontology']:
                self.G_sim[i] = args['G_sim_' + i].to(args['device'])
                self.G_sim_weight[i] = args['G_sim_weight_' + i].to(args['device'])
                
                sim_layers = torch.nn.ModuleList()
                for l in range(1, self.num_layers + 1):
                    sim_layers.append(SGConv(hidden_size, hidden_size, 1))
                self.sim_layers_networks[i] = sim_layers
            self.sim_layers_networks = nn.ModuleDict(self.sim_layers_networks)
            
        else:   
            # perturbation similarity network
            self.G_sim = args['G_sim'].to(args['device'])
            self.G_sim_weight = args['G_sim_weight'].to(args['device'])

            self.sim_layers = torch.nn.ModuleList()
            for i in range(1, self.num_layers + 1):
                self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))

        # batchnorms
        self.bn_pert_emb = nn.BatchNorm1d(hidden_size)
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        self.bn_final = nn.BatchNorm1d(hidden_size)
        self.bn_gene_base = nn.BatchNorm1d(hidden_size)
        self.bn_gene_base_trans = nn.BatchNorm1d(hidden_size)
            
        self.bn_post_gnn = nn.BatchNorm1d(hidden_size)
        self.bn_base_emb = nn.BatchNorm1d(hidden_size)
        
        self.mlp_pert_fuse=args['mlp_pert_fuse']
        self.set_self_attention = args['set_self_attention']
        
        if self.mlp_pert_fuse:
            if self.set_self_attention:
                self.set_self_attention_num_head = args['set_self_attention_num_head']
                self.set_self_attention_layernorm = args['set_self_attention_layernorm']
                self.set_self_attention_agg = args['set_self_attention_agg']
                self.set_self_attention_post_mlp = args['set_self_attention_post_mlp']
                self.pert_fuse_linear_to_mlp = args['pert_fuse_linear_to_mlp']

                self.pert_fuse = Set_Self_Attention(hidden_size, hidden_size, hidden_size, self.set_self_attention_num_head, self.set_self_attention_layernorm)

                if self.pert_fuse_linear_to_mlp:
                    self.pert_fuse_linear = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='ReLU')
                else:
                    self.pert_fuse_linear = nn.Linear(hidden_size, 1)

                if self.set_self_attention_post_mlp:
                    self.pert_fuse_mlp = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='ReLU')
            else:
                if args['pert_fuse_mlp']:
                    self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
                else:
                    self.pert_fuse = nn.ReLU()

        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        
        if self.indv_out_layer_uncertainty:	
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, self.indv_out_hidden_size], last_layer_act='linear')
            self.indv_w1_unc = nn.Parameter(torch.rand(self.num_genes, self.indv_out_hidden_size, 1))
            self.indv_b1_unc = nn.Parameter(torch.rand(self.num_genes, 1))	
            nn.init.kaiming_uniform_(self.indv_w1_unc, a=math.sqrt(5))	
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.indv_w1_unc)	
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0	
            nn.init.uniform_(self.indv_b1_unc, -bound, bound)	

        else:	
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')


    def forward(self, data, return_att = False):
        x, batch = data.x, data.batch
        num_graphs = len(data.batch.unique())
        
        if self.mean_control:
            x[:, 0] = torch.FloatTensor(np.tile(self.ctrl, num_graphs)).to(self.args['device'])  
            
        ## get base gene embeddings
        emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))        
        emb = self.bn_emb(emb)
        base_emb = self.emb_trans(emb)
        #base_emb = self.bn_base_emb(base_emb)
        
        if not self.add_gene_expression_back:
            if self.add_gene_expression:	
                ## add the gene expression positional embedding	
                gene_base = x[:, 0].reshape(-1,1)	
                
                if self.expression_concat:
                    combined = torch.hstack((gene_base, base_emb))
                else:
                    gene_emb = self.gene_basal_w(gene_base)	
                    combined = gene_emb+base_emb

                if self.expression_no_bn:    
                    base_emb = self.gene_base_trans_w(combined)	
                else:
                    base_emb = self.gene_base_trans_w(combined)	
                    base_emb = self.bn_gene_base(base_emb)
        
        
        if self.gene_sim_pos_emb:
            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            #base_emb = self.emb_trans(torch.cat((emb, pos_emb), axis = 1))

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)


        ## get perturbation index and embeddings
        pert = x[:, 1].reshape(-1,1)
        pert_index = torch.where(pert.reshape(num_graphs, int(x.shape[0]/num_graphs)) == 1)
        pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_genes))).to(self.args['device']))
        #pert_global_emb = self.bn_pert_emb(pert_global_emb)
        
        if self.network_type == 'all':
            pert_global_emb_all = 0
            for i in ['string_ppi', 'co-expression_train', 'gene_ontology']:
                sim_layers = self.sim_layers_networks[i]
                
                for idx, layer in enumerate(sim_layers):
                    pert_global_emb = layer(pert_global_emb, self.G_sim[i], self.G_sim_weight[i])
                    if idx < self.num_layers - 1:
                        pert_global_emb = pert_global_emb.relu()
                
                pert_global_emb_all += pert_global_emb
            pert_global_emb = pert_global_emb_all               
        else:
            ## augment global perturbation embedding with GNN
            for idx, layer in enumerate(self.sim_layers):
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()
            #pert_global_emb = self.bn_post_gnn(pert_global_emb)

        ## add global perturbation embedding to each gene in each cell in the batch
        base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

        
        if self.mlp_pert_fuse:
            if self.set_self_attention:
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = torch.stack((pert_track[j.item()], pert_global_emb[pert_index[1][i]]))
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]
                
                if len(list(pert_track.values())) > 0:
                    pert_track_ssa = {}

                    for j, emb in pert_track.items():
                        if len(emb.shape) == 1:
                            emb = emb.unsqueeze(0).unsqueeze(0)
                        else:
                            emb = emb.unsqueeze(0)

                        pert_track_ssa[j] = self.pert_fuse(emb, emb)[0]

                    agg_track = {}
                    if self.set_self_attention_agg == 'sum':
                        for j, emb in pert_track_ssa.items():
                            pert_fuse_emb = torch.sum(emb, axis = 0)
                            agg_track[j] = pert_fuse_emb
                    else:
                        # batch calculation of attention
                        att = self.pert_fuse_linear(torch.vstack(list(pert_track_ssa.values())))
                        
                        if self.set_self_attention_agg == 'weight_ori_emb':
                            pert_track_ = pert_track
                        elif self.set_self_attention_agg == 'weight_post_emb':
                            pert_track_ = pert_track_ssa

                        count = 0
                        for j, emb in pert_track_.items():
                            if len(emb.shape) == 1:
                                sh = 1
                            else:
                                sh = emb.shape[0]
                            pert_fuse_emb = torch.sum(att[count:count+sh] * emb, axis = 0)
                            count += sh
                            agg_track[j] = pert_fuse_emb
                            
                            
                    if self.set_self_attention_post_mlp:                        
                        emb_total = self.pert_fuse_mlp(torch.stack(list(agg_track.values())))
                        for idx, j in enumerate(agg_track.keys()):
                            base_emb[j] += emb_total[idx]
                    else:
                        for j, emb in agg_track.items():
                            base_emb[j] += emb                     
                    
            else:
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j in pert_track:
                        pert_track[j] += pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] += emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)

        else:
            for i, j in enumerate(pert_index[0]):
                lambda_i = self.pert_emb_lambda
                base_emb[j] += lambda_i * pert_global_emb[pert_index[1][i]]
            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
        
        if self.add_gene_expression_back:	
            ## add the gene expression positional embedding	
            gene_base = x[:, 0].reshape(-1,1)	
                
            if self.expression_concat:
                combined = torch.hstack((gene_base, base_emb))
            else:
                gene_emb = self.gene_basal_w(gene_base)	
                combined = gene_emb+base_emb

            if self.expression_no_bn:    
                base_emb = self.gene_base_trans_w(combined)	
            else:
                base_emb = self.gene_base_trans_w(combined)	
                base_emb = self.bn_gene_base(base_emb)
        
        
        ## add the perturbation positional embedding
        pert_emb = self.pert_w(pert)
        combined = pert_emb+base_emb
        combined = self.bn_pert_base_trans(combined)
        base_emb = self.pert_base_trans_w(combined)
        base_emb = self.bn_pert_base(base_emb)
        
        ## apply the first MLP
        base_emb = self.transform(base_emb)
        func_out = None
        #base_emb = self.bn_final(base_emb)
        
        if self.indv_out_layer:
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis = 2)
            out = w + self.indv_b1
            
            if self.add_gene_expression_before_cross_gene:
                out = out.reshape(num_graphs * self.num_genes, -1) + x[:, 0].reshape(-1,1)
                out = out.reshape(num_graphs, self.num_genes, -1)
            
            if self.cross_gene_decoder in ['mlp', 'skip-connect', 'skip-connect-mlp']:
                output = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
                if self.cross_gene_decoder == 'skip-connect':
                    out = output + out.reshape(num_graphs, self.num_genes, -1).squeeze(2)
                elif self.cross_gene_decoder == 'skip-connect-mlp':
                    out = self.cross_gene_state_2(output + out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
                else:
                    out = output
                                 
            elif self.cross_gene_MLP:
                # Compute global gene embedding
                cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))

                # repeat embedding num_genes times
                cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

                # stack it under out
                cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])
                cross_gene_out = torch.cat([out, cross_gene_embed], 2)
                
                if self.add_gene_expression_gene_specific:
                    cross_gene_out = torch.cat([x[:, 0].reshape(num_graphs, self.num_genes, -1), cross_gene_out], 2)

                # First pass through MLP
                cross_gene_out = cross_gene_out * self.indv_w2
                cross_gene_out = torch.sum(cross_gene_out, axis=2)
                out = cross_gene_out + self.indv_b2
                
            if self.de_drop:
                out = out.reshape(num_graphs * self.num_genes, -1)
            else:
                out = out.reshape(num_graphs * self.num_genes, -1) + x[:, 0].reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)
            
        else:
            ## apply the final MLP to predict delta only and then add back the x. 
            out = self.recovery_w(base_emb) + x[:, 0].reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)

        if self.args['func_readout']:
            func_out = torch.flatten(self.func_w2(torch.stack(out)))

        ## uncertainty head
        if self.uncertainty:
            
            if self.indv_out_layer_uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = out_logvar.reshape(num_graphs, self.num_genes, -1)
                out_logvar = out_logvar.unsqueeze(-1) * self.indv_w1_unc
                w = torch.sum(out_logvar, axis = 2)
                out_logvar = w + self.indv_b1_unc
                out_logvar = out_logvar.reshape(num_graphs * self.num_genes, -1) + x[:, 0].reshape(-1,1)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)	
                return torch.stack(out), torch.stack(out_logvar), func_out
            else:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar), func_out
        
        if return_att:
            return torch.stack(out), func_out, att
        else:
            return torch.stack(out), func_out
        
