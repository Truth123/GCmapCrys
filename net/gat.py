import torch.nn as nn
import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


class GCmapCrys(nn.Module):
    def __init__(self, conf):
        super(GCmapCrys, self).__init__()
        ## node_feature
        self.aa_emb_layer = nn.Embedding(
            num_embeddings= 21,
            embedding_dim= conf["embedding_aa_dim"]
        )
        self.ss_emb_layer = nn.Embedding(
            num_embeddings= 8,
            embedding_dim= conf["embedding_ss_dim"]
        )

        in_n = 590 + conf['embedding_aa_dim'] + conf["embedding_ss_dim"] # (PSSM:20; length:1; Gravy:1; pI:1; AAindex:566; RSA:1) + (AAC; SS)
        in_e = 1

        fc_dim = conf['fc_dim']
        fc_dropout = conf['fc_dropout']

        layers = zip(
            eval(conf['mp']['node_feats']),
            eval(conf['mp']['edge_feats']),
            eval(conf['mp']['multi_heads']),
            eval(conf['mp']['gat_dropout']),
        )
        self.all_mp = nn.ModuleList()
        for out_n, out_e, n_heads, gat_dropout in layers:
            gat_mp = nn.ModuleList()
            update_edge = nn.Sequential(
                nn.Linear(in_e + in_n + in_n, out_e),
                nn.BatchNorm1d(out_e),
                nn.ReLU()
            )
            gat = GATConv(in_n, out_n, n_heads, concat=True, dropout=gat_dropout, fill_value="mean", edge_dim=out_e)
            batch_normal = nn.BatchNorm1d(out_n*n_heads)
            gat_mp.append(update_edge)
            gat_mp.append(gat)
            gat_mp.append(batch_normal)
            self.all_mp.append(gat_mp)
            in_e, in_n = out_e, out_n*n_heads
        
        self.gcmap_dense = nn.Sequential(
            nn.Linear(in_n, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout, inplace=True)
        )
        self.out_layer = nn.Linear(fc_dim, 1)
        self.apply(weight_init)
    
    
    def forward(self, graph):
        x_feature = graph.x                            ## (L, 590)
        x_emb = graph.x_emb
        ss8 = graph.ss8                                ## (L, )
        batch = graph.batch                            ## (L, )
        edge_attr = graph.edge_attr                    ## (E, 1)
        edge_index = graph.edge_index                  ## (2, E)

        x_emb = self.aa_emb_layer(x_emb)                  
        ss8 = self.ss_emb_layer(ss8)                  
        x = torch.cat((x_emb, ss8, x_feature),dim=1) 
        
        u = []
        for update_edge, gat, batch_normal in self.all_mp:
            x_dst = x[edge_index[1]]
            x_src = x[edge_index[0]]
            edge_attr = update_edge(torch.cat((edge_attr, x_dst, x_src),dim=-1))
            x = gat(x, edge_index, edge_attr)
            x = batch_normal(x)
            x = F.relu(x)
        u = gap(x, batch)
        u = self.gcmap_dense(u)
        u = self.out_layer(u)
        return u