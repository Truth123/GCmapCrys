from genericpath import isdir
import torch
from torch.utils.data import Dataset,Sampler,WeightedRandomSampler
from torch.utils.data import DataLoader as default_Dataloader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as gemetric_Dataloader
import util
import os
import random
import h5py
import numpy as np

projectPath = os.path.dirname(os.path.abspath(__file__))


class MyDataset(Dataset):
    def __init__(self, 
                input, 
                feature_dir, 
                use_ccmap=True,
                limit_ccmap_length=False) -> None:
        super(MyDataset, self).__init__()
        if os.path.isdir(input):
            self.x = util.read_fasta(os.path.join(input, 'sequence.fasta'))
            self.y = util.read_fasta(os.path.join(input, 'label.txt'))
        else:
            self.x = util.read_fasta(input)
            self.y = {}

        self.ids = []
        if not use_ccmap or limit_ccmap_length:
            for id in self.x.keys():
                s = self.x[id]
                if len(s) <= 800:
                    self.ids.append(id)
        else:
            self.ids = list(self.x.keys())
        random.shuffle(self.ids)
        self.feature_dir = feature_dir
        self.use_ccmap = use_ccmap

    
    def __getitem__(self, index: int):
        id = self.ids[index]
        seq = str(self.x[id])
        y = int(self.y[id])
        y = 1 if y == 1 else 0
        feature_file = os.path.join(self.feature_dir, id + '.h5')
        f = h5py.File(feature_file, 'r')
        ## f.keys(): ['AA', 'AAindex', 'Gravy', 'PSSM', 'RSA', 'SS', 'edge_attr', 'edge_index', 'id', 'log_length', 'pI', 'seq']
        assert seq == str(f["seq"][()], 'utf-8') and id == str(f["id"][()], 'utf-8')
        num_nodes = len(seq)
        aa = torch.tensor(f["AA"][()], dtype=torch.long)
        log_length = f["log_length"][()]

        ss8 = torch.tensor(f["SS"][()], dtype=torch.long)
        x_emb = aa * 8 + ss8
        emb_dim = 168
        
        x = torch.zeros((num_nodes, 1), dtype=torch.float32).fill_(log_length)

        x_aaindex = torch.tensor(f["AAindex"][()], dtype=torch.float32)
        x = torch.cat((x, x_aaindex.T), dim=-1)

        x_gravy = torch.zeros((num_nodes, 1), dtype=torch.float32).fill_(f["Gravy"][()])
        x = torch.cat((x, x_gravy), dim=-1)

        x_pI = torch.zeros((num_nodes, 1), dtype=torch.float32).fill_(f["pI"][()])
        x = torch.cat((x, x_pI), dim=-1)

        x_pssm = torch.tensor(f["PSSM"][()], dtype=torch.float32)
        x = torch.cat((x, x_pssm), dim=-1)

        x_rsa = torch.tensor(f["RSA"][()], dtype=torch.float32).unsqueeze(1)
        x = torch.cat((x, x_rsa), dim=-1)

        if self.use_ccmap:
            edge_attr = torch.tensor(f["edge_attr"][()], dtype=torch.float32)
            edge_index = torch.tensor(f["edge_index"][()], dtype=torch.long)
            f.close()
            return Data.from_dict({
                "id": id,
                "x": x,
                "x_emb": x_emb,
                "edge_attr": edge_attr,
                "edge_index": edge_index,
                "y": y
            })
        else:
            f.close()
            tmp = torch.zeros((800 - num_nodes, x.shape[1]), dtype=torch.float32)
            x = torch.cat((x, tmp), dim=0)  ## (800, n)
            x = x.permute(1,0)              ## (n, 800)
            tmp_emb = torch.zeros((800 - num_nodes), dtype=torch.long).fill_(emb_dim)
            x_emb = torch.cat((x_emb, tmp_emb), dim=0) ## (800)
            return x, x_emb, y, id


    def __len__(self) -> int:
        return len(self.ids)
