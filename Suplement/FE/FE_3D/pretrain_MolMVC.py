#coding=utf-8
import os
import torch
from torch import nn
from Suplement.FE.FE_3D.method import GNN, SchNet
import numpy as np
from torch_geometric.nn import global_mean_pool
from Suplement.FE.FE_3D.method.model_helper import Encoder_1d, Embeddings
torch.set_printoptions(precision=None, threshold=10000, edgeitems=None, linewidth=None, profile=None)
torch.set_printoptions(threshold=np.inf)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

device_ids = [0, 1]
device = "cuda" if torch.cuda.is_available() else "cpu"
molecule_readout_func = global_mean_pool

# Transformer encoder for ESPF
class transformer_1d(nn.Sequential):
    def __init__(self):
        super(transformer_1d, self).__init__()
        input_dim_drug = 2586 + 1 # last for mask
        transformer_emb_size_drug = 128
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        self.emb = Embeddings(input_dim_drug,
                         transformer_emb_size_drug,
                         50,
                         transformer_dropout_rate)

        self.encoder = Encoder_1d(transformer_n_layer_drug,
                                         transformer_emb_size_drug,
                                         transformer_intermediate_size_drug,
                                         transformer_num_attention_heads_drug,
                                         transformer_attention_probs_dropout,
                                         transformer_hidden_dropout_rate)
    def forward(self, emb, mask):
        e = emb.long().to(device)
        e_mask = mask.long().to(device)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers, attention_scores = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers, attention_scores

class EmbeddingModel(nn.Module):
    def __init__(self):
        super(EmbeddingModel, self).__init__()
        self.model_2d = GNN()
        self.model_3d = SchNet()
        self.model_1d = transformer_1d()

    def forward(self, batch_nol):
        batch_nol = batch_nol.to(device)
        preprocessor = PreprocessBatch()
        preprocessor.process(batch_nol)
        
        # 1D embedding from ESPF
        smiles_emb = torch.from_numpy(np.asarray(batch_nol.smiles)).to(device)
        smi_mask = torch.from_numpy(np.asarray(batch_nol.mask)).to(device)
        emb_1d, _ = self.model_1d(smiles_emb, smi_mask)
        emb_1d_low = emb_1d[3]
        emb_1d_high = emb_1d[7]
        emb_1d_low = torch.sum(emb_1d_low * smi_mask.unsqueeze(2), dim=1) / torch.sum(smi_mask, dim=1, keepdim=True)
        emb_1d_high = torch.sum(emb_1d_high * smi_mask.unsqueeze(2), dim=1) / torch.sum(smi_mask, dim=1, keepdim=True)
        
        # 2D embedding
        emb_2d_low, emb_2d_res, _ = self.model_2d(batch_nol.x, batch_nol.edge_index, batch_nol.edge_attr)
        emb_2d = molecule_readout_func(emb_2d_low + emb_2d_res, batch_nol.batch)

        # 3D embedding
        emb_3d_low, emb_3d_res, _ = self.model_3d(batch_nol.x[:, 0], batch_nol.pos, batch_nol.batch)

        if emb_3d_low.shape[0] == 0 or emb_3d_res.shape[0] == 0:
            print("3D not valid")
            emb_3d = torch.zeros_like(emb_1d_low)  
        else:
            emb_3d = molecule_readout_func(emb_3d_low + emb_3d_res, batch_nol.batch)

        return emb_1d_low, emb_1d_high, emb_2d, emb_3d

class PreprocessBatch:
    def process(self, batch):
        pos = batch.pos
        batch_node = batch.batch.tolist()
        pos_mean = global_mean_pool(pos, batch.batch)

        flag = 0
        num = []
        for x in range(batch_node[len(batch_node)-1] + 1):
            flag = batch_node.count(x)
            num.append(flag)
        pos = pos - torch.repeat_interleave(pos_mean, torch.tensor(num).to(device), dim=0)
        batch.pos = pos
