import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
torch.cuda.manual_seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-15

# dense_diff_pool 函数保持不变，确保 GPU 兼容
def dense_diff_pool(x, z, adj, s, mask=None):
    return out_x, out_z, out_adj, link_loss, ent_loss

# Attention 类保持不变
class Attention(nn.Module):
        return attention.to(device)

class NEGAN(nn.Module):
    def __init__(self, layer, feature_size, top_k):
        self.to(device)

    def _initialize_weights(self):
                nn.init.constant_(m.bias, 0)

    def forward(self, X, A,Z,adj):
        return Y, link_loss, ent_loss,final_A,S_list
class CAGNN(nn.Module):
        return x, link_loss,ent_loss,final_A,S_list