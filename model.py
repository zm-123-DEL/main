import torch
import torch.nn as nn
torch.cuda.manual_seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-15
# dense_diff_pool 函数保持不变，确保 GPU 兼容
def dense_diff_pool(x, z, adj, s, mask=None):
    x = x.unsqueeze(0) if x.dim() == 2 else x
    z = z.unsqueeze(0) if x.dim() == 2 else z
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()
    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype).to(device)
        x, s, z = x * mask, s * mask, z * mask

    out_x = torch.matmul(s.transpose(1, 2), x)
    out_z = torch.matmul(s.transpose(1, 2), z)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()

    return out_x, out_z, out_adj, link_loss, ent_loss
class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X,A,Z):
        X = X.permute(0, 2, 1)
        Q = self.conv1(X).permute(0, 2, 1)
        Z = Z.permute(0, 2, 1)
        K = self.conv2(Z)
        attention = self.softmax(torch.matmul(Q, K) / (Q.size()[-1] ** 0.5))*A
        return attention.to(device)

class NEGAN(nn.Module):
    def __init__(self, layer, feature_size, top_k):
        super(NEGAN, self).__init__()
        self.top_k = top_k
        self.layer = layer
        self.relu = nn.ReLU()
        self.atten = nn.ModuleList([Attention(feature_size, feature_size) for i in range(layer)])
        self.w_s = nn.ParameterList(
            [nn.Parameter(torch.randn(feature_size, int(feature_size * (self.top_k ** (i + 1))), dtype=torch.float32))
             for i in range(layer)])
        self.w_z = nn.ParameterList([nn.Parameter(
            torch.randn(feature_size * int(feature_size * (self.top_k ** (i + 1))),
                        feature_size * int(feature_size * (self.top_k ** (i + 1))), dtype=torch.float32)) for i in
                                     range(layer)])
        self.norm_n = nn.ModuleList([nn.BatchNorm1d(feature_size) for i in range(layer)])
        self.norm_e = nn.ModuleList([nn.BatchNorm1d(feature_size) for i in range(layer)])
        self.softmax = nn.Softmax()
        self.line_n = nn.ModuleList(
            [nn.Sequential(nn.Linear(200, 128), nn.ReLU(), nn.BatchNorm1d(128)) for i in range(layer + 1)])
        self.line_e = nn.ModuleList(
            [nn.Sequential(nn.Linear(200, 128), nn.ReLU(), nn.BatchNorm1d(128)) for i in range(layer + 1)])
        self.clase = nn.Sequential(nn.Linear(128 * 2 * (self.layer + 1), 1024), nn.Dropout(0.2), nn.ReLU(),
                                   nn.Linear(1024, 128))
        self.gcn = nn.Sequential(nn.Linear(feature_size, feature_size), nn.LeakyReLU(negative_slope=0.2))
        self._initialize_weights()
        self.to(device)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, X, A,Z,adj):
        link_loss = torch.tensor(0.0, device=device)
        ent_loss = torch.tensor(0.0, device=device)
        bz, roi_num, feature_dim = X.size()
        X_n = self.line_n[0](Z.reshape((bz * roi_num, -1)))
        XX = X_n.reshape((bz, roi_num, -1))
        Z_n = self.line_e[0](Z.reshape((bz * roi_num, -1)))
        ZZ = Z_n.reshape((bz, roi_num, -1))
        S_list = []
        final_A = None
        for i in range(self.layer):
            A = self.atten[i](X, A, Z)
            X1 = torch.matmul(A, X)
            S = torch.matmul(X1, self.w_s[i].to(device))
            S_list.append(S)
            X, Z, A, linkloss, entloss = dense_diff_pool(X, Z, A, S)
            final_A = A
            Z1 = Z.reshape((bz, -1)).to(device)
            Z2 = adj @ Z1
            Z3 = Z2.reshape((bz, Z.size()[1], Z.size()[2]))
            Z = self.gcn(Z3)
            X_normal = self.line_n[i + 1](X.reshape((X.size()[0] * X.size()[1], -1)))
            padded_tensor_x = torch.zeros(bz, roi_num, 128, device=device)
            padded_tensor_x[:, :X.size()[1], :] = X_normal.reshape((X.size()[0], X.size()[1], -1))
            XX = torch.cat((XX, padded_tensor_x), dim=2)
            Z_normal = self.line_e[i + 1](Z.reshape((Z.size()[0] * Z.size()[1], -1)))
            padded_tensor_z = torch.zeros(bz, roi_num, 128, device=device)
            padded_tensor_z[:, :Z.size()[1], :] = Z_normal.reshape((Z.size()[0], Z.size()[1], -1))
            ZZ = torch.cat((ZZ, padded_tensor_z), dim=2)
            link_loss = link_loss + linkloss
            ent_loss = ent_loss + entloss
        XZ = torch.cat((XX, ZZ), 2)
        Y = self.clase(XZ)
        return Y, link_loss, ent_loss,final_A,S_list
class CAGNN(nn.Module):
    def __init__(self, feature_size, layer, nclass, top_k):
        super(CAGNN, self).__init__()
        self.model_n= NEGAN(layer, feature_size, top_k)
        self.fc1 = torch.nn.Linear(128 * 200, 256)
        self.relu = nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 32)
        self.fc3 = torch.nn.Linear(32, nclass)
    def forward(self, node_feature, A,Z, adj):
        x, link_loss, ent_loss,final_A,S_list = self.model_n(node_feature, A,Z, adj)
        x = x.view(x.shape[0], -1).to(device)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x, link_loss,ent_loss,final_A,S_list
