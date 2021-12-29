from torch_geometric.nn import GINEConv, GCNConv, MessagePassing, BatchNorm, GATConv
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_add_pool as gadp
from torch.nn import Sequential, Linear, ReLU, ModuleList
import torch.nn.functional as F
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def GIN_Block(h_dim, nGIN):
    torch.manual_seed(0)
    nn = Sequential(Linear(h_dim, h_dim), ReLU(), Linear(h_dim, h_dim))
    gins = ModuleList([GINEConv(nn) for _ in range(nGIN)])
    bns = ModuleList([BatchNorm(h_dim) for _ in range(nGIN)])
    return gins.to(device), bns.to(device)

def GCN_Block(h_dim, nGCN):
    torch.manual_seed(0)
    gcn_modules = ModuleList([GCNConv(h_dim, h_dim) for _ in range(nGCN)]).append(GCNConv(h_dim, h_dim//2))
    return gcn_modules.to(device)


def AGG_Block(x, batch):
    return gadp(x, batch).add(gmp(x, batch).add(gap(x, batch)))

class HybridGICN(MessagePassing):
    def __init__(self, trial, props):
        super(HybridGICN, self).__init__()
        torch.manual_seed(0)
        h_dim = trial.suggest_int("h_dim",64,128)
        self.h_dim = h_dim
        self.props = props
        self.device = device 
        self.DOP = [trial.suggest_float(f"DO{d+1}", 0.2,0.5) for d in range(self.props)]
        

        self.nGIN = trial.suggest_int(f'nGIN', 1,4)
        self.GIN, self.BN = GIN_Block(h_dim, nGIN=self.nGIN)
        self.GCN_List = [_ for _ in range(self.props)]

        
        
        for l in range(self.props):
            self.GCN_List[l] = GCN_Block(h_dim, trial.suggest_int(f'nGCN{l+1}', 1,3))

        torch.manual_seed(0)
        self.lins = ModuleList([Linear(h_dim//2, h_dim//4) for _ in range(self.props)])
        self.outs = ModuleList([Linear(int(h_dim//4), 1) for _ in range(self.props)])

    def forward(self, x, edge_index, edge_attr, batch):
        torch.manual_seed(0)
        node_emb = Linear(x.size(-1), self.h_dim).to(device)
        edge_emb = Linear(edge_attr.size(-1), self.h_dim).to(device)
        x = node_emb(x)
        edge_attr = edge_emb(edge_attr)
        results = []
        inter_val = {}
        for i in range(self.nGIN):
            x = self.GIN[i](x, edge_index = edge_index, edge_attr = edge_attr)
            x = self.BN[i](x)
        for npr in range(self.props):
            inter_val[f'x{npr}'] = self.GCN_List[npr][0](x, edge_index)
            if len(self.GCN_List[npr]) > 1:    
                for j in range(1, len(self.GCN_List[npr])):
                    inter_val[f'x{npr}'] = self.GCN_List[npr][j](inter_val[f'x{npr}'], edge_index)
            
            inter_val[f'x{npr}'] = AGG_Block(inter_val[f'x{npr}'], batch)
            inter_val[f'x{npr}']  = F.dropout(inter_val[f'x{npr}'], p=self.DOP[npr], training=self.training)
            
            results.append(self.outs[npr](torch.relu(self.lins[npr](inter_val[f'x{npr}']))))

        return results
# %%
