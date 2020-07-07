import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, gma, learnable, normalization = True, renormalization = False):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.gma = nn.Parameter(gma)
        self.normalization = normalization
        self.renormalization = renormalization
        if not learnable:
            self.gma.requires_grad = False
    def forward(self, x, adj):
#        print(torch.diag(self.gma))
#        adj = adj.to_dense()
        if self.normalization:            
            hat_adj = F.normalize(torch.diag(self.gma)+ adj , p=1, dim=1)
        else:
            hat_adj = torch.diag(self.gma)+ adj
        if self.renormalization:
            hat_adj = torch.diag(self.gma)+ adj 
#            print(adj.sparse_sum(1))
#            temp = hat_adj.to_dense()
            D = torch.inverse( torch.sqrt(torch.diag(hat_adj.sum(1))))
#            print(D)
#            D = D.cuda()
            hat_adj = D @ hat_adj @ D
#        adj = adj.to_sparse() 
        x = F.relu(self.gc1(x, hat_adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, hat_adj)
        return F.log_softmax(x, dim=1)
