import torch
from torch.linalg import inv, qr
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from numpy import sqrt
from numpy.linalg import svd


from copy import deepcopy


def coo_to_adj(coo_matr, num_nodes):
    adj_matr = torch.tensor( [[0 for _ in range(num_nodes)] for __ in range(num_nodes)], 
                              dtype=torch.float )
    for i_edge in range(len(coo_matr[0])):
        adj_matr[ coo_matr[0][i_edge] ][ coo_matr[1][i_edge] ] = 1
    return adj_matr
    

def row_sum_diag(matr):
    n = len(matr)
    ans = torch.tensor( [ [ 0 for _ in range(n) ] for __ in range(n)], 
                        dtype=torch.float )
    for i in range(n):
        ans[i][i] = torch.sum(matr[i])
    return ans


def add_selfloops(matr):
    ans = deepcopy(matr)
    for i in range(len(matr)):
        ans[i][i] += 1
    return ans


def sqrt_diag(diag_matr):
    matr_sqrt = deepcopy(diag_matr)
    for i in range(len(matr_sqrt)):
        matr_sqrt[i][i] = sqrt(matr_sqrt[i][i])
    return matr_sqrt


def g(matr):
    # orthogonalization
    ans, _ = qr(matr)

    # pca
    ans = PCA().fit_transform(ans)        

    # l2 normalization
    ans = normalize(ans)

    return torch.tensor(ans, dtype=torch.float)


dataset = Planetoid(root="./Cora", name="Cora")
data = dataset[0]

# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
# data = Data(x=x, edge_index=edge_index)

y = data.y
x = data.x
num_nodes = data.num_nodes
num_edges = data.num_edges
edge_index = data.edge_index

# adj = coo_to_adj(edge_index, num_nodes)
# adj_sl = add_selfloops(adj)
# u, _, _ = svd(adj)
# u = torch.tensor(u, dtype=torch.float)
# d = row_sum_diag(adj)
# d_sl = add_selfloops(d)
# adj_row = inv(d) @ adj

# d_sl_sqrt = sqrt_diag(d_sl)
# adj_sl_sym = inv(d_sl_sqrt) @ adj_sl @ inv(d_sl_sqrt)

# propagator = torch.cat( ( u, 
#                           g(x), 
#                           g(adj_row @ adj_row @ x), 
#                           g( adj_sl_sym @ adj_sl_sym @ x ) 
#                         ), 
#                         dim=1 )

# model = LogisticRegression().fit(propagator, y)

# print(propagator)
# print(g(x))

# print(d_sl, d_sl_sqrt, sep='\n')
# print(u)
# print(adj, adj_sl, sep="\n")
# print(d, d_sl, sep="\n")


