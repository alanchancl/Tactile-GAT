# model.py
import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU, Dropout, Embedding
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F
from utils.utils import get_batch_edge_index

class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out

class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1,**kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        
        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        """"""
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))

        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,
                             return_attention_weights=return_attention_weights)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i,
                embedding,
                edges,
                return_attention_weights):

        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            embedding_i = embedding_i.unsqueeze(1).repeat(1,self.heads,1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1,self.heads,1)

            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

            cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
            cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)
        else:
            key_i = x_i
            key_j = x_j

            cat_att_i = self.att_i
            cat_att_j = self.att_j

        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)


        alpha = alpha.view(-1, self.heads, 1)


        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class GATLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GATLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        
        out = self.bn(out)
        out = self.relu(out)
        return out, att_weight
    
class TactileGAT(nn.Module):
    def __init__(self, edge_index_sets, node_num, graph_name, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20):
        super(TactileGAT, self).__init__()
        self.edge_index_sets = edge_index_sets
        self.node_num = node_num
        self.graph_name = graph_name
        self.dim = dim
        self.device = None
        self.topk = topk
        self.embedding = Embedding(node_num, dim)
        self.bn_outlayer_in = BatchNorm1d(dim * 2 if graph_name == '' else dim)
        self.gnn_layers = nn.ModuleList([
            GATLayer(input_dim, dim, inter_dim=dim+dim, heads=1) for _ in range(len(edge_index_sets))
        ])
        self.out_layer = OutLayer(dim * len(edge_index_sets), node_num, out_layer_num, inter_num=out_layer_inter_dim)
        self.class_linear = Linear(node_num, 20)
        self.dropout = Dropout(0.2)
        self.cache_edge_index_sets = [None] * len(edge_index_sets)
        self.init_params()
        self.learned_graph = None

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data):
        x, self.device = data.clone().detach(), data.device
        batch_num = x.size(0)
        x = x.view(-1, x.size(-1))

        gat_outputs = []
        attention_weights_list = []
        for i, edge_index in enumerate(self.edge_index_sets):
            batch_edge_index = self.prepare_batch_edge_index(i, edge_index, batch_num)
            gat_out, attn_weights = self.gnn_layers[i](x, batch_edge_index, node_num=self.node_num * batch_num, embedding=None)
            gat_outputs.append(gat_out)
            attention_weights_list.append(attn_weights)

        self.learned_graph = torch.cat(attention_weights_list, dim=0)
        x = torch.cat(gat_outputs, dim=1).view(batch_num, self.node_num, -1)
        x = self.process_output(x)

        return self.class_linear(x.view(-1, self.node_num))

    def prepare_batch_edge_index(self, i, edge_index, batch_num):
        if self.cache_edge_index_sets[i] is None or self.cache_edge_index_sets[i].size(1) != edge_index.size(1) * batch_num:
            self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, self.node_num).to(self.device)
        return self.cache_edge_index_sets[i]

    def process_output(self, x):
        x = F.relu(self.bn_outlayer_in(x.permute(0, 2, 1))).permute(0, 2, 1)
        x = self.dropout(self.out_layer(x))
        return x

    def get_learned_graph(self):
        return self.learned_graph
