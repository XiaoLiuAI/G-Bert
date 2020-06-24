import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax, add_self_loops

from build_tree import build_icd9_tree, build_atc_tree
from build_tree import build_stage_one_edges, build_stage_two_edges


class OntologyEmbedding(nn.Module):
    """
    并不是单纯的Embedding，包含图模型
    """
    def __init__(self, voc, build_tree_func,
                 in_channels=100, out_channels=20, heads=5):
        super(OntologyEmbedding, self).__init__()

        # initial tree edges
        res, graph_voc = build_tree_func(list(voc.idx2word.values()))
        """
        res: list of sample, while each sample represents the raw code and sub codes 
            which could be used as nodes in code graph
        graph_voc: vocabulary of nodes/codes in samples in res
        """
        stage_one_edges = build_stage_one_edges(res, graph_voc)
        stage_two_edges = build_stage_two_edges(res, graph_voc)

        self.edges1 = torch.tensor(stage_one_edges)
        self.edges2 = torch.tensor(stage_two_edges)
        self.graph_voc = graph_voc

        # construct model
        assert in_channels == heads * out_channels
        self.g = GATConv(in_channels=in_channels,
                         out_channels=out_channels,
                         heads=heads)

        # tree embedding
        num_nodes = len(graph_voc.word2idx)
        # 包括路径上的节点的embedding，超过实际code的数量
        self.embedding = nn.Parameter(torch.Tensor(num_nodes, in_channels))

        # idx mapping: FROM leaf node in graphvoc TO voc
        self.idx_mapping = [self.graph_voc.word2idx[word]
                            for word in voc.idx2word.values()]

        self.init_params()

    def get_all_graph_emb(self):
        emb = self.embedding
        """
        对所有nodes，所有edges进行图卷积(MessagePassing)操作
        """
        emb = self.g(self.g(emb, self.edges1.to(emb.device)),
                     self.edges2.to(emb.device))
        return emb

    def forward(self):
        """
        每次都会对所有nodes调用graph attention network.
        运行效率有点低
        :param idx: [N, L]
        :return:
        """
        emb = self.get_all_graph_emb()

        return emb[self.idx_mapping]

    def init_params(self):
        glorot(self.embedding)


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{j} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions. (default:
            :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = nn.Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        # attention layer的参数,attention layer为一个单层神经网络,激活函数为LeakyReLU
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index):
        """
        x: N * in_channels, all nodes embeddings
        edge_index: 2 * E, all edges
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # 加上当前node的self edge

        # 线性层, x: N * head * out_channels
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        # return self.propagate('add', edge_index, x=x, num_nodes=x.size(0))  # 返回conv的结果,并不会更新node
        return self.propagate(edge_index, aggr="add", x=x, size=(x.size(0), x.size(0)),
                              num_nodes=x.size(0), pass_edge_index=edge_index)  # 返回conv的结果,并不会更新node

    def message(self, x_i, x_j, pass_edge_index, num_nodes):
        """
        x_i: (E x head x out_channels)
        x_j: (E x head x out_channels)
        pass_edge_index: 2 * E TODO 新的pytorch_geometric框架不认同这种做法
        num_nodes: embedding size

        return: 注意力权重相乘下的特征
        """
        # Compute attention coefficients.
        # compute multi head attention based on head and tail nodes
        # E * head
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, pass_edge_index[0], num_nodes)

        alpha = F.dropout(alpha, p=self.dropout)

        # (E * head * out_channels) * (E * head) 每个head采用不同attention权重
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        """
        aggr_out: N * head * out_channels
        return: N * out_channels/(head * out_channels)
        """
        if self.concat is True:  # 决定多head是合并还是取均值
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class ConcatEmbeddings(nn.Module):
    """Concat rx and dx ontology embedding for easy access
    """

    def __init__(self, config, dx_voc, rx_voc):
        super(ConcatEmbeddings, self).__init__()
        # special token: "[PAD]", "[CLS]", "[MASK]"
        self.special_embedding = nn.Parameter(
            torch.Tensor(config.vocab_size - len(dx_voc.idx2word) - len(rx_voc.idx2word), config.hidden_size))
        self.rx_embedding = OntologyEmbedding(rx_voc, build_atc_tree,
                                              config.hidden_size, config.graph_hidden_size,
                                              config.graph_heads)
        self.dx_embedding = OntologyEmbedding(dx_voc, build_icd9_tree,
                                              config.hidden_size, config.graph_hidden_size,
                                              config.graph_heads)
        self.init_params()

    def forward(self, input_ids):
        # 每次调用 OntologyEmbedding 的 forward
        emb = torch.cat(
            [self.special_embedding, self.rx_embedding(), self.dx_embedding()], dim=0)
        return emb[input_ids]

    def init_params(self):
        glorot(self.special_embedding)


class FuseEmbeddings(nn.Module):
    """Construct the embeddings from ontology, patient info and type embeddings.
    """

    def __init__(self, config, dx_voc, rx_voc):
        super(FuseEmbeddings, self).__init__()
        self.ontology_embedding = ConcatEmbeddings(config, dx_voc, rx_voc)
        self.type_embedding = nn.Embedding(2, config.hidden_size)
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, input_ids, input_types=None, input_positions=None):
        """
        :param input_ids: [B, L]
        :param input_types: [B, L]
        :param input_positions:
        :return:
        """
        # return self.ontology_embedding(input_ids)
        ontology_embedding = self.ontology_embedding(
            input_ids) + self.type_embedding(input_types)
        return ontology_embedding
