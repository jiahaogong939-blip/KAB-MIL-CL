""" import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import math
from collections import OrderedDict
from torch_geometric.nn import GATConv


#下面这个类就是机器学习过程中的嵌入
class ProjectionHead(nn.ModuleDict):
    def __init__(self, inputs, outputs):
        super(ProjectionHead, self).__init__()
        assert len(inputs) == len(outputs)
        layer_num = len(inputs)
        for i in range(layer_num):
            if i == layer_num - 1:
                self.add_module("linear%d" % i, nn.Linear(inputs[i], outputs[i]))
                # self.add_module("BN%d" % i, nn.BatchNorm1d(outputs[i]))
            else:
                self.add_module("linear%d" % i, nn.Linear(inputs[i], outputs[i]))
                # self.add_module("droput%d " % i, nn.Dropout(0.3))
                # self.add_module("BN%d" % i, nn.BatchNorm1d(outputs[i]))
                self.add_module("relu%d" % i, nn.ReLU())

    def forward(self, X):
        x_ = X
        for name, layer in self.items():
            x_ = layer(x_)

        return x_

#单层线性分类器，将输入特征 x（形状为 (batch_size, D)）通过线性层映射到类别空间（形状为 (batch_size, class_num)）。输出的 x_ 是一个未归一化的 logits 向量，表示每个类别的得分
class OneLayerClassifier(nn.Module):
    def __init__(self, D, class_num):
        super(OneLayerClassifier, self).__init__()
        self.linear = nn.Linear(D, class_num)

    def forward(self, x):
        x_ = self.linear(x)
        return x_

#双层非线性分类器
class TwoLayerClassifier(nn.Module):
    def __init__(self, D, class_num, dropout):
        super(TwoLayerClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(D,  int(D/2)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(D/2), class_num)
        )

    def forward(self, x):
        x_ = self.encoder(x)
        return x_

#将输入特征 X（形状为 (batch_size, D)）通过线性层和 Tanh 激活函数映射到隐藏空间（形状为 (batch_size, H)）。
#输出的 atten 可以理解为输入特征的注意力权重或特征表示。
class SimpleAttention(nn.Module):
    def __init__(self, D, H):
        super(SimpleAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(D, H),
            nn.Tanh()
        )

    def forward(self, X):
        atten = self.attention(X)
        return atten


class MILAttenBlock(nn.Module):
    def __init__(self, D, out_chan, head, dropout):
        super(MILAttenBlock, self).__init__()
        self.D, self.L = D, out_chan
        self.attention = nn.Sequential(
            nn.Linear(self.D, self.L),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(self.L, head)
        )

    def forward(self, X):
        A = self.attention(X)
        W = F.softmax(A, dim=1)
        z = torch.bmm(X.transpose(1, 2), W)
        z_ = z.squeeze()
        return z_


class GatedAttenBlock(nn.Module):
    def __init__(self, D, L, dropout):
        super(GatedAttenBlock, self).__init__()
        self.D, self.L = D, L

        self.attention_U = nn.Sequential(
            nn.Linear(self.D, self.L),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.D, self.L),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.weight = nn.Linear(self.L, 1)


    def forward(self, X):
        A_U = self.attention_U(X)
        A_V = self.attention_V(X)
        W_ = self.weight(A_V * A_U)
        W = F.softmax(W_, dim=1)
        z = torch.bmm(X.transpose(1, 2), W)
        z_ = z.squeeze()
        return z_


class MILAttenFusion(nn.ModuleDict):

    def __init__(self, params, class_num):
        
        super(MILAttenFusion, self).__init__()
        #A= params["B"]从params字典中获取B的值并将其赋予给A
        out_chan = params["out_chans"]
        D = params["dims"]
        dropout = params["dropout"]
        heads = params["heads"]

        self.mil_atten_layer1 = MILAttenBlock(D[0], out_chan[0], heads[0], dropout[0])
        self.mil_atten_layer2 = MILAttenBlock(D[1], out_chan[1], heads[1], dropout[1])

        self.flatten_dim = heads[-1]*D[-1]
        self.linear = nn.Linear(self.flatten_dim, 32)
        self.classifier = nn.Linear(32, class_num)


    def forward(self, X):
        z = None
        x_ = deepcopy(X)

        mil_atten_layer1_x_ = self.mil_atten_layer1(x_)
        mil_atten_layer1_x = mil_atten_layer1_x_.transpose(1, 2)

        mil_atten_layer2_x_ = self.mil_atten_layer2(mil_atten_layer1_x)
        mil_atten_layer2_x = mil_atten_layer2_x_.view(mil_atten_layer2_x_.shape[0], -1)
        x__ = self.linear(mil_atten_layer2_x)

        z = self.classifier(x__)

        return x__, z




class GMILAttenFusion(nn.ModuleDict):

    def __init__(self, D, L, CD,pinputs, poutputs, dropout):
        super(GMILAttenFusion, self).__init__()
        self.add_module("GAttention", GatedAttenBlock(D, L, dropout[0]))
        # self.add_module("Classifier", OneLayerClassifier(D))
        self.add_module("Classifier", TwoLayerClassifier(D, CD, dropout[1]))
        self.add_module("Projection", ProjectionHead(pinputs, poutputs))

    def forward(self, X):
        z = None
        projection = None
        x_ = deepcopy(X)
        for name, layer in self.items():
            if name == "Classifier":
                z = layer(x_)
            elif name == "Projection":
                projection = layer(x_)
            else:
                x_ = layer(x_)

        return projection, x_, z

 """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import math
from collections import OrderedDict
from torch_geometric.nn import GATConv


#下面这个类就是机器学习过程中的嵌入
class ProjectionHead(nn.ModuleDict):
    def __init__(self, inputs, outputs):
        super(ProjectionHead, self).__init__()
        assert len(inputs) == len(outputs)
        layer_num = len(inputs)
        for i in range(layer_num):
            if i == layer_num - 1:
                self.add_module("linear%d" % i, nn.Linear(inputs[i], outputs[i]))
                # self.add_module("BN%d" % i, nn.BatchNorm1d(outputs[i]))
            else:
                self.add_module("linear%d" % i, nn.Linear(inputs[i], outputs[i]))
                # self.add_module("droput%d " % i, nn.Dropout(0.3))
                # self.add_module("BN%d" % i, nn.BatchNorm1d(outputs[i]))
                self.add_module("relu%d" % i, nn.ReLU())

    def forward(self, X):
        x_ = X
        for name, layer in self.items():
            x_ = layer(x_)

        return x_

#单层线性分类器，将输入特征 x（形状为 (batch_size, D)）通过线性层映射到类别空间（形状为 (batch_size, class_num)）。输出的 x_ 是一个未归一化的 logits 向量，表示每个类别的得分
class OneLayerClassifier(nn.Module):
    def __init__(self, D, class_num):
        super(OneLayerClassifier, self).__init__()
        self.linear = nn.Linear(D, class_num)

    def forward(self, x):
        x_ = self.linear(x)
        return x_

#双层非线性分类器
class TwoLayerClassifier(nn.Module):
    def __init__(self, D, class_num, dropout):
        super(TwoLayerClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(D,  int(D/2)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(D/2), class_num)
        )

    def forward(self, x):
        x_ = self.encoder(x)
        return x_

#将输入特征 X（形状为 (batch_size, D)）通过线性层和 Tanh 激活函数映射到隐藏空间（形状为 (batch_size, H)）。
#输出的 atten 可以理解为输入特征的注意力权重或特征表示。
class SimpleAttention(nn.Module):
    def __init__(self, D, H):
        super(SimpleAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(D, H),
            nn.Tanh()
        )

    def forward(self, X):
        atten = self.attention(X)
        return atten


class MILAttenBlock(nn.Module):
    def __init__(self, D, out_chan, head, dropout):
        super(MILAttenBlock, self).__init__()
        self.D, self.L = D, out_chan
        self.attention = nn.Sequential(
            nn.Linear(self.D, self.L),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(self.L, head)
        )

    def forward(self, X):
        A = self.attention(X)
        W = F.softmax(A, dim=1)
        z = torch.bmm(X.transpose(1, 2), W)
        z_ = z.squeeze()
        return z_


class GatedAttenBlock(nn.Module):
    def __init__(self, D, L, dropout):
        super(GatedAttenBlock, self).__init__()
        self.D, self.L = D, L

        self.attention_U = nn.Sequential(
            nn.Linear(self.D, self.L),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.D, self.L),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.weight = nn.Linear(self.L, 1)


    def forward(self, X):
        A_U = self.attention_U(X)
        A_V = self.attention_V(X)
        W_ = self.weight(A_V * A_U)
        W = F.softmax(W_, dim=1)
        z = torch.bmm(X.transpose(1, 2), W)
        z_ = z.squeeze()
        return z_


class MILAttenFusion(nn.ModuleDict):

    def __init__(self, params, class_num):
        
        super(MILAttenFusion, self).__init__()
        #A= params["B"]从params字典中获取B的值并将其赋予给A
        out_chan = params["out_chans"]
        D = params["dims"]
        dropout = params["dropout"]
        heads = params["heads"]

        self.mil_atten_layer1 = MILAttenBlock(D[0], out_chan[0], heads[0], dropout[0])
        self.mil_atten_layer2 = MILAttenBlock(D[1], out_chan[1], heads[1], dropout[1])

        self.flatten_dim = heads[-1]*D[-1]
        self.linear = nn.Linear(self.flatten_dim, 32)
        self.classifier = nn.Linear(32, class_num)


    def forward(self, X):
        z = None
        x_ = deepcopy(X)

        mil_atten_layer1_x_ = self.mil_atten_layer1(x_)
        mil_atten_layer1_x = mil_atten_layer1_x_.transpose(1, 2)

        mil_atten_layer2_x_ = self.mil_atten_layer2(mil_atten_layer1_x)
        mil_atten_layer2_x = mil_atten_layer2_x_.view(mil_atten_layer2_x_.shape[0], -1)
        x__ = self.linear(mil_atten_layer2_x)

        z = self.classifier(x__)

        return x__, z




class GMILAttenFusion(nn.ModuleDict):

    def __init__(self, D, L, CD,pinputs, poutputs, dropout):
        super(GMILAttenFusion, self).__init__()
        self.add_module("GAttention", GatedAttenBlock(D, L, dropout[0]))
        # self.add_module("Classifier", OneLayerClassifier(D))
        self.add_module("Classifier", TwoLayerClassifier(D, CD, dropout[1]))
        self.add_module("Projection", ProjectionHead(pinputs, poutputs))

    def forward(self, X):
        z = None
        projection = None
        x_ = deepcopy(X)
        for name, layer in self.items():
            if name == "Classifier":
                z = layer(x_)
            elif name == "Projection":
                projection = layer(x_)
            else:
                x_ = layer(x_)

        return projection, x_, z




""" 
class SimpleAttention(nn.Module):
    def __init__(self, D, H):
        super(SimpleAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(D, H),
            nn.Tanh()
        )

    def forward(self, X):
        atten = self.attention(X)
        return atten

class GATAttentionHead(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha=0.2):
        super(GATAttentionHead, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.attention = nn.Linear(2 * out_features, 1, bias=False)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.gate = nn.Linear(out_features, out_features, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_i, h_j):
        Wh_i = self.W(h_i)
        Wh_j = self.W(h_j)
        gate_i = self.sigmoid(self.gate(Wh_i))
        gate_j = self.sigmoid(self.gate(Wh_j))
        Wh_i = Wh_i * gate_i
        Wh_j = Wh_j * gate_j
        e = self.attention(torch.cat([Wh_i, Wh_j], dim=-1))
        e = self.leakyrelu(e)
        return e

class GatedAttenBlock(nn.Module):
    def __init__(self, in_features, out_features, heads, dropout, alpha=0.2):
        super(GatedAttenBlock, self).__init__()
        self.heads = heads
        self.attentions = nn.ModuleList([
            GATAttentionHead(in_features, out_features, dropout, alpha)
            for _ in range(heads)
        ])
        self.output_linear = nn.Linear(heads * out_features, out_features)

    def forward(self, X):
        node_num = X.size(1)
        attention_outputs = []
        for attention in self.attentions:
            h_i = X.unsqueeze(2).repeat(1, 1, node_num, 1)
            h_j = X.unsqueeze(1).repeat(1, node_num, 1, 1)
            e = attention(h_i, h_j)
            attention_weights = F.softmax(e, dim=-1)
            attention_weights = F.dropout(attention_weights, self.dropout, training=self.training)
            h = torch.matmul(attention_weights.squeeze(-1), X)
            attention_outputs.append(h)
        z = torch.cat(attention_outputs, dim=-1)
        z = self.output_linear(z)
        return z

class MILAttenFusion(nn.ModuleDict):
    def __init__(self, params, class_num):
        super(MILAttenFusion, self).__init__()
        out_chan = params["out_chans"]
        D = params["dims"]
        dropout = params["dropout"]
        heads = params["heads"]

        self.mil_atten_layer1 = GatedAttenBlock(D[0], out_chan[0], heads[0], dropout[0])
        self.mil_atten_layer2 = GatedAttenBlock(D[1], out_chan[1], heads[1], dropout[1])

        self.flatten_dim = sum([heads[i] * out_chan[i] for i in range(len(heads))])
        self.linear = nn.Linear(self.flatten_dim, 32)
        self.classifier = nn.Linear(32, class_num)

    def forward(self, X):
        z = None
        x_ = X.clone()

        mil_atten_layer1_x_ = self.mil_atten_layer1(x_)
        mil_atten_layer2_x_ = self.mil_atten_layer2(mil_atten_layer1_x_)
        mil_atten_layer2_x = mil_atten_layer2_x_.view(mil_atten_layer2_x_.shape[0], -1)
        x__ = self.linear(mil_atten_layer2_x)

        z = self.classifier(x__)

        return x__, z

class GMILAttenFusion(nn.ModuleDict):
    def __init__(self, D, L, CD, pinputs, poutputs, dropout):
        super(GMILAttenFusion, self).__init__()
        self.add_module("GAttention", GatedAttenBlock(D, L, 1, dropout[0]))
        self.add_module("Classifier", TwoLayerClassifier(D, CD, dropout[1]))
        self.add_module("Projection", ProjectionHead(pinputs, poutputs))

    def forward(self, X):
        z = None
        projection = None
        x_ = X.clone()
        for name, layer in self.items():
            if name == "Classifier":
                z = layer(x_)
            elif name == "Projection":
                projection = layer(x_)
            else:
                x_ = layer(x_)
        return projection, x_, z """