#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from model.graph_models.attention import GraphAtten

class AirObject(nn.Module):
  def __init__(self, config):
    super(AirObject, self).__init__()
    points_encoder_dims = config['points_encoder_dims']
    descriptor_dim = config['descriptor_dim']
    nhid = config['hidden_dim']
    alpha = config['alpha']
    nheads = config['nheads']
    nout = config['nout']
    nfeat = descriptor_dim + points_encoder_dims[-1]
    temporal_nfeat = config['temporal_encoder_dim']
    temporal_nout = config['temporal_encoder_out_dim']
    temporal_kernel_size = config['temporal_kernel_size']
    self.points_encoder = PointsEncoder(points_encoder_dims)
    self.gcn = GCN(nfeat, nhid, nout, alpha, nheads)
    self.tcn = TCN(temporal_nfeat, temporal_nout, temporal_kernel_size)

  def forward(self, batch_points, batch_descs, batch_adj):
    '''
    inputs:
      batch_points: List[Tensor], normalized points, sequence of tensors belonging to an object
      batch_descs: List[Tensor], local feature descriptors, sequence of tensors belonging to an object
      batch_adj: List[Tensor], sequence of adjacency matrices corresponding to the triangulation based object points graph
    '''
    
    batch_node_features = []

    for points, descs, adj in zip(batch_points, batch_descs, batch_adj):
      encoded_points = self.points_encoder(points)
      features = torch.cat((descs, encoded_points), dim=1)

      node_features = self.gcn(features, adj)
      batch_node_features.append(node_features)

    if len(batch_node_features) != 1:
      evol_features = torch.cat(batch_node_features)
    else:
      evol_features = batch_node_features[0]

    airobj_desc = self.tcn(evol_features.unsqueeze(0))

    return airobj_desc


class PointsEncoder(nn.Module):
  def __init__(self, dims):
    super(PointsEncoder, self).__init__()  

    layers = []
    for i in range(len(dims)-1):
      layers.append(nn.Linear(dims[i], dims[i+1]))
      if i != len(dims)-2:
        layers.append(nn.BatchNorm1d((dims[i+1])))
        layers.append(nn.ReLU())

    self.layers = layers
    for i, layer in enumerate(self.layers):
      self.add_module('point_encoder{}'.format(i), layer)

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    x = nn.functional.normalize(x, p=2, dim=-1)
    return x


class GCN(nn.Module):
  def __init__(self, nfeat, nhid, nout, alpha=0.2, nheads=8):
    super(GCN, self).__init__()

    self.atten1 = GraphAtten(nfeat, nhid, nfeat, alpha, nheads)
    self.atten2 = GraphAtten(nfeat, nhid, nfeat, alpha, nheads)
    self.tran1 = nn.Linear(nfeat, nfeat)
    self.relu = nn.ReLU()
    self.sparsification = Sparsification(nfeat, nout)

  def forward(self, x, adj):
    x = self.atten1(x, adj)
    x = self.atten2(x, adj)
    x = self.relu(self.tran1(x))

    features = self.sparsification(x)

    return features


class Sparsification(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(Sparsification, self).__init__()

    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=-1)
    self.location_encoder1 = nn.Linear(input_dim, input_dim)
    self.location_encoder2 = nn.Linear(input_dim, output_dim)

    self.feature_encoder1 = nn.Linear(input_dim, input_dim)
    self.feature_encoder2 = nn.Linear(input_dim, output_dim)

  def forward(self, x):

    descriptor = self.relu(self.feature_encoder1(x))
    descriptor = self.relu(self.feature_encoder2(descriptor))

    locations = self.relu(self.location_encoder1(x))
    locations = self.relu(self.location_encoder2(locations))

    features = locations * descriptor

    return features


class TCN(nn.Module):
    def __init__(self, inDims, outDims, w=1):
        super(TCN, self).__init__()

        self.inDims = inDims
        self.outDims = outDims
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):

        x = x.permute(0,2,1) # from [B,T,C] to [B,C,T]
        temporal_Ft = self.conv(x)
        temporal_Ft = torch.mean(temporal_Ft,-1)

        temporal_Ft = temporal_Ft.view(temporal_Ft.size(0), -1)
        temporal_Ft = nn.functional.normalize(temporal_Ft, p=2, dim=1)

        return temporal_Ft