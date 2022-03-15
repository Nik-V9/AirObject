#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class NetVLADDescriptor(nn.Module):
  def __init__(self, config):
    super(NetVLADDescriptor, self).__init__()
    descriptor_dim = config['descriptor_dim']
    vlad_numc = config['vlad_numc']
    nfeat = descriptor_dim
    self.netvlad = NetVLAD(vlad_numc, nfeat)

  def forward(self, batch_descs):
    '''
    inputs:
      batch_descs: List[Tensor], local descriptors belonging to an object
    '''
    batch_features = []
    for descs in batch_descs:
      features = descs
      features = features.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # reshape, N * C ---> 1 * C * N * 1
      features = self.netvlad(features)   # 1 * D
      batch_features.append(features.squeeze())
    batch_features = torch.stack(batch_features)
    batch_features = nn.functional.normalize(batch_features, p=2, dim=-1)
    return batch_features


class NetVLAD(nn.Module):
  """NetVLAD layer implementation"""

  def __init__(self, num_clusters=32, dim=256, alpha=100.0,
                normalize_input=False):
    """
    Args:
      num_clusters : int
        The number of clusters
      dim : int
        Dimension of descriptors
      alpha : float
        Parameter of initialization. Larger value is harder assignment.
      normalize_input : bool
        If true, descriptor-wise L2 normalization is applied to input.
    """
    super(NetVLAD, self).__init__()
    self.num_clusters = num_clusters
    self.dim = dim
    self.alpha = alpha
    self.normalize_input = normalize_input
    self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
    self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
    self._init_params()

  def _init_params(self):
    self.conv.weight = nn.Parameter(
        (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
    )
    self.conv.bias = nn.Parameter(
        - self.alpha * self.centroids.norm(dim=1)
    )

  def forward(self, x):
    N, C = x.shape[:2]

    if self.normalize_input:
      x = F.normalize(x, p=2, dim=1)  # across descriptor dim

    # soft-assignment
    soft_assign = self.conv(x).view(N, self.num_clusters, -1)
    soft_assign = F.softmax(soft_assign, dim=1)

    x_flatten = x.view(N, C, -1)
    
    # calculate residuals to each clusters
    residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
        self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
    residual *= soft_assign.unsqueeze(2)
    vlad = residual.sum(dim=-1)

    vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
    vlad = vlad.view(x.size(0), -1)  # flatten
    vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

    return vlad