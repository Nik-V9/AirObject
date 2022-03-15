#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import re
import collections
from torch._six import string_classes

class BatchCollator(object):
    '''
    pack dict batch
    '''
    def __init__(self):
        super(BatchCollator,self).__init__()

    def __call__(self, batch):
        data= {}
        size = len(batch)
        for key in batch[0]:
            l = []
            for i in range(size):
                l = l + [batch[i][key]]
            data[key] = l
        return data

# def vis_custom_collate(batch):
#     r"""Puts each tensor data field into a tensor with outer dimension batch size
#     and Puts list data into list with length batch size"""

#     tensors = []
#     if len(batch[0]) == 3:
#       list_1 = []
#       list_2 = []
#     elif len(batch[0]) == 2:
#       list_1 = []
    
#     for i in range(len(batch)):
#       tensors.append(batch[i][0])
#       if len(batch[0]) == 3:
#         list_1.append(batch[i][1])
#         list_2.append(batch[i][2])
#       elif len(batch[0]) == 2:
#         list_1.append(batch[i][1])
    
#     tensor = torch.stack(tensors, 0)

#     if len(batch[0]) == 3:
#       return tensor, list_1, list_2
#     elif len(batch[0]) == 2:
#       return tensor, list_1
#     else:
#       return tensor
    
def vis_custom_collate(batch):
    r"""Puts each tensor data field into a tensor with outer dimension batch size
    and Puts list data into list with length batch size"""

    if len(batch[0]) == 3:
      tensors = []
      list_1 = []
      list_2 = []
    if len(batch[0]) == 2:
      list_1 = []
      list_2 = []
    elif len(batch[0]) == 1:
      list_1 = []
    
    for i in range(len(batch)):
      if len(batch[0]) == 3:
        tensors.append(batch[i][0])
        list_1.append(batch[i][1])
        list_2.append(batch[i][2])
      elif len(batch[0]) == 2:
        list_1.append(batch[i][0])
        list_2.append(batch[i][1])
      elif len(batch[0]) == 1:
        list_1.append(batch[i][0])

    if len(batch[0]) == 3:
      tensor = torch.stack(tensors, 0)
      return tensor, list_1, list_2
    elif len(batch[0]) == 2:
      return list_1, list_2
    elif len(batch[0]) == 1:
      return list_1

def eval_custom_collate(batch):
    r"""Puts each tensor data field into a tensor with outer dimension batch size
    and Puts list data into list with length batch size"""

    if len(batch[0]) == 4:
      tensors = []
      list_1 = []
      list_2 = []
      list_3 = []
    elif len(batch[0]) == 3:
      list_1 = []
      list_2 = []
      list_3 = []
    elif len(batch[0]) == 2:
      list_1 = []
      list_2 = []
    elif len(batch[0]) == 1:
      list_1 = []
    
    for i in range(len(batch)):
      if len(batch[0]) == 4:
        tensors.append(batch[i][0])
        list_1.append(batch[i][1])
        list_2.append(batch[i][2])
        list_3.append(batch[i][3])
      elif len(batch[0]) == 3:
        list_1.append(batch[i][0])
        list_2.append(batch[i][1])
        list_3.append(batch[i][2])
      elif len(batch[0]) == 2:
        list_1.append(batch[i][0])
        list_2.append(batch[i][1])
      elif len(batch[0]) == 1:
        list_1.append(batch[i][0])

    if len(batch[0]) == 4:
      tensor = torch.stack(tensors, 0)
      return tensor, list_1, list_2, list_3
    elif len(batch[0]) == 3:
      return list_1, list_2, list_3
    elif len(batch[0]) == 2:
      return list_1, list_2
    elif len(batch[0]) == 1:
      return list_1