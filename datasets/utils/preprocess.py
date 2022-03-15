#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')

import torch
from datasets.utils import transforms as T

def unify_size(size_list):
  v, _ = size_list.max(0)

  max_H, max_W = v[0].item(), v[1].item()
  
  new_H = (1 + (max_H - 1) // 32) * 32
  new_W = (1 + (max_W - 1) // 32) * 32

  return (new_H, new_W)

def pad_images(image_list, new_size=None, to_stack=False):
  if new_size is None:
    image_sizes = [(img.shape[-2], img.shape[-1]) for img in image_list]
    image_sizes = torch.tensor(image_sizes)
    new_size = unify_size(image_sizes)
  
  new_images = []
  for i in range(len(image_list)):
    size = image_list[i].shape[-2:]
    padding_bottom = new_size[0] - size[0]
    padding_right = new_size[1] - size[1]
    new_images += [torch.nn.functional.pad(image_list[i], (0, padding_right, 0, padding_bottom))]
  
  if to_stack:
    new_images = torch.stack(new_images, 0)
  
  return new_images, new_size


def preprocess_data(batch, config):

  min_size, max_size = config['normal_size']

  # resize images
  images = batch['image']
  new_images = []
  original_sizes = []
  new_sizes = []
  for i in range(len(images)):
    image = images[i]
    original_size = [image.shape[-2], image.shape[-1]]
    original_sizes.append(original_size)
    image, _ = T.resize(image, None, min_size, max_size)
    new_size = [image.shape[-2], image.shape[-1]]
    new_sizes.append(new_size)
    new_images.append(image)
  images = new_images

  # batch data
  images, new_size = pad_images(images, to_stack=True)
  original_sizes = torch.tensor(original_sizes)
  new_sizes = torch.tensor(new_sizes)
  sizes = {'original_sizes':original_sizes, 'new_sizes':new_sizes}

  return images, sizes