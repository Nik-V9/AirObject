#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')

import torch
from datasets.utils.preprocess import preprocess_data
from datasets.utils import postprocess as post


def superpoint_inference(model, batch, data_config, detection_threshold, save_dir=None):
  with torch.no_grad():

    # preprocess
    images, sizes = preprocess_data(batch, data_config)
    original_sizes = sizes['original_sizes']
    new_sizes = sizes['new_sizes']

    # model inference
    points_output = model(images) 

    # postprocess
    points_output = post.postprocess(new_sizes, original_sizes, detection_threshold, points_output)

    # save superpoint features 
    if save_dir is not None:
      image_names = batch['image_name']
      post.save_superpoint_features(image_names, save_dir, points_output)

  return points_output