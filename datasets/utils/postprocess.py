#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')
import os
import torch
import pickle
import numpy as np


def nms_fast(in_corners, H, W, dist_thresh):
  """
  Run a faster approximate Non-Max-Suppression on numpy corners shaped:
    3xN [x_i,y_i,conf_i]^T

  Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
  are zeros. Iterate through all the 1's and convert them either to -1 or 0.
  Suppress points by setting nearby values to 0.

  Grid Value Legend:
  -1 : Kept.
    0 : Empty or suppressed.
    1 : To be processed (converted to either kept or supressed).

  NOTE: The NMS first rounds points to integers, so NMS distance might not
  be exactly dist_thresh. It also assumes points are within image boundaries.

  Inputs
    in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
    H - Image height.
    W - Image width.
    dist_thresh - Distance to suppress, measured as an infinty norm distance.
  Returns
    nmsed_corners - 3xN numpy matrix with surviving corners.
    nmsed_inds - N length numpy vector with surviving corner indices.
  """
  grid = np.zeros((H, W)).astype(int) # Track NMS data.
  inds = np.zeros((H, W)).astype(int) # Store indices of points.
  # Sort by confidence and round to nearest int.
  inds1 = np.argsort(-in_corners[2,:])
  corners = in_corners[:,inds1]
  rcorners = corners[:2,:].round().astype(int) # Rounded corners.
  # Check for edge case of 0 or 1 corners.
  if rcorners.shape[1] == 0:
    return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
  if rcorners.shape[1] == 1:
    out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
    return out, np.zeros((1)).astype(int)
  # Initialize the grid.
  for i, rc in enumerate(rcorners.T):
    grid[rcorners[1,i], rcorners[0,i]] = 1
    inds[rcorners[1,i], rcorners[0,i]] = i
  # Pad the border of the grid, so that we can NMS points near the border.
  pad = dist_thresh
  grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
  # Iterate through points, highest to lowest conf, suppress neighborhood.
  count = 0
  for i, rc in enumerate(rcorners.T):
    # Account for top and left padding.
    pt = (rc[0]+pad, rc[1]+pad)
    if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
      grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
      grid[pt[1], pt[0]] = -1
      count += 1
  # Get all surviving -1's and return sorted array of remaining corners.
  keepy, keepx = np.where(grid==-1)
  keepy, keepx = keepy - pad, keepx - pad
  inds_keep = inds[keepy, keepx]
  out = corners[:, inds_keep]
  values = out[-1, :]
  inds2 = np.argsort(-values)
  out = out[:, inds2]
  out_inds = inds1[inds_keep[inds2]]
  return out, out_inds


def extract_points(prob, thr, border_remove=4, nms_dist=4, desc=None, image_shape=None):
  '''
  output:
    pts: numpy array, 3 * N, (x, y, score)
    point_descs: numpy array, N * 256
  '''

  # Convert pytorch -> numpy.
  heatmap = prob.squeeze().cpu() # H * W
  ys, xs = np.where(heatmap > thr) # Confidence threshold.
  if len(xs) == 0:
    pts = np.empty([3, 0])
    point_descs = np.empty([0, 256]) if desc is not None else None
    return pts, point_descs
  pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
  pts[0, :] = xs
  pts[1, :] = ys
  pts[2, :] = heatmap[ys, xs]
  H, W = heatmap.shape[-2:] if image_shape is None else image_shape
  pts, _ = nms_fast(pts, heatmap.shape[-2], heatmap.shape[-1], dist_thresh=nms_dist)
  inds = np.argsort(pts[2,:])
  pts = pts[:,inds[::-1]] # Sort by confidence.
  # Remove points along border.
  bord = border_remove
  toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
  toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
  toremove = np.logical_or(toremoveW, toremoveH)
  pts = pts[:, ~toremove]

  if pts.shape[1] == 0:
    point_descs = np.empty([0, 256])
  else:
    point_descs = []
    if desc is not None:
      desc_data = desc.squeeze().cpu() # 256 * H * W
      for i in range(pts.shape[1]):
        xs = int(pts[0][i])
        ys = int(pts[1][i])
        point_descs = point_descs + [desc_data[:, ys, xs]]

      point_descs = np.stack(point_descs)

  return pts, point_descs


def resize_results(new_image_size, original_image_size, detection=None, points=None):
  '''
  inputs:
    new_image_size: [h, w]
    original_image_size: [h, w]
    detection: Dict[Tensor], optional
    points: numpy array, N * 2, (hy, wx), optional

  output:
    Dict[Tensor],  numpy array
  '''
  # process maskrcnn result
  if detection is not None:
    if 'boxes' in detection:
      boxes = detection['boxes']
      if len(boxes) != 0:
        h, w = original_image_size
        boxes[:, 0] = torch.clamp(boxes[:, 0], 0, w)
        boxes[:, 1] = torch.clamp(boxes[:, 1], 0, h)
        boxes[:, 2] = torch.clamp(boxes[:, 2], 0, w)
        boxes[:, 3] = torch.clamp(boxes[:, 3], 0, h)
        detection['boxes'] = boxes

  # process superpoint result
  if points is not None:
    scales = np.array(original_image_size, dtype=float) / np.array(new_image_size)
    scale_h, scale_w = scales.tolist()
    
    points[:, 0] = points[:, 0] * scale_h
    points[:, 1] = points[:, 1] * scale_w

    points = np.around(points).astype(int)

  return detection, points


def move_dict_to_cpu(d):
  for k in d:
    d[k] = d[k].cpu()
  return d


def postprocess(new_image_sizes, original_image_sizes, points_detection_thr=0.3, points_output=None):
  '''
  inputs:
    new_image_sizes: tensor, N*2, [h, w]
    original_image_sizes: tensor, N*2, [h, w]
    points_detection_thr: points detection threshold
    detections: List[Dict[Tensor]], optional
    points_output: Dict[Tensor]

  output:
    List[Dict[Tensor]], List[Dict[Tensor]]
  '''
  new_image_sizes = new_image_sizes.detach().cpu().numpy().tolist()
  original_image_sizes = original_image_sizes.detach().cpu().numpy().tolist()

  # superpoint results
  new_points_output = None
  if points_output is not None:
    new_points_output = []
    points_output = move_dict_to_cpu(points_output)
    probs = points_output['prob']
    descs = points_output['desc']

    for i in range(probs.shape[0]):
      points, point_descs = extract_points(probs[i], points_detection_thr, desc=descs[i], image_shape=new_image_sizes[i])
      points = np.flip(points[:2, :].T, 1)

      _, points = resize_results(new_image_sizes[i], original_image_sizes[i], points=points)

      points = torch.tensor(points)
      point_descs = torch.tensor(point_descs)
      new_points_output.append({'points':points, 'point_descs':point_descs})

  return new_points_output


def postprocess_detections(new_image_sizes, original_image_sizes, detections=None):
  '''
  inputs:
    new_image_sizes: tensor, N*2, [h, w]
    original_image_sizes: tensor, N*2, [h, w]
    detections: List[Dict[Tensor]], optional

  output:
    List[Dict[Tensor]], List[Dict[Tensor]]
  '''
  new_image_sizes = new_image_sizes.detach().cpu().numpy().tolist()
  original_image_sizes = original_image_sizes.detach().cpu().numpy().tolist()

  # maskcrcnn resluts
  if detections is not None:
    for i in range(len(detections)):
      move_dict_to_cpu(detections[i])
      detections[i], _ = resize_results(new_image_sizes[i], original_image_sizes[i], detection=detections[i])

      # select top scores
      scores, boxes, labels, masks = detections[i]['scores'], detections[i]['boxes'], detections[i]['labels'], detections[i]['masks']
      
      sort_scores, idx = scores.sort(0, descending=True)
      good_scores = torch.nonzero(sort_scores>0.5).squeeze(1)
      good_inx = idx[good_scores]

      boxes = boxes[good_inx]
      labels = labels[good_inx]
      scores = scores[good_inx]
      masks = (masks[good_inx] > 0.15).float()

      detections[i]['scores'], detections[i]['boxes'], detections[i]['labels'], detections[i]['masks'] = scores, boxes, labels, masks

  return detections


def save_detection_results(image_names, save_root=None, detections=None):

  if save_root is not None:
    os.makedirs(save_root, exist_ok=True)

  for image_name, detection in zip(image_names, detections):
    save_name = image_name+'.pkl'
    with open(os.path.join(save_root, save_name), 'wb') as fp:
      pickle.dump(detection, fp)


def save_superpoint_features(image_names, save_root=None, points_output=None):

  if save_root is not None:
    os.makedirs(save_root, exist_ok=True)

  for points_o, image_name in zip(points_output, image_names):
    save_name = image_name + '.pkl'
    save_path = os.path.join(save_root, save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(os.path.join(save_path), 'wb') as fp:
      pickle.dump(points_o, fp)