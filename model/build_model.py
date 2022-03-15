#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

from model.backbone.resnet_fpn import resnet_fpn_backbone
from model.mask_rcnn.mask_rcnn import MaskRCNN
from model.backbone.fcn import VGGNet
from model.superpoint.vgg_like import VggLike
from model.graph_models.object_descriptor import ObjectDescriptor
from model.netvlad import NetVLADDescriptor
from model.seqnet import SeqNet
from model.airobject import AirObject

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

def build_maskrcnn(configs):
  ## command line config
  num_gpu = configs['num_gpu']
  use_gpu = (len(num_gpu) > 0) and configs['use_gpu']
  pretrained_model_path = configs['maskrcnn_model_path']
  public_model = configs['public_model']
  ## data cofig
  nclass = configs['data']['nclass']
  ## mask_rcnn config
  maskrcnn_model_config = configs['model']['maskrcnn']
  backbone_type = maskrcnn_model_config['backbone_type']
  image_mean = maskrcnn_model_config['image_mean']
  image_std = maskrcnn_model_config['image_std']
  trainable_layers = maskrcnn_model_config['trainable_layers']

  backbone = resnet_fpn_backbone(backbone_type, False, trainable_layers=trainable_layers)
  model = MaskRCNN(backbone, nclass, image_mean=image_mean, image_std=image_std)

  if pretrained_model_path != "" and public_model:
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model_path)
    remove_dict = ['roi_heads.box_predictor.cls_score.weight',
                   'roi_heads.box_predictor.cls_score.bias',
                   'roi_heads.box_predictor.bbox_pred.weight',
                   'roi_heads.box_predictor.bbox_pred.bias',
                   'roi_heads.mask_predictor.mask_fcn_logits.weight',
                   'roi_heads.mask_predictor.mask_fcn_logits.bias']
    pretrained_dict = {k:v for k, v in pretrained_dict.items() if ((k in model_dict) and (k not in remove_dict))}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

  if use_gpu:
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=num_gpu)
    print("Finish cuda loading")

  if pretrained_model_path != "":
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model_path, map_location=torch.device('cuda:{}'.format(num_gpu[0])))
    model_dict.update(pretrained_dict)
    if use_gpu:
      model.load_state_dict(model_dict)
    else:
      print("loading on cpu")
      model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
    print("load model from {}".format(pretrained_model_path))

  return model

def build_superpoint_model(configs, requires_grad=True):
  num_gpu = configs['num_gpu']
  use_gpu = (len(num_gpu) > 0) and configs['use_gpu']
  pretrained_model_path = configs['superpoint_model_path']

  vgg_model = VGGNet(requires_grad=requires_grad)
  model = VggLike(vgg_model)

  if use_gpu:
    model = model.to(torch.device('cuda:{}'.format(num_gpu[0])))
    model = torch.nn.DataParallel(model, device_ids=num_gpu)
    print("Finish cuda loading")

  if pretrained_model_path != "":
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model_path, map_location=torch.device('cuda:{}'.format(num_gpu[0])))
    model_dict.update(pretrained_dict)
    if use_gpu:
      model.load_state_dict(model_dict)
    else:
      print("loading on cpu")
      model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
    print("load model from {}".format(pretrained_model_path))
  
  return model

def build_gcn(configs):
  num_gpu = configs['num_gpu']
  use_gpu = (len(num_gpu) > 0)
  gcn_config = configs['model']['gcn']
  pretrained_model_path = configs['graph_model_path']

  model = ObjectDescriptor(gcn_config)

  if use_gpu:
    model = model.to(torch.device('cuda:{}'.format(num_gpu[0])))
    model = torch.nn.DataParallel(model, device_ids=num_gpu)
    print("Finish cuda loading")

  if pretrained_model_path != "":
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model_path, map_location=torch.device('cuda:{}'.format(num_gpu[0])))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    if use_gpu:
      model.load_state_dict(model_dict)
    else:
      print("loading on cpu")
      model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
    print("load model from {}".format(pretrained_model_path))

  return model

def build_netvlad(configs):
  num_gpu = configs['num_gpu']
  use_gpu = (len(num_gpu) > 0)
  netvlad_config = configs['model']['netvlad']
  pretrained_model_path = configs['netvlad_model_path']

  model = NetVLADDescriptor(netvlad_config)

  if use_gpu:
    model = model.to(torch.device('cuda:{}'.format(num_gpu[0])))
    model = torch.nn.DataParallel(model, device_ids=num_gpu)
    print("Finish cuda loading")
    
  if pretrained_model_path != "":
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model_path, map_location=torch.device('cuda:{}'.format(num_gpu[0])))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    if use_gpu:
      model.load_state_dict(model_dict)
    else:
      print("loading on cpu")
      model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
    print("load model from {}".format(pretrained_model_path))

  return model

def build_seqnet(configs):
  num_gpu = configs['num_gpu']
  use_gpu = (len(num_gpu) > 0)
  seqnet_config = configs['model']['seqnet']
  pretrained_model_path = configs['seqnet_model_path']
  
  model = torch.nn.Module()
  seqFt = SeqNet(seqnet_config['encoder_dim'], seqnet_config['out_dim'], seqnet_config['kernel_size'])
  model.add_module('pool', torch.nn.Sequential(*[seqFt, Flatten(), L2Norm()]))

  if use_gpu:
    model = model.to(torch.device('cuda:{}'.format(num_gpu[0])))
    model = torch.nn.DataParallel(model, device_ids=num_gpu)
    print("Finish cuda loading")
    
  if pretrained_model_path != "":
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model_path, map_location=torch.device('cuda:{}'.format(num_gpu[0])))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    if use_gpu:
      model.load_state_dict(model_dict)
    else:
      print("loading on cpu")
      model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
    print("load model from {}".format(pretrained_model_path))

  return model

def build_airobj(configs):
  num_gpu = configs['num_gpu']
  use_gpu = (len(num_gpu) > 0)
  airobj_config = configs['model']['airobj']
  pretrained_model_path = configs['airobj_model_path']

  model = AirObject(airobj_config)

  if use_gpu:
    model = model.to(torch.device('cuda:{}'.format(num_gpu[0])))
    model = torch.nn.DataParallel(model, device_ids=num_gpu)
    print("Finish cuda loading")

  if pretrained_model_path != "":
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model_path, map_location=torch.device('cuda:{}'.format(num_gpu[0])))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    if use_gpu:
      model.load_state_dict(model_dict)
    else:
      print("loading on cpu")
      model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
    print("load model from {}".format(pretrained_model_path))

  return model