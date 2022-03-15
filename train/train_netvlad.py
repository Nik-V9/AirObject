import os
import sys
sys.path.append('.')
import yaml
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from datasets.vis.vis import YouTubeVIS
from datasets.utils.batch_collator import vis_custom_collate

import numpy as np
from scipy.spatial import Delaunay

from model.build_model import build_netvlad
from model.graph_models.descriptor_loss import DescriptorLoss

def train(configs):
    ## netvlad model config
    netvlad_config = configs['model']['netvlad']
    batch_size = netvlad_config['train']['batch_size']
    epochs = netvlad_config['train']['epochs']
    lr = netvlad_config['train']['lr']
    checkpoint = netvlad_config['train']['checkpoint']
    lambda_d = netvlad_config['train']['lambda_d']
    ## files config
    base_dir = configs['base_dir']
    videos = configs['videos']
    log_dir = configs['log_dir']
    ## other
    configs['num_gpu'] = [0]
    configs['public_model'] = 0

    # data
    dataset = YouTubeVIS(basedir=base_dir, videos=videos, return_img=False, return_points=True)

    loader = data.DataLoader(dataset=dataset, collate_fn=vis_custom_collate, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model
    netvlad_model = build_netvlad(configs)
    netvlad_model.train()

    # loss
    criterion = DescriptorLoss(netvlad_config)

    # tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S')+'_netvlad'))

    # write checkpoints in logdir
    logdir = writer.file_writer.get_logdir()
    save_dir = os.path.join(logdir, 'saved_model')
    os.makedirs(save_dir, exist_ok=True)

    print('===> Saving state to:', logdir)

    # Optimizer
    optimizer = torch.optim.Adam(netvlad_model.parameters(), lr=lr)

    sum_iter = 0
    
    if configs['resume']:
        optimizer_dict = torch.load(configs['optimizer_path'], map_location=torch.device('cuda:{}'.format(configs['num_gpu'][0])))
        optimizer.load_state_dict(optimizer_dict)
        print('===> Optimizer Loaded')
        sum_iter = configs['sum_iter']

    for epoch in tqdm(range(epochs), desc='train'):
        
        for iteration, batch in enumerate(tqdm(loader)):
            
            ann_masks, points = batch[0], batch[1]
            
            # Get Sequence of Points, Descs & Adjs grouped by Objects
            batch_objects = {}
            obj_key = 0
            for i in range(len(ann_masks)): # Batch Size
                objects = {}
                for j in range(len(ann_masks[i])): # Sequence Length
                    keypoints = points[i][j]['points']
                    descriptors = points[i][j]['point_descs']
                    for a in range(len(ann_masks[i][j])):
                        
                        ann_mask = ann_masks[i][j][a]['ann_mask']
                        object_filter = ann_mask[keypoints[:,0].T,keypoints[:,1].T]
                        
                        np_obj_pts = keypoints[np.where(object_filter==1)[0]].numpy()
                        try:
                            tri = Delaunay(np_obj_pts, qhull_options='QJ')
                        except:
                            continue

                        obj_id = str(ann_masks[i][j][a]['obj_id'])

                        if obj_id not in objects.keys():
                            objects[obj_id] = {}
                            objects[obj_id]['seq_points'] = []
                            objects[obj_id]['seq_descs'] = []

                        objects[obj_id]['seq_points'].append(keypoints[np.where(object_filter==1)[0]].float().to(device))
                        objects[obj_id]['seq_descs'].append(descriptors[np.where(object_filter==1)[0]].float().to(device))

                for obj_id in objects.keys():
                    batch_objects[str(obj_key)] = {}
                    batch_objects[str(obj_key)]['seq_points'] = objects[obj_id]['seq_points']
                    batch_objects[str(obj_key)]['seq_descs'] = objects[obj_id]['seq_descs']
                    batch_objects[str(obj_key)]['obj_id'] = int(obj_id)
                    obj_key += 1

            batch_descs = []
            batch_obj_ids = []
            num_objs = []
            
            for obj_key in batch_objects.keys():
                iter_points, iter_descs = [], []
                iter_points = batch_objects[obj_key]['seq_points']
                iter_descs = batch_objects[obj_key]['seq_descs']

                iter_obj_descs = netvlad_model(iter_descs)
                
                batch_descs.append(iter_obj_descs)
                batch_obj_ids.append(batch_objects[obj_key]['obj_id'])
                num_objs.append(len(iter_descs))

            batch_descs = torch.cat(batch_descs, 0)
            
            batch_obj_ids = torch.repeat_interleave(torch.tensor(batch_obj_ids), torch.tensor(num_objs)).unsqueeze(-1)
            connections = torch.eq(batch_obj_ids, batch_obj_ids.t())
            connections = ((~torch.eye(connections.shape[0]).type(torch.bool))*connections).type(torch.float) # Mask Diag self-connections for loss

            # descriptor loss
            ploss, nloss = criterion(batch_descs, connections.to(device))

            loss = ploss * lambda_d + nloss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_iter += 1     

            writer.add_scalar('Train/Loss', loss, sum_iter)
            writer.add_scalar('Train/PLoss', ploss, sum_iter)
            writer.add_scalar('Train/NLoss', nloss, sum_iter)

            if sum_iter % 50 == 0:
                print("sum_iter = {}, loss = {}, ploss = {}, nloss={}".format(sum_iter, loss.item(), ploss.item(), nloss.item()))        

            if sum_iter % checkpoint == 0:
                model_saving_path = os.path.join(save_dir, "netvlad_model_{}.pth".format(sum_iter))
                torch.save(netvlad_model.state_dict(), model_saving_path)
                print("saving model to {}".format(model_saving_path))

    writer.close()

def main():
    parser = argparse.ArgumentParser(description="Training NetVLAD for Object Matching")
    parser.add_argument(
        "-c", "--config_file",
        dest = "config_file",
        type = str, 
        default = ""
    )
    parser.add_argument(
        "-g", "--gpu",
        dest = "gpu",
        type = int, 
        default = 1
    )
    args = parser.parse_args()

    config_file = args.config_file
    f = open(config_file, 'r', encoding='utf-8')
    configs = f.read()
    configs = yaml.safe_load(configs)
    configs['use_gpu'] = args.gpu

    train(configs)

if __name__ == "__main__":
    main()