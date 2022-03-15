import os
import pickle
import yaml
import argparse

import torch
import numpy as np
from tqdm import tqdm

from model.build_model import build_netvlad, build_seqnet, build_airobj

def get_neighbor(vertex_id,tri):
    # get neighbor vertexes of a vertex
    helper = tri.vertex_neighbor_vertices
    index_pointers = helper[0]
    indices = helper[1]
    result_ids = indices[index_pointers[vertex_id]:index_pointers[vertex_id+1]]

    return result_ids

def get_adj(points, tri):
    adj = np.zeros((points.shape[0], points.shape[0]))

    for i in range(points.shape[0]):
        adj[i,get_neighbor(i,tri)] = 1

    return adj

def pickle_read(path):
    with open(path, 'rb') as fp:
        pickle_file = pickle.load(fp)
    return pickle_file

def pickle_write(path, dump_file):
    with open(path, 'wb') as fp:
        pickle.dump(dump_file, fp)

def get_pr_curve_area(pr_curve):
  '''
  pr_curve: [[p0, r0], [p1, r1]... [pn, rn]], thr: small->big, precision: small->big, recall: big->small
  '''
  area = 0.0
  for i in range(1, len(pr_curve)):
    p0, r0 = pr_curve[i-1]
    p1, r1 = pr_curve[i]

    area = area + (r0 - r1) * (p1 + p0) / 2

  return area

def eval(configs):
    ## command line config
    base_dir = configs['base_dir']
    video_list = configs['video_list']
    ## other
    configs['num_gpu'] = [0]
    configs['public_model'] = 0

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if configs['method'] in ['seqnet']:
        netvlad_model = build_netvlad(configs)
        netvlad_model.eval()
    
    if configs['method'] in ['seqnet']:
        seqnet_model = build_seqnet(configs)
        seqnet_model.eval()
    
    if configs['method'] in ['airobj']:
        airobj_model = build_airobj(configs)
        airobj_model.eval()

    video_objects = pickle_read(configs['video_objects_path'])

    with torch.no_grad():
        for video in tqdm(video_objects.keys()):

            objects = video_objects[video]
            
            for obj_id in objects.keys():

                vid_obj_length = int(len(objects[obj_id]['seq_points']))
                split = int(vid_obj_length/2)

                if split < 2:
                    continue

                # Query
                iter_points, iter_descs, iter_adj = [], [], []

                for l in range(split):
                    iter_points.append(objects[obj_id]['seq_points'][l].detach().clone())
                    iter_descs.append(objects[obj_id]['seq_descs'][l].detach().clone())
                    iter_adj.append(objects[obj_id]['seq_adjs'][l].detach().clone())

                if configs['method'] in ['airobj']:

                    airobj_descs = []
                    idx = np.arange(configs['seqlen'])
                    for start_index in range(0, len(iter_points), configs['seqlen']):
                        valid_idx = idx[(start_index + idx) < len(iter_points)]
                        inds = start_index + valid_idx

                        evol_points = [iter_points[ind] for ind in inds]
                        evol_descs = [iter_descs[ind] for ind in inds]
                        evol_adj = [iter_adj[ind] for ind in inds]
                        
                        airobj_descs.append(airobj_model(evol_points, evol_descs, evol_adj))

                    objects[obj_id]['q_obj_descs'] = torch.cat(airobj_descs)

                elif configs['method'] in ['seqnet']:
                    iter_obj_descs = netvlad_model(iter_descs)

                    evol_descs = []
                    idx = np.arange(configs['seqlen'])
                    for start_index in range(0, iter_obj_descs.shape[0], configs['seqlen']):
                        valid_idx = idx[(start_index + idx) < iter_obj_descs.shape[0]]
                        inds = start_index + valid_idx
                        evol_features = iter_obj_descs[inds].detach().clone()
                        evol_descs.append(seqnet_model.module.pool(evol_features.unsqueeze(0)))

                    objects[obj_id]['q_obj_descs'] = torch.cat(evol_descs)

                # Reference
                iter_points, iter_descs, iter_adj = [], [], []

                for l in range(split, vid_obj_length):
                    iter_points.append(objects[obj_id]['seq_points'][l].detach().clone())
                    iter_descs.append(objects[obj_id]['seq_descs'][l].detach().clone())
                    iter_adj.append(objects[obj_id]['seq_adjs'][l].detach().clone())

                if configs['method'] in ['airobj']:

                    airobj_descs = []
                    idx = np.arange(configs['seqlen'])
                    for start_index in range(0, len(iter_points), configs['seqlen']):
                        valid_idx = idx[(start_index + idx) < len(iter_points)]
                        inds = start_index + valid_idx

                        evol_points = [iter_points[ind] for ind in inds]
                        evol_descs = [iter_descs[ind] for ind in inds]
                        evol_adj = [iter_adj[ind] for ind in inds]
                        
                        airobj_descs.append(airobj_model(evol_points, evol_descs, evol_adj))

                    objects[obj_id]['ref_obj_descs'] = torch.cat(airobj_descs)

                elif configs['method'] in ['seqnet']:
                    iter_obj_descs = netvlad_model(iter_descs)

                    evol_descs = []
                    idx = np.arange(configs['seqlen'])
                    for start_index in range(0, iter_obj_descs.shape[0], configs['seqlen']):
                        valid_idx = idx[(start_index + idx) < iter_obj_descs.shape[0]]
                        inds = start_index + valid_idx
                        evol_features = iter_obj_descs[inds].detach().clone()
                        evol_descs.append(seqnet_model.module.pool(evol_features.unsqueeze(0)))

                    objects[obj_id]['ref_obj_descs'] = torch.cat(evol_descs)
            
            video_objects[video] = objects
    
    thrs = [float(i)/50 for i in range(51)]
    pr_curve = []

    for thr in tqdm(thrs):

        pr_numbers = []

        for video in video_objects.keys():
            
            objects = video_objects[video]

            q_batch_descs, ref_batch_descs = [], []
            
            batch_obj_ids, q_num_objs, ref_num_objs = [], [], []

            for obj_id in objects.keys():
                    
                if 'q_obj_descs' not in objects[obj_id].keys():
                    continue

                q_batch_descs.append(objects[obj_id]['q_obj_descs'].detach().clone())
                ref_batch_descs.append(objects[obj_id]['ref_obj_descs'].detach().clone())
                q_num_objs.append(objects[obj_id]['q_obj_descs'].shape[0])
                ref_num_objs.append(objects[obj_id]['ref_obj_descs'].shape[0])
                
                batch_obj_ids.append(int(obj_id))

            if len(q_batch_descs)==0:
                continue
            
            q_batch_descs = torch.cat(q_batch_descs, 0)
            ref_batch_descs = torch.cat(ref_batch_descs, 0)
                
            q_batch_obj_ids = torch.repeat_interleave(torch.tensor(batch_obj_ids), torch.tensor(q_num_objs)).unsqueeze(-1)
            ref_batch_obj_ids = torch.repeat_interleave(torch.tensor(batch_obj_ids), torch.tensor(ref_num_objs)).unsqueeze(-1)
            connections = torch.eq(q_batch_obj_ids, ref_batch_obj_ids.t())

            distances = torch.matmul(q_batch_descs, ref_batch_descs.t())
            
            match_matrix = (np.around(distances.cpu(), 4) > thr).float() 
            tp = torch.sum(match_matrix * connections.cpu())
            match_num = torch.sum(match_matrix).item()
            gt_num = torch.sum(connections).item()

            pr_number = [tp, match_num, gt_num]
            pr_numbers.append(pr_number)
            
        pr_numbers = torch.tensor(pr_numbers)
        pr_numbers = torch.sum(pr_numbers, 0)

        TP, MatchNum, GTNum = pr_numbers.cpu().numpy().tolist()

        precision = TP / MatchNum if MatchNum > 0 else 1
        recall = TP / GTNum if GTNum > 0 else 1
        pr_curve.append([precision, recall])

    area = get_pr_curve_area(pr_curve)
    print('PR-AUC(%): {:.2f}'.format(area*100))

    results = {}
    results['pr_curve'] = pr_curve
    results['area'] = area

    os.makedirs(configs['save_dir'], exist_ok=True)
    pickle_write(os.path.join(configs['save_dir'], configs['method']+'_'+str(configs['seqlen'])+'_pr_curve.pkl'), results)

def main():
    parser = argparse.ArgumentParser(description="Evaluating")
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

    eval(configs)

if __name__ == "__main__":
    main()