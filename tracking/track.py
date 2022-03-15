import os
import sys
sys.path.append('.')
import copy
import pickle
import yaml
import argparse

import cv2
import torch
import numpy as np
from scipy.spatial import Delaunay
from tqdm import tqdm

from scipy import optimize

from model.build_model import build_airobj
from utils import viz

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

def eval(configs):
    ## config
    base_dir = configs['base_dir']
    ## other
    configs['num_gpu'] = [0]
    configs['public_model'] = 0
    thr = configs['match_threshold']

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    airobj_model = build_airobj(configs)
    airobj_model.eval()

    image_dir = os.path.join(base_dir, 'image_0')
    seg_dir = os.path.join(base_dir, 'semantic')
    points_dir = os.path.join(base_dir, 'sp_0')
    file_names = sorted(os.listdir(image_dir))

    for i in tqdm(range(0, len(file_names))):
        image = cv2.imread(os.path.join(image_dir, file_names[i]))

        points = pickle_read(os.path.join(points_dir, file_names[i].replace('.png', '.pkl')))
        keypoints = points['points']
        descriptors = points['point_descs']
        seg = pickle_read(os.path.join(seg_dir, file_names[i].replace('.png', '.pkl')))

        frame_adjs, frame_points, frame_descs = [], [], []
        ann_masks, ann_boxes = [], []

        for instance in range(seg['masks'].shape[0]):
            mask = seg['masks'][instance, 0, :, :].numpy()

            object_filter = mask[keypoints[:,0].T,keypoints[:,1].T]

            np_obj_pts = keypoints[np.where(object_filter==1)[0]].numpy()
            try:
                tri = Delaunay(np_obj_pts, qhull_options='QJ')
            except:
                continue
            adj = get_adj(np_obj_pts, tri)

            frame_adjs.append(torch.from_numpy(adj).float().to(device))
            frame_points.append(keypoints[np.where(object_filter==1)[0]].float().to(device))
            frame_descs.append(descriptors[np.where(object_filter==1)[0]].float().to(device))
            
            inds = np.array(np.where(mask))
            y1, x1 = np.amin(inds, axis=1)
            y2, x2 = np.amax(inds, axis=1)

            ann_masks.append(mask)
            ann_boxes.append(np.array([x1, y1, x2, y2]))
        
        if len(ann_masks) == 0:
            continue

        frame_instances = np.arange(len(ann_masks))
        
        with torch.no_grad():
            frame_obj_descs = []
            for k in range(len(frame_points)):
                frame_obj_descs.append(airobj_model([frame_points[k]], [frame_descs[k]], [frame_adjs[k]]))
            frame_obj_descs = torch.cat(frame_obj_descs)
        
        if i == 0:
            global_dict = {}
            for inst in range(frame_instances.shape[0]):
                global_dict[inst] = {}
                global_dict[inst]['obj_desc'] = frame_obj_descs[inst].unsqueeze(0)
                global_dict[inst]['points'] = [frame_points[inst]]
                global_dict[inst]['descs'] = [frame_descs[inst]]
                global_dict[inst]['adjs'] = [frame_adjs[inst]]
            max_instances = frame_instances[-1]
            labels = frame_instances
        else:
            global_obj_descs = []
            for key in global_dict.keys():
                global_obj_descs.append(global_dict[key]['obj_desc'])
            global_obj_descs = torch.cat(global_obj_descs)

            distances = torch.matmul(frame_obj_descs, global_obj_descs.t())

            _, labels = optimize.linear_sum_assignment(distances.cpu(), maximize=True)

            filtered_labels = []
            for m in range(distances.shape[0]):
                if m < labels.shape[0] and distances[m, labels[m]] >= thr:
                    label = labels[m]
                    filtered_labels.append(label)
        
                    # Single-frame AirObject - Updating Global Object Dictionary
                    if configs['method'] == 'airobj_single_frame_track':
                        global_dict[label]['obj_desc'] = frame_obj_descs[m].unsqueeze(0)
                        global_dict[label]['points'] = [frame_points[m]]
                        global_dict[label]['descs'] = [frame_descs[m]]
                        global_dict[label]['adjs'] = [frame_adjs[m]]
                    
                    # Temporal AirObject
                    if configs['method'] == 'airobj':
                        global_dict[label]['points'].append(frame_points[m])
                        global_dict[label]['descs'].append(frame_descs[m])
                        global_dict[label]['adjs'].append(frame_adjs[m])

                        with torch.no_grad():
                            global_dict[label]['obj_desc'] = airobj_model(copy.deepcopy(global_dict[label]['points']), 
                                                copy.deepcopy(global_dict[label]['descs']), copy.deepcopy(global_dict[label]['adjs']))
                else:
                    max_instances += 1
                    new_label = max_instances
                    filtered_labels.append(new_label)

                    global_dict[new_label] = {}
                    global_dict[new_label]['obj_desc'] = frame_obj_descs[m].unsqueeze(0)
                    global_dict[new_label]['points'] = [frame_points[m]]
                    global_dict[new_label]['descs'] = [frame_descs[m]]
                    global_dict[new_label]['adjs'] = [frame_adjs[m]]

            filtered_labels = np.array(filtered_labels)
            labels = filtered_labels

        detection = {}
        detection['boxes'] = ann_boxes
        detection['masks'] = ann_masks
        detection['labels'] = torch.from_numpy(labels)
        detection['scores'] = torch.ones(frame_instances.shape[0])
        
        results = viz.save_detection_results([image], [file_names[i]], configs['output_dir'],
                                            [detection], None, [points], True, False)

    if configs['save_video']:
        images = [img for img in sorted(os.listdir(configs['output_dir'])) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(configs['output_dir'], images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(configs['video_path'], cv2.VideoWriter_fourcc(*'mp4v'), 3, (width,height))

        for image in tqdm(images):
            video.write(cv2.imread(os.path.join(configs['output_dir'], image)))

        video.release()


def main():
    parser = argparse.ArgumentParser(description="AirObject-Global-Object-Tracking")
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