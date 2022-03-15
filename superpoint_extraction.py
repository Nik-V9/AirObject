import os
import argparse
import yaml
from tqdm import tqdm

import torch
import cv2
from model.build_model import build_superpoint_model
from model.inference import superpoint_inference
from cocoapi.PythonAPI.pycocotools.ytvos import YTVOS

def find(lst, key, value):
    ind = []
    id = []
    for i, dic in enumerate(lst):
        if value in dic[key][0]:
            ind.append(i)
            id.append(lst[i]['id'])
    return ind, id

def inference(configs):
    ## data cofig
    data_config = configs['data']
    ## superpoint model config
    superpoint_model_config = configs['model']['superpoint']
    detection_threshold = superpoint_model_config['detection_threshold']
    ## others
    configs['num_gpu'] = [0]
    configs['public_model'] = 0

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_dir = configs['img_data_path']
    base_points_dir = configs['points_dir']

    # model
    superpoint_model = build_superpoint_model(configs)

    # YouTube-VIS
    base_dir = configs['img_data_path']
    ovis = YTVOS(configs['annotations_path'])

    with open(configs['video_list_path'], "r") as f:
        video_list = f.read().split("\n")

    for video_id in tqdm(video_list):
        points_dir = os.path.join(base_points_dir, video_id)

        # Identfy video id
        vid_ind, vid_id = find(ovis.dataset['videos'], 'file_names', video_id)
        image_paths = ovis.dataset['videos'][vid_ind[0]]['file_names']

        for i in range(len(image_paths)):
            data_path = os.path.join(base_dir, image_paths[i])
            file_name = os.path.splitext(os.path.basename(image_paths[i]))[0]

            data = {}
            src = cv2.imread(data_path)
            image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            image = cv2.merge([image, image, image])
            image = torch.from_numpy(image).type(torch.float32).to(device)
            image = image.permute(2,0,1)
            image /= 255
            data['image'] = [image]
            data['image_name'] = [str(file_name)]

            with torch.no_grad():
                result = superpoint_inference(superpoint_model, data, data_config, detection_threshold, points_dir)

def main():
    parser = argparse.ArgumentParser(description="SuperPoint Feature Extraction")
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

    inference(configs)

if __name__ == "__main__":
    main()