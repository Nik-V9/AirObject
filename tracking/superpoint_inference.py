import os
import sys
sys.path.append('.')
import argparse
import yaml
from tqdm import tqdm

import torch
import cv2
from model.build_model import build_superpoint_model
from model.inference import superpoint_inference

def validate(configs):
    # read configs
    ## command line config
    use_gpu = configs['use_gpu']
    ## data cofig
    data_config = configs['data']
    ## superpoint model config
    superpoint_model_config = configs['model']['superpoint']
    detection_threshold = superpoint_model_config['detection_threshold']
    ## others
    configs['num_gpu'] = [0]
    configs['public_model'] = 0

    # model
    superpoint_model = build_superpoint_model(configs)

    base_dir = configs['base_dir']

    points_dir = os.path.join(base_dir, os.path.join('sp_0'))
    video_dir = os.path.join(base_dir, os.path.join('image_0'))
    image_paths = os.listdir(video_dir)

    for i in tqdm(range(len(image_paths))):

        if not image_paths[i].endswith(".png"):
            continue
        
        data_path = os.path.join(video_dir, image_paths[i])
        file_name = os.path.basename(image_paths[i]).replace('.png','')

        data = {}
        src = cv2.imread(data_path)
        image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        image = cv2.merge([image, image, image])
        image = torch.from_numpy(image).type(torch.float32)
        image = image.permute(2,0,1)
        image /= 255
        data['image'] = [image]
        data['image_name'] = [str(file_name)]

        with torch.no_grad():
            result = superpoint_inference(superpoint_model, data, data_config, detection_threshold, points_dir)
       

def main():
    parser = argparse.ArgumentParser(description="SuperPoint Inference")
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

    validate(configs)

if __name__ == "__main__":
    main()