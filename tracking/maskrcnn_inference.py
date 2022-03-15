import os
import sys
sys.path.append('.')
import argparse
import yaml
from tqdm import tqdm

import torch
import cv2
from model.build_model import build_maskrcnn

from utils.tools import tensor_to_numpy
from datasets.utils import postprocess as post
from datasets.utils.preprocess import preprocess_data

def maskrcnn_inference(model, batch, data_config, save_dir=None):
    with torch.no_grad():
        original_images = batch['image']
        original_images = [tensor_to_numpy(img.clone()) for img in original_images]

        # preprocess
        images, sizes = preprocess_data(batch, data_config)
        original_sizes = sizes['original_sizes']
        new_sizes = sizes['new_sizes']

        # model inference
        _, detections = model(images, sizes) 

        # postprocess
        detections = post.postprocess_detections(new_sizes, original_sizes, detections=detections)

        # save results
        if save_dir is not None:
            image_names = batch['image_name']
            results = post.save_detection_results(image_names, save_dir, detections)

    return results

def validate(configs):
    # read configs
    ## command line config
    use_gpu = configs['use_gpu']
    ## data cofig
    data_config = configs['data']
    ## maskrcnn model config
    maskrcnn_model_config = configs['model']['maskrcnn']
    ## others
    configs['num_gpu'] = [0]
    configs['public_model'] = 0

    # model
    maskrcnn_model = build_maskrcnn(configs)
    maskrcnn_model.eval()

    base_dir = configs['base_dir']

    seg_dir = os.path.join(base_dir, os.path.join('semantic'))
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
            result = maskrcnn_inference(maskrcnn_model, data, data_config, seg_dir)
       

def main():
    parser = argparse.ArgumentParser(description="MaskRCNN Inference")
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