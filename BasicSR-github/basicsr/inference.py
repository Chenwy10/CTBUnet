import logging
import torch
from os import path as osp
import os 
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor
import numpy as np
import cv2 

import pdb

def inference_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)
    
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create model
    model = build_model(opt)
    
    test_set_name = opt['datasets']['test']['name']
    dataroot = opt['datasets']['test']['dataroot_input']
    logger.info(f'Inferecing {test_set_name}...')
    imageroot = osp.join(root_path, dataroot)
    image_files = os.listdir(imageroot)
    
    #pdb.set_trace()
    for image_file in image_files:
        image_path = os.path.join(imageroot, image_file)
        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            image = image.astype(np.float32) / 255.0
            image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
            output_image_tensor = model.inference(image_tensor.cuda())
            output_image_tensor = output_image_tensor.permute(0, 2, 3, 1).squeeze(0)
            output_iamge = output_image_tensor.detach().cpu().numpy()*255.0
            #pdb.set_trace()
            output_iamge = cv2.cvtColor(output_iamge, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(opt['path']['visualization'],image_file), output_iamge.astype(np.int))

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    inference_pipeline(root_path)