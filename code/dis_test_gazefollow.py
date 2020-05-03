import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from gazenet import GazeNet

import time
import os
import numpy as np
import json
import cv2
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import operator
import itertools
from scipy.io import  loadmat
import logging

from scipy import signal

from utils import data_transforms
from utils import get_paste_kernel, kernel_map

from dis_utils import dataset_wrapper, load_pretrained_model

# log setting
log_dir = 'log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = log_dir + 'test_gazefollowdata.log'

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s',
                    filename=log_file,
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

cosine_similarity = nn.CosineSimilarity()
mse_distance = nn.MSELoss()
bce_loss = nn.BCELoss()


def F_loss(direction, predict_heatmap, eye_position, gt_position, gt_heatmap):
    # point loss
    heatmap_loss = bce_loss(predict_heatmap, gt_heatmap)

    # angle loss
    gt_direction = gt_position - eye_position
    middle_angle_loss = torch.mean(1 - cosine_similarity(direction, gt_direction))

    return heatmap_loss, middle_angle_loss

def test(net, test_data_loader):
    net.eval()
    total_loss = []
    total_error = []
    info_list = []
    heatmaps = []

    with torch.no_grad():
        for data in test_data_loader:
            image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
                data['image'], data['face_image'], data['gaze_field'], data['eye_position'], data['gt_position'], data['gt_heatmap']
            image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
                map(lambda x: x.cuda(), [image, face_image, gaze_field, eye_position, gt_position, gt_heatmap])
                #map(lambda x: Variable(x.cuda(), volatile=True), [image, face_image, gaze_field, eye_position, gt_position, gt_heatmap])

            direction, predict_heatmap = net([image, face_image, gaze_field, eye_position])

            heatmap_loss, m_angle_loss = \
                F_loss(direction, predict_heatmap, eye_position, gt_position, gt_heatmap)

            loss = heatmap_loss + m_angle_loss


            total_loss.append([heatmap_loss.item(),
                              m_angle_loss.item(), loss.item()])
            logging.info('loss: %.5lf, %.5lf, %.5lf'%( \
                  heatmap_loss.item(), m_angle_loss.item(), loss.item()))

            middle_output = direction.cpu().data.numpy()
            final_output = predict_heatmap.cpu().data.numpy()
            target = gt_position.cpu().data.numpy()
            eye_position = eye_position.cpu().data.numpy()
            for m_direction, f_point, gt_point, eye_point in \
                zip(middle_output, final_output, target, eye_position):
                f_point = f_point.reshape([224 // 4, 224 // 4])
                heatmaps.append(f_point)

                h_index, w_index = np.unravel_index(f_point.argmax(), f_point.shape)
                f_point = np.array([w_index / 56., h_index / 56.])

                f_error = f_point - gt_point
                f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)

                # angle 
                f_direction = f_point - eye_point
                gt_direction = gt_point - eye_point

                norm_m = (m_direction[0] **2 + m_direction[1] ** 2 ) ** 0.5
                norm_f = (f_direction[0] **2 + f_direction[1] ** 2 ) ** 0.5
                norm_gt = (gt_direction[0] **2 + gt_direction[1] ** 2 ) ** 0.5
                
                m_cos_sim = (m_direction[0]*gt_direction[0] + m_direction[1]*gt_direction[1]) / \
                            (norm_gt * norm_m + 1e-6)
                m_cos_sim = np.maximum(np.minimum(m_cos_sim, 1.0), -1.0)
                m_angle = np.arccos(m_cos_sim) * 180 / np.pi

                f_cos_sim = (f_direction[0]*gt_direction[0] + f_direction[1]*gt_direction[1]) / \
                            (norm_gt * norm_f + 1e-6)
                f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
                f_angle = np.arccos(f_cos_sim) * 180 / np.pi

                
                total_error.append([f_dist, m_angle, f_angle])
                info_list.append(list(f_point))
    info_list = np.array(info_list)
    #np.savez('multi_scale_concat_prediction.npz', info_list=info_list)

    heatmaps = np.stack(heatmaps)
    #np.savez('multi_scale_concat_heatmaps.npz', heatmaps=heatmaps)

    logging.info('average loss : %s'%str(np.mean(np.array(total_loss), axis=0)))
    logging.info('average error [mean dist, angle, mean angle]: %s'%str(np.mean(np.array(total_error), axis=0)))
    
    net.train()
    return np.mean(np.array(total_loss), axis=0), np.mean(np.array(total_error), axis=0), info_list, heatmaps


def main():
    dis_test_sets = dataset_wrapper(root_dir='../../test_data/',
                                    mat_file='../../test_data/test2_annotations.mat',
                                    training='test')

    net = GazeNet()
    net = DataParallel(net)
    net.cuda()

    net = load_pretrained_model(net, '../model/pretrained_models/')

    area_count = 8
    area_in_network = 2

    dis_test_data_loaders = []
    all_losses, all_errors = [], []
    info_lists, heatmaps = [], []
    for i in range(16):
        test_data_loader = DataLoader(dis_test_sets[i], batch_size=1,
                                      shuffle=False, num_workers=8)

        area_idx = int(i/area_in_network)
        net.module.change_fpn(area_idx)
        if not next(net.module.fpn_net.parameters()).is_cuda:
            net.module.fpn_net.cuda()

        cur_loss, cur_error, info_list, heatmap = test(net, test_data_loader)
        all_losses.append(cur_loss)
        all_errors.append(cur_error)

        np.savez('../npzs/multi_scale_concat_prediction_{}.npz'.format(str(i)), info_list=info_list)
        np.savez('../npzs/multi_scale_concat_heatmaps_{}.npz'.format(str(i)), heatmaps=heatmap)
        for info in info_list:
            info_lists.append(info)
        for cur_heatmap in heatmap:
            heatmaps.append(cur_heatmap)

    print(np.mean(all_losses, axis=0))
    print(np.mean(all_errors, axis=0))

    info_lists, heatmaps = np.array(info_lists), np.array(heatmaps)
    np.savez('../npzs/multi_scale_concat_prediction.npz', info_list=info_lists)
    np.savez('../npzs/multi_scale_concat_heatmaps.npz', heatmaps=heatmaps)

if __name__ == '__main__':
    main()

