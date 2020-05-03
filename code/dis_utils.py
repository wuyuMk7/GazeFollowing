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
from scipy.io import loadmat
import logging

from scipy import signal

from utils import data_transforms
from utils import get_paste_kernel, kernel_map

class FilteredGazeDataset(Dataset):
    def __init__(self, filtered_data, root_dir, mat_file, training='train'):
        assert (training in set(['train', 'test']))
        self.root_dir = root_dir
        self.mat_file = mat_file
        self.training = training

        #anns = loadmat(self.mat_file)
        self.bboxes = filtered_data[self.training + '_bbox']
        self.gazes = filtered_data[self.training + '_gaze']
        self.paths = filtered_data[self.training + '_path']
        self.eyes = filtered_data[self.training + '_eyes']
        self.meta = filtered_data[self.training + '_meta']
        self.image_num = self.paths.shape[0]

        #print(self.bboxes.shape, self.gazes.shape, self.paths.shape, self.eyes.shape, self.meta.shape, self.image_num)

        logging.info('%s contains %d images' % (self.mat_file, self.image_num))

    def generate_data_field(self, eye_point):
        """eye_point is (x, y) and between 0 and 1"""
        height, width = 224, 224
        x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
        y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
        grid = np.stack((x_grid, y_grid)).astype(np.float32)

        x, y = eye_point
        x, y = x * width, y * height

        grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
        norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
        # avoid zero norm
        norm = np.maximum(norm, 0.1)
        grid /= norm
        return grid

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        image_path = self.paths[idx][0][0]
        image_path = os.path.join(self.root_dir, image_path)
        box = self.bboxes[0, idx][0]
        eye = self.eyes[0, idx][0]
        # todo: process gaze differently for training or testing
        gaze = self.gazes[0, idx].mean(axis=0)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if random.random() > 0.5 and self.training == 'train':
            eye = [1.0 - eye[0], eye[1]]
            gaze = [1.0 - gaze[0], gaze[1]]
            image = cv2.flip(image, 1)

        # crop face
        x_c, y_c = eye
        x_0 = x_c - 0.15
        y_0 = y_c - 0.15
        x_1 = x_c + 0.15
        y_1 = y_c + 0.15
        if x_0 < 0:
            x_0 = 0
        if y_0 < 0:
            y_0 = 0
        if x_1 > 1:
            x_1 = 1
        if y_1 > 1:
            y_1 = 1
        h, w = image.shape[:2]
        face_image = image[int(y_0 * h):int(y_1 * h), int(x_0 * w):int(x_1 * w), :]
        # process face_image for face net
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)
        face_image = data_transforms[self.training](face_image)
        # process image for saliency net
        # image = image_preprocess(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = data_transforms[self.training](image)

        # generate gaze field
        gaze_field = self.generate_data_field(eye_point=eye)
        # generate heatmap
        heatmap = get_paste_kernel((224 // 4, 224 // 4), gaze, kernel_map, (224 // 4, 224 // 4))
        '''
        direction = gaze - eye
        norm = (direction[0] ** 2.0 + direction[1] ** 2.0) ** 0.5
        if norm <= 0.0:
            norm = 1.0

        direction = direction / norm
        '''
        sample = {'image': image,
                  'face_image': face_image,
                  'eye_position': torch.FloatTensor(eye),
                  'gaze_field': torch.from_numpy(gaze_field),
                  'gt_position': torch.FloatTensor(gaze),
                  'gt_heatmap': torch.FloatTensor(heatmap).unsqueeze(0)}

        return sample

def convert_list(src_list, tar_shape, dtype=object):
    arr = np.empty(len(src_list), dtype=dtype)
    for i in range(len(src_list)):
        arr[i] = src_list[i]
    return arr.reshape(tar_shape)

def dataset_wrapper(root_dir, mat_file, training='train'):
    assert (training in set(['train', 'test']))
    mat = loadmat(mat_file)
    total_image_num = mat[training + '_path'].shape[0]
    #print(total_image_num)

    key_bbox = training + '_bbox'
    key_gaze = training + '_gaze'
    key_path = training + '_path'
    key_eyes = training + '_eyes'
    key_meta = training + '_meta'

    type_bbox = mat[key_bbox].dtype

    wrapped_set = []
    for i in range(16):
        sub_dataset = {
            key_bbox: [],
            key_gaze: [],
            key_path: [],
            key_eyes: [],
            key_meta: []
        }
        wrapped_set.append(sub_dataset)

    for i in range(total_image_num):
        eye_x, eye_y = mat[training + '_eyes'][0][i][0][0], mat[training + '_eyes'][0][i][0][1]
        wrapper_r, wrapper_c = int(eye_y / 0.25), int(eye_x / 0.25)
        wrapper_index = wrapper_r * 4 + wrapper_c

        wrapped_set[wrapper_index][key_bbox].append(mat[key_bbox][0][i])
        wrapped_set[wrapper_index][key_gaze].append(mat[key_gaze][0][i])
        wrapped_set[wrapper_index][key_path].append(mat[key_path][i][0])
        wrapped_set[wrapper_index][key_eyes].append(mat[key_eyes][0][i])
        wrapped_set[wrapper_index][key_meta].append(mat[key_meta][i][0])

    ret_dataset = []
    for i in range(16):
        sub_dataset = wrapped_set[i]

        wrapped_set[i][key_bbox] = convert_list(sub_dataset[key_bbox], (1, len(sub_dataset[key_bbox])))
        wrapped_set[i][key_gaze] = convert_list(sub_dataset[key_gaze], (1, len(sub_dataset[key_gaze])))
        wrapped_set[i][key_path] = convert_list(sub_dataset[key_path], (len(sub_dataset[key_path]), 1))
        wrapped_set[i][key_eyes] = convert_list(sub_dataset[key_eyes], (1, len(sub_dataset[key_eyes])))
        wrapped_set[i][key_meta] = convert_list(sub_dataset[key_meta], (len(sub_dataset[key_meta]), 1))

        # Create GazeDataSet Here
        # 16 dataset
        ret_dataset.append(FilteredGazeDataset(sub_dataset, root_dir, mat_file, training))

    return ret_dataset

def load_pretrained_model(net, root_dir):
    main_model = '{}pretrain.pkl'.format(root_dir)
    
    pretrained_dict = torch.load(main_model)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    for i in range(8):
        fpn_model = '{}fpn_{}.pkl'.format(root_dir, str(i))

        fpn_pretrained_dict = torch.load(fpn_model)
        fpn_model_dict = net.module.fpn_nets[i].state_dict()
        fpn_pretrained_dict = {k: v for k, v in fpn_pretrained_dict.items() if k in fpn_model_dict}
        fpn_model_dict.update(fpn_pretrained_dict)
        net.module.fpn_nets[i].load_state_dict(fpn_model_dict)

    return net
    
