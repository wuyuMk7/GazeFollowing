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

# log setting
log_dir = 'log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = log_dir + 'train.log'

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s',
                    filename=log_file,
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

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


class GazeDataset(Dataset):
    def __init__(self, root_dir, mat_file, training='train'):
        assert (training in set(['train', 'test']))
        self.root_dir = root_dir
        self.mat_file = mat_file
        self.training = training

        anns = loadmat(self.mat_file)
        self.bboxes = anns[self.training + '_bbox']
        self.gazes = anns[self.training + '_gaze']
        self.paths = anns[self.training + '_path']
        self.eyes = anns[self.training + '_eyes']
        self.meta = anns[self.training + '_meta']
        self.image_num = self.paths.shape[0]

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
        #image = image_preprocess(image)
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
        sample = {'image' : image,
                  'face_image': face_image,
                  'eye_position': torch.FloatTensor(eye),
                  'gaze_field': torch.from_numpy(gaze_field),
                  'gt_position': torch.FloatTensor(gaze),
                  'gt_heatmap': torch.FloatTensor(heatmap).unsqueeze(0)}

        return sample

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

            direction, predict_heatmap = net([image, face_image, gaze_field, eye_position])

            heatmap_loss, m_angle_loss = \
                F_loss(direction, predict_heatmap, eye_position, gt_position, gt_heatmap)

            loss = heatmap_loss + m_angle_loss

            '''
            total_loss.append([heatmap_loss.data[0],
                              m_angle_loss.data[0], loss.data[0]])
            logging.info('loss: %.5lf, %.5lf, %.5lf'%( \
                  heatmap_loss.data[0], m_angle_loss.data[0], loss.data[0]))
            '''

            total_loss.append([heatmap_loss.item(),
                               m_angle_loss.item(), loss.item()])
            logging.info('loss: %.5lf, %.5lf, %.5lf' % ( \
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

                norm_m = (m_direction[0] ** 2 + m_direction[1] ** 2) ** 0.5
                norm_f = (f_direction[0] ** 2 + f_direction[1] ** 2) ** 0.5
                norm_gt = (gt_direction[0] ** 2 + gt_direction[1] ** 2) ** 0.5

                m_cos_sim = (m_direction[0] * gt_direction[0] + m_direction[1] * gt_direction[1]) / \
                            (norm_gt * norm_m + 1e-6)
                m_cos_sim = np.maximum(np.minimum(m_cos_sim, 1.0), -1.0)
                m_angle = np.arccos(m_cos_sim) * 180 / np.pi

                f_cos_sim = (f_direction[0] * gt_direction[0] + f_direction[1] * gt_direction[1]) / \
                            (norm_gt * norm_f + 1e-6)
                f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
                f_angle = np.arccos(f_cos_sim) * 180 / np.pi

                total_error.append([f_dist, m_angle, f_angle])
                info_list.append(list(f_point))
    info_list = np.array(info_list)
    # np.savez('multi_scale_concat_prediction.npz', info_list=info_list)

    # heatmaps = np.stack(heatmaps)
    # np.savez('multi_scale_concat_heatmaps.npz', heatmaps=heatmaps)

    logging.info('average loss : %s' % str(np.mean(np.array(total_loss), axis=0)))
    logging.info('average error: %s' % str(np.mean(np.array(total_error), axis=0)))

    net.train()
    return


def main():
    '''
    train_set = GazeDataset(root_dir='../../data/',
                            mat_file='../../data/train_annotations.mat',
                            training='train')
    train_data_loader = DataLoader(train_set, batch_size=48,
                                   shuffle=True, num_workers=8)

    test_set = GazeDataset(root_dir='../../test_data/',
                           mat_file='../../test_data/test2_annotations.mat',
                           training='test')
    test_data_loader = DataLoader(test_set, batch_size=32,
                                  shuffle=False, num_workers=8)
    '''

    dis_train_sets = dataset_wrapper(root_dir='../../data/',
                                     mat_file='../../data/train_annotations.mat',
                                     training='train')
    #dis_train_data_loader = DataLoader(dis_train_sets[0], batch_size=48,
    #                                   shuffle=True, num_workers=8)

    dis_test_sets = dataset_wrapper(root_dir='../../test_data/',
                                    mat_file='../../test_data/test2_annotations.mat',
                                    training='test')
    #dis_test_data_loader = DataLoader(dis_test_sets[0], batch_size=32,
    #                                  shuffle=False, num_workers=8)

    dis_train_data_loaders, dis_test_data_loaders = [], []
    for i in range(16):
        dis_train_data_loaders.append(DataLoader(dis_train_sets[i], batch_size=48,
                                      shuffle=True, num_workers=8))
        dis_test_data_loaders.append(DataLoader(dis_test_sets[i], batch_size=32,
                                     shuffle=False, num_workers=1))

    net = GazeNet()
    net = DataParallel(net)
    net.cuda()

    
    #print(next(net.module.fpn_net.parameters()).is_cuda)
    ##print(next(net.module.fpn_net.parameters()).is_cuda)
    area_count = 8
    area_in_network = int(16/area_count)
    cur_area_idx = 0
    fpn_weights_transferred = False
    for i in range(area_count):
        net.module.change_fpn(i)
        if not next(net.module.fpn_net.parameters()).is_cuda:
            net.module.fpn_net.cuda()
    net.module.change_fpn(cur_area_idx)
    ##print(next(net.module.fpn_net.parameters()).is_cuda)
    #exit(0)

    resume_training = False
    if resume_training:
        pretrained_dict = torch.load('../model/pretrained_model.pkl')
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        test(net, test_data_loader)
        exit()

    method = 'Adam'
    learning_rate = 0.0001

    optimizer_s1 = optim.Adam([{'params': net.module.face_net.parameters(),
                                'initial_lr': learning_rate},
                               {'params': net.module.face_process.parameters(),
                                'initial_lr': learning_rate},
                               {'params': net.module.eye_position_transform.parameters(),
                                'initial_lr': learning_rate},
                               {'params': net.module.fusion.parameters(),
                                'initial_lr': learning_rate}],
                              lr=learning_rate, weight_decay=0.0001)
    #optimizer_s2 = optim.Adam([{'params': net.module.fpn_net.parameters(),
    #                            'initial_lr': learning_rate}],
    #                          lr=learning_rate, weight_decay=0.0001)
    optimizer_s2s, optimizer_s3s = [], []
    for i in range(area_count):
        net.module.change_fpn(i)
        optimizer_s2 = optim.Adam([{'params': net.module.fpn_nets[i].parameters(),
                                    'initial_lr': learning_rate}],
                                    lr=learning_rate, weight_decay=0.0001)
        optimizer_s3 = optim.Adam([{'params': net.parameters(), 
                                    'initial_lr': learning_rate}],
                                    lr=learning_rate * 0.1, weight_decay=0.0001)
        optimizer_s2s.append(optimizer_s2)
        optimizer_s3s.append(optimizer_s3)
    optimizer_s2 = optimizer_s2s[0]
    optimizer_s3 = optimizer_s3s[0]

    lr_scheduler_s1 = optim.lr_scheduler.StepLR(optimizer_s1, step_size=5, gamma=0.1, last_epoch=-1)
    #lr_scheduler_s2 = optim.lr_scheduler.StepLR(optimizer_s2, step_size=5, gamma=0.1, last_epoch=-1)
    lr_scheduler_s2s, lr_scheduler_s3s = [], []
    for i in range(area_count):
        lr_scheduler_s2 = optim.lr_scheduler.StepLR(optimizer_s2s[i], step_size=5, 
                gamma=0.1, last_epoch=-1)
        lr_scheduler_s3 = optim.lr_scheduler.StepLR(optimizer_s3s[i], step_size=5, 
                gamma=0.1, last_epoch=-1)
        lr_scheduler_s2s.append(lr_scheduler_s2)
        lr_scheduler_s3s.append(lr_scheduler_s3)
    lr_scheduler_s2 = lr_scheduler_s2s[0]
    lr_scheduler_s3 = lr_scheduler_s3s[0]

    # Set the model to use the first FPN
    net.module.change_fpn(cur_area_idx)

    max_epoch = 20

    epoch = 0
    #epoch = 7
    while epoch < max_epoch:
        logging.info('\n--- Epoch: %s\n' % str(epoch))
        if epoch == 0:
            lr_scheduler = lr_scheduler_s1
            optimizer = optimizer_s1
        elif epoch == 7:
            lr_scheduler = lr_scheduler_s2
            optimizer = optimizer_s2
        elif epoch == 15:
            lr_scheduler = lr_scheduler_s3
            optimizer = optimizer_s3

        #lr_scheduler.step()
        #lr_scheduler.step()

        running_loss = []
        
        #for data_loader_idx in range(len(dis_train_data_loaders)):
        for data_loader_idx in range(len(dis_train_data_loaders)):
            train_data_loader = dis_train_data_loaders[data_loader_idx]

            if epoch >= 10:
            #if epoch >= 7:
                #if not fpn_weights_transferred:
                #    net.module.transfer_fpn_weights()
                #    fpn_weights_transferred = True

                area_idx = int(data_loader_idx/area_in_network)
                if cur_area_idx != area_idx:
                    cur_area_idx = area_idx
                    net.module.change_fpn(cur_area_idx)
                    if epoch < 15:
                        lr_scheduler = lr_scheduler_s2s[cur_area_idx]
                        optimizer = optimizer_s2s[cur_area_idx]
                    else:
                        lr_scheduler = lr_scheduler_s3s[cur_area_idx]
                        optimizer = optimizer_s3s[cur_area_idx]

            #if not next(net.module.fpn_net.parameters()).is_cuda:
            #    net.module.fpn_net.cuda()

            #test_data_loader = dis_test_data_loaders[data_loader_idx]
            #train_data_loader = DataLoader(dis_train_sets[data_loader_idx], batch_size=48,
            #                              shuffle=True, num_workers=2)
            #test_data_loaders = DataLoader(dis_test_sets[data_loader_idx], batch_size=32,
            #                              shuffle=False, num_workers=2)

            for i, data in tqdm(enumerate(train_data_loader)):
                image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
                    data['image'], data['face_image'], data['gaze_field'], data['eye_position'], data['gt_position'], data['gt_heatmap']
                image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
                    map(lambda x: x.cuda(), [image, face_image, gaze_field, eye_position, gt_position, gt_heatmap])
                # for var in [image, face_image, gaze_field, eye_position, gt_position]:
                #    print var.shape

                optimizer.zero_grad()

                direction, predict_heatmap = net([image, face_image, gaze_field, eye_position])

                heatmap_loss, m_angle_loss = \
                    F_loss(direction, predict_heatmap, eye_position, gt_position, gt_heatmap)

                if epoch == 0:
                    loss = m_angle_loss
                elif epoch >= 7 and epoch <= 14:
                    loss = heatmap_loss
                else:
                    loss = m_angle_loss + heatmap_loss

                loss.backward()
                optimizer.step()

                # running_loss.append([heatmap_loss.data[0],
                #                     m_angle_loss.data[0], loss.data[0]])
                running_loss.append([heatmap_loss.item(),
                                     m_angle_loss.item(), loss.item()])
                if i % 10 == 9:
                    logging.info('%s %s %s' % (str(np.mean(running_loss, axis=0)), method, str(lr_scheduler.get_last_lr())))
                    running_loss = []

        lr_scheduler.step()
        epoch += 1

        save_path = '../model/two_stage_fpn_concat_multi_scale_' + method
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(net.state_dict(), save_path + '/model_epoch{}.pkl'.format(epoch))

        for i in range(16):
            torch.save(net.module.fpn_nets[i], save_path + '/fpn_{}.pkl'.format(i))

        for data_loader_idx in range(len(dis_test_data_loaders)):
            test_data_loader = dis_test_data_loaders[data_loader_idx]
            if epoch >= 10:
                area_idx = int(data_loader_idx/area_in_network)
                net.module.change_fpn(area_idx)
            test(net, test_data_loader)


if __name__ == '__main__':
    main()

