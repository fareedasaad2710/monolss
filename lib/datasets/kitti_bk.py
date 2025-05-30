import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.utils import get_angle_from_box3d,check_range
from lib.datasets.kitti_utils import get_objects_from_label
from lib.datasets.kitti_utils import Calibration
from lib.datasets.kitti_utils import get_affine_transform
from lib.datasets.kitti_utils import affine_transform
from lib.datasets.kitti_utils import compute_box_3d
import pdb

import cv2 as cv
import torchvision.ops.roi_align as roi_align
import math
from lib.datasets.kitti_utils import Object3d



class KITTI(data.Dataset):
    def __init__(self, root_dir, split, cfg):
        # basic configuration
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = cfg['use_3d_center']
        self.writelist = cfg['writelist']
        if cfg['class_merging']:
            self.writelist.extend(['Van', 'Truck'])
        if cfg['use_dontcare']:
            self.writelist.extend(['DontCare'])
        '''    
        ['Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
         'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
         'Cyclist': np.array([1.76282397,0.59706367,1.73698127])] 
        ''' 
        ##l,w,h
        self.cls_mean_size = np.array([[1.76255119    ,0.66068622   , 0.84422524   ],
                                       [1.52563191462 ,1.62856739989, 3.88311640418],
                                       [1.73698127    ,0.59706367   , 1.76282397   ]])                              
                              
        # data split loading
        assert split in ['train', 'val', 'trainval', 'test']
        self.split = split
        split_dir = os.path.join(root_dir, cfg['data_dir'], 'ImageSets', split + '.txt')
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # path configuration
        self.data_dir = os.path.join(root_dir, cfg['data_dir'], 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.depth_dir = os.path.join(self.data_dir, 'depth')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')
        self.dense_depth_dir = cfg['dense_depth_dir']
        self.mask_np_dir = './lib/dense_mask_gt_avg.npy'
        self.mask_np =  np.load(self.mask_np_dir)
        # if not os.path.exists(self.mask_np_dir):
        #     print('warninginging!!!!!!mask_np_dir is not exist')
        #     exit()

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 4

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)    # (H, W, 3) RGB mode


    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)
    
    def get_mask_np(self,idx):
        mask_np_file = os.path.join(self.mask_np_dir, '%06d.npy' % idx)
        if not os.path.exists(mask_np_file):
            # print('mask_np_file is not exist')
            return np.ones((1,7,7))
        else:
            np_file = np.load(mask_np_file)
            return np_file





    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        img = self.get_image(index)
        img_size = np.array(img.size)

        

        if self.split!='test':
            d = cv.imread('{}/{:0>6}.png'.format(self.dense_depth_dir, index), -1) / 256.
            dst_W, dst_H = img_size
            pad_h, pad_w = dst_H - d.shape[0], (dst_W - d.shape[1]) // 2
            pad_wr = dst_W - pad_w - d.shape[1]
            d = np.pad(d, ((pad_h, 0), (pad_w, pad_wr)), mode='edge')
            d = Image.fromarray(d)


        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
        random_crop_flag, random_flip_flag = False, False
        random_mix_flag = False
        calib = self.get_calib(index)

        if self.data_augmentation:
            if np.random.random() < 0.5:
                random_mix_flag = True
                
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                d = d.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random() < self.random_crop:
                crop_size = img_size * np.clip(np.random.randn()*self.scale + 1, 1 - self.scale, 1 + self.scale)
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                
        if random_mix_flag == True:
            count_num = 0
            random_mix_flag = False
            while count_num < 50:
                count_num += 1
                random_index = np.random.randint(len(self.idx_list))
                calib_temp = self.get_calib(random_index)
                
                if calib_temp.cu == calib.cu and calib_temp.cv == calib.cv and calib_temp.fu == calib.fu and calib_temp.fv == calib.fv:
                    img_temp = self.get_image(random_index)
                    img_size_temp = np.array(img.size)
                    dst_W_temp, dst_H_temp = img_size_temp
                    if dst_W_temp == dst_W and dst_H_temp == dst_H:
                        objects_1 = self.get_label(index)
                        objects_2 = self.get_label(random_index)
                        if len(objects_1) + len(objects_2) < self.max_objs: 
                            d_temp = cv.imread('{}/{:0>6}.png'.format(self.dense_depth_dir, random_index), -1) / 256.
                            d_temp = np.pad(d_temp, ((pad_h, 0), (pad_w, pad_wr)), mode='edge')
                            d_temp = Image.fromarray(d_temp)
                            random_mix_flag = True
                            if random_flip_flag == True:
                                img_temp = img_temp.transpose(Image.FLIP_LEFT_RIGHT)
                                d_temp = d_temp.transpose(Image.FLIP_LEFT_RIGHT)
                            img_blend = Image.blend(img, img_temp, alpha=0.5)
                            img = img_blend
                            break

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        if self.split!='test':
            d_trans = d.transform(tuple(self.resolution.tolist()),
                                method=Image.AFFINE,
                                data=tuple(trans_inv.reshape(-1).tolist()),
                                resample=Image.BILINEAR)
            d_trans = np.array(d_trans)
            down_d_trans = cv.resize(d_trans, (self.resolution[0]//self.downsample, self.resolution[1]//self.downsample),
                            interpolation=cv.INTER_AREA)
        
            if random_mix_flag == True:
                d_trans_temp = d_temp.transform(tuple(self.resolution.tolist()),
                                method=Image.AFFINE,
                                data=tuple(trans_inv.reshape(-1).tolist()),
                                resample=Image.BILINEAR)
                d_trans_temp = np.array(d_trans_temp)
                
                down_d_trans_temp = cv.resize(d_trans_temp, (self.resolution[0]//self.downsample, self.resolution[1]//self.downsample),
                            interpolation=cv.INTER_AREA)

        coord_range = np.array([center-crop_size/2,center+crop_size/2]).astype(np.float32)
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W


        features_size = self.resolution // self.downsample# W * H
        #  ============================   get labels   ==============================
        if self.split!='test':
            objects = self.get_label(index)
            # data augmentation for labels
            if random_flip_flag:
                calib.flip(img_size)
                for object in objects:
                    [x1, _, x2, _] = object.box2d
                    object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                    object.ry = np.pi - object.ry
                    object.pos[0] *= -1
                    if object.ry > np.pi:  object.ry -= 2 * np.pi
                    if object.ry < -np.pi: object.ry += 2 * np.pi
            # labels encoding
            heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32) # C * H * W
            size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            depth = np.zeros((self.max_objs, 1), dtype=np.float32)
            heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
            heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
            src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
            height2d = np.zeros((self.max_objs, 1), dtype=np.float32)
            cls_ids = np.zeros((self.max_objs), dtype=np.int64)
            indices = np.zeros((self.max_objs), dtype=np.int64)
            mask_2d = np.zeros((self.max_objs), dtype=np.bool_)
            object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

            vis_depth = np.zeros((self.max_objs, 7, 7), dtype=np.float32)
            att_depth = np.zeros((self.max_objs, 7, 7), dtype=np.float32)
            depth_mask = np.zeros((self.max_objs, 7, 7), dtype=np.bool)


            count = 0
            for i in range(object_num):
                # filter objects by writelist
                if objects[i].cls_type not in self.writelist:
                    continue
    
                # filter inappropriate samples by difficulty
                if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                    continue

                # process 2d bbox & get 2d center
                bbox_2d = objects[i].box2d.copy()
                # add affine transformation for 2d boxes.
                bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                # modify the 2d bbox according to pre-compute downsample ratio
                bbox_2d[:] /= self.downsample


                # process 3d bbox & get 3d center
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
                center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                center_3d = center_3d[0]  # shape adjustment
                center_3d = affine_transform(center_3d.reshape(-1), trans)
                center_3d /= self.downsample

                # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
                if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
                if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue
    
                # generate the radius of gaussian heatmap
                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                radius = gaussian_radius((w, h))
                radius = max(0, int(radius))
    
                if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
                    draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                    continue
    
                cls_id = self.cls2id[objects[i].cls_type]
                cls_ids[i] = cls_id
                draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)
    
                # encoding 2d/3d offset & 2d size
                indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
                offset_2d[i] = center_2d - center_heatmap
                size_2d[i] = 1. * w, 1. * h
    
                # encoding depth
                depth[i] = objects[i].pos[-1]
    
                # encoding heading angle
                #heading_angle = objects[i].alpha
                heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0]+objects[i].box2d[2])/2)
                if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                if heading_angle < -np.pi: heading_angle += 2 * np.pi
                heading_bin[i], heading_res[i] = angle2class(heading_angle)

                offset_3d[i] = center_3d - center_heatmap
                src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                size_3d[i] = src_size_3d[i] - mean_size

                #objects[i].trucation <=0.5 and objects[i].occlusion<=2 and (objects[i].box2d[3]-objects[i].box2d[1])>=25:
                if objects[i].trucation <=0.5 and objects[i].occlusion<=2:
                    mask_2d[i] = 1

                roi_depth = roi_align(torch.from_numpy(down_d_trans).unsqueeze(0).unsqueeze(0).type(torch.float32),
                                      [torch.tensor(bbox_2d).unsqueeze(0)], [7, 7]).numpy()[0, 0]
                # maintain interested points
                # roi_depth_ind = (roi_depth > depth[i] - 3) & \
                #                 (roi_depth < depth[i] + 3) & \
                #                 (roi_depth > 0)
                roi_depth_ind = (roi_depth != -100000)
                roi_depth[~roi_depth_ind] = 0
                vis_depth[i] = depth[i]
                # att_depth[i] = depth[i] - vis_depth[i]
                depth_mask[i] = roi_depth_ind
                # print("depth_mask[i] : ", depth_mask[i])
                # avg_depth_mask[i] = self.mask_np
                # if self.split=='train' or self.split=='trainval':
                    # depth_mask[i] = mask_np[count]
                    # count +=1
                    #print(depth_mask[i])
                    #print(count)
            # print(np.sum(mask_2d),count,mask_np.shape[0])
            if random_mix_flag == True:
            # if False:
                objects = self.get_label(random_index)
                # data augmentation for labels
                if random_flip_flag:
                    for object in objects:
                        [x1, _, x2, _] = object.box2d
                        object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                        object.ry = np.pi - object.ry
                        object.pos[0] *= -1
                        if object.ry > np.pi:  object.ry -= 2 * np.pi
                        if object.ry < -np.pi: object.ry += 2 * np.pi
                object_num_temp = len(objects) if len(objects) < (self.max_objs - object_num) else (self.max_objs - object_num)
                for i in range(object_num_temp):
                    if objects[i].cls_type not in self.writelist:
                        continue

                    if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                        continue
                    # process 2d bbox & get 2d center
                    bbox_2d = objects[i].box2d.copy()
                    # add affine transformation for 2d boxes.
                    bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                    bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                    # modify the 2d bbox according to pre-compute downsample ratio
                    bbox_2d[:] /= self.downsample


                    # process 3d bbox & get 3d center
                    center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
                    center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                    center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                    center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                    center_3d = center_3d[0]  # shape adjustment
                    center_3d = affine_transform(center_3d.reshape(-1), trans)
                    center_3d /= self.downsample

                    # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                    center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
                    if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
                    if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue
        
                    # generate the radius of gaussian heatmap
                    w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                    radius = gaussian_radius((w, h))
                    radius = max(0, int(radius))
        
                    if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
                        draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                        continue
        
                    cls_id = self.cls2id[objects[i].cls_type]
                    cls_ids[i + object_num] = cls_id
                    draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)
        
                    # encoding 2d/3d offset & 2d size
                    indices[i + object_num] = center_heatmap[1] * features_size[0] + center_heatmap[0]
                    offset_2d[i + object_num] = center_2d - center_heatmap
                    size_2d[i + object_num] = 1. * w, 1. * h
        
                    # encoding depth
                    depth[i + object_num] = objects[i].pos[-1]
        
                    # encoding heading angle
                    #heading_angle = objects[i].alpha
                    heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0]+objects[i].box2d[2])/2)
                    if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                    if heading_angle < -np.pi: heading_angle += 2 * np.pi
                    heading_bin[i + object_num], heading_res[i + object_num] = angle2class(heading_angle)

                    offset_3d[i + object_num] = center_3d - center_heatmap
                    src_size_3d[i + object_num] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                    mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                    size_3d[i + object_num] = src_size_3d[i + object_num] - mean_size

                    #objects[i].trucation <=0.5 and objects[i].occlusion<=2 and (objects[i].box2d[3]-objects[i].box2d[1])>=25:
                    if objects[i].trucation <=0.5 and objects[i].occlusion<=2:
                        mask_2d[i + object_num] = 1

                    roi_depth = roi_align(torch.from_numpy(down_d_trans_temp).unsqueeze(0).unsqueeze(0).type(torch.float32),
                                        [torch.tensor(bbox_2d).unsqueeze(0)], [7, 7]).numpy()[0, 0]
                    # maintain interested points
                    # roi_depth_ind = (roi_depth > depth[i + object_num] - 3) & \
                    #                 (roi_depth < depth[i + object_num] + 3) & \
                    #                 (roi_depth > 0)
                    roi_depth_ind = (roi_depth != -100000)
                    roi_depth[~roi_depth_ind] = 0
                    vis_depth[i + object_num] = depth[i + object_num]
                    # att_depth[i + object_num] = depth[i + object_num] - vis_depth[i + object_num]
                    depth_mask[i + object_num] = roi_depth_ind
                    # print("depth_mask[i + object_num] : ", depth_mask[i + object_num])
                    
                    # avg_depth_mask[i + object_num] = self.mask_np


            targets = {'depth': depth,
                       'size_2d': size_2d,
                       'heatmap': heatmap,
                       'offset_2d': offset_2d,
                       'indices': indices,
                       'size_3d': size_3d,
                       'offset_3d': offset_3d,
                       'heading_bin': heading_bin,
                       'heading_res': heading_res,
                       'cls_ids': cls_ids,
                       'mask_2d': mask_2d,
                    #    'avg_depth_mask': avg_depth_mask,

                       'vis_depth': vis_depth,
                       #'att_depth': att_depth,
                    #    'depth_mask': depth_mask
                       }
        else:
            targets = {}

        inputs = img
        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size/features_size}

        return inputs, calib.P2, coord_range, targets, info   #calib.P2


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    cfg = {'random_flip':0.0, 'random_crop':1.0, 'scale':0.4, 'shift':0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False}
    dataset = KITTI('../../data', 'train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.show()
        # print(targets['size_3d'][0][0])

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        heatmap.show()

        break


    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
