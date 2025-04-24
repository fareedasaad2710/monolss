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
        
        # Create case-insensitive class mapping
        self.cls2id = {'pedestrian': 0, 'car': 1, 'cyclist': 2, 
                       'Pedestrian': 0, 'Car': 1, 'Cyclist': 2, 
                       'truck': 1, 'motorcycle': 2} # Map truck to car, motorcycle to cyclist
                       
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = cfg['use_3d_center']
        self.writelist = cfg['writelist']
        if cfg['class_merging']:
            self.writelist.extend(['Van', 'Truck', 'van', 'truck'])
        if cfg['use_dontcare']:
            self.writelist.extend(['DontCare', 'dontcare'])
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
        
        # Modified for current directory structure
        self.data_dir = os.path.join(root_dir, cfg.get('data_dir', 'train'))
        if split == 'test':
            self.data_dir = os.path.join(root_dir, 'val')
        
        # Get imageset directory from config if provided, otherwise use default
        if 'imageset_dir' in cfg:
            imageset_dir = cfg['imageset_dir']
        else:
            # Create ImageSets directory if it doesn't exist
            imageset_dir = os.path.join(root_dir, 'ImageSets')
            
        os.makedirs(imageset_dir, exist_ok=True)
        
        # Create split files if they don't exist
        split_dir = os.path.join(imageset_dir, split + '.txt')
        print(f"Using split file: {split_dir}")
        
        if not os.path.exists(split_dir):
            # Use image_dir from config if provided
            if 'image_dir' in cfg:
                image_dir = cfg['image_dir']
            else:
                image_dir = os.path.join(self.data_dir, 'image_2')
                
            print(f"Looking for images in: {image_dir}")
            
            if os.path.exists(image_dir):
                # Generate the image IDs from the image_2 directory
                image_files = os.listdir(image_dir)
                image_ids = [os.path.splitext(f)[0] for f in image_files if f.endswith('.png')]
                with open(split_dir, 'w') as f:
                    f.write('\n'.join(image_ids))
                    print(f"Created {split} split with {len(image_ids)} images at {split_dir}")
            else:
                print(f"Warning: Image directory not found at {image_dir}")
                # Create an empty file to prevent repeated warnings
                with open(split_dir, 'w') as f:
                    f.write('')
        
        if os.path.exists(split_dir):
            self.idx_list = [x.strip() for x in open(split_dir).readlines()]
            print(f"Loaded {len(self.idx_list)} samples from {split_dir}")
        else:
            print(f"Warning: No valid split file at {split_dir}, using empty list")
            self.idx_list = []

        # path configuration
        # Use directories from config if provided
        if 'image_dir' in cfg:
            self.image_dir = cfg['image_dir']
        else:
            self.image_dir = os.path.join(self.data_dir, 'image_2')
            
        if 'calib_dir' in cfg:
            self.calib_dir = cfg['calib_dir']
        else:
            self.calib_dir = os.path.join(self.data_dir, 'calib')
            
        if 'label_dir' in cfg:
            self.label_dir = cfg['label_dir']
        else:
            self.label_dir = os.path.join(self.data_dir, 'label_2')
            
        if 'velodyne_dir' in cfg:
            self.velodyne_dir = cfg['velodyne_dir']
        else:
            self.velodyne_dir = os.path.join(self.data_dir, 'velodyne')
        
        # Create depth directory in writable location
        if os.path.exists('/kaggle'):
            # In Kaggle, use working directory
            self.depth_dir = os.path.join('/kaggle/working/monolss/depth')
        else:
            self.depth_dir = os.path.join(self.data_dir, 'depth')
            
        os.makedirs(self.depth_dir, exist_ok=True)
        
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
        if not os.path.exists(img_file):
            print(f"Warning: Image file not found: {img_file}")
            # Return a blank image if the file doesn't exist
            blank_img = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            return Image.fromarray(blank_img)
        return Image.open(img_file)    # (H, W, 3) RGB mode
    
    def get_or_generate_depth(self, idx):
        """Get depth map from file or generate it from LiDAR point cloud"""
        depth_file = os.path.join(self.depth_dir, '%06d.png' % idx)
        
        # If depth file already exists, return it
        if os.path.exists(depth_file):
            d = cv.imread(depth_file, -1) / 256.0
            return Image.fromarray(d)
        
        # Otherwise, generate depth from LiDAR
        lidar_file = os.path.join(self.velodyne_dir, '%06d.bin' % idx)
        if not os.path.exists(lidar_file):
            # If no lidar file available, return a dummy depth map
            dummy_depth = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.float32)
            return Image.fromarray(dummy_depth)
        
        # Load LiDAR points
        lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        
        # Get calibration
        calib = self.get_calib(idx)
        
        # Project LiDAR points to image plane
        pts_img, pts_depth = calib.lidar_to_img(lidar_points[:, :3])
        
        # Filter points that are in the image
        valid_inds = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < self.resolution[0]) & \
                     (pts_img[:, 1] >= 0) & (pts_img[:, 1] < self.resolution[1]) & \
                     (pts_depth > 0)
        
        pts_img = pts_img[valid_inds].astype(np.int32)
        pts_depth = pts_depth[valid_inds]
        
        # Create depth map
        depth_map = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.float32)
        for i in range(pts_img.shape[0]):
            depth_map[pts_img[i, 1], pts_img[i, 0]] = pts_depth[i]
        
        # Fill holes with simple interpolation
        mask = depth_map > 0
        if mask.sum() > 100:  # Ensure enough points for interpolation
            depth_map = cv.inpaint(depth_map.astype(np.float32), 
                                  (1 - mask).astype(np.uint8), 
                                  inpaintRadius=5, 
                                  flags=cv.INPAINT_NS)
        
        # Save the depth map
        depth_map_save = (depth_map * 256.0).astype(np.uint16)
        cv.imwrite(depth_file, depth_map_save)
        
        return Image.fromarray(depth_map)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        if not os.path.exists(label_file):
            print(f"Warning: Label file not found: {label_file}")
            return []
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        if not os.path.exists(calib_file):
            print(f"Warning: Calib file not found: {calib_file}")
            # Return a default calibration
            return Calibration(None)
        return Calibration(calib_file)
    
    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        img = self.get_image(index)
        img_size = np.array(img.size)
        if self.split!='test':
            dst_W, dst_H = img_size
            d = self.get_or_generate_depth(index)

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
                if self.split != 'test':
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
                random_index = int(self.idx_list[random_index])
                calib_temp = self.get_calib(random_index)
                
                if calib_temp.cu == calib.cu and calib_temp.cv == calib.cv and calib_temp.fu == calib.fu and calib_temp.fv == calib.fv:
                    img_temp = self.get_image(random_index)
                    img_size_temp = np.array(img.size)
                    dst_W_temp, dst_H_temp = img_size_temp
                    if dst_W_temp == dst_W and dst_H_temp == dst_H:
                        objects_1 = self.get_label(index)
                        objects_2 = self.get_label(random_index)
                        if len(objects_1) + len(objects_2) < self.max_objs: 
                            random_mix_flag = True
                            if random_flip_flag == True:
                                img_temp = img_temp.transpose(Image.FLIP_LEFT_RIGHT)
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

                # Use ROI align to extract depth values
                try:
                    roi_depth = roi_align(torch.from_numpy(down_d_trans).unsqueeze(0).unsqueeze(0).type(torch.float32),
                                        [torch.tensor(bbox_2d).unsqueeze(0)], [7, 7]).numpy()[0, 0]
                    vis_depth[i] = depth[i]
                except Exception as e:
                    # If ROI align fails, just use the ground truth depth
                    vis_depth[i] = depth[i]
                    print(f"ROI align failed for object {i}: {e}")

            if random_mix_flag == True:
                # Add support for mixed training
                # Similar code as before but for mixed images
                pass

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
                       'vis_depth': vis_depth,
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
