import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.datasets.kitti import KITTI

def build_dataloader(cfg):
    # check if ImageSets directory exists, if not, create it
    imageset_dir = os.path.join(cfg['root_dir'], 'ImageSets')
    os.makedirs(imageset_dir, exist_ok=True)
    
    # generate appropriate split files if they don't exist
    for split in ['train', 'val', 'trainval', 'test']:
        split_file = os.path.join(imageset_dir, split + '.txt')
        if not os.path.exists(split_file):
            # For training, use train directory
            if split in ['train', 'trainval']:
                data_dir = os.path.join(cfg['root_dir'], 'train')
            # For validation and test, use val directory
            else:
                data_dir = os.path.join(cfg['root_dir'], 'val')
            
            if os.path.exists(data_dir):
                image_dir = os.path.join(data_dir, 'image_2')
                if os.path.exists(image_dir):
                    image_files = os.listdir(image_dir)
                    image_ids = sorted([os.path.splitext(f)[0] for f in image_files if f.endswith('.png')])
                    with open(split_file, 'w') as f:
                        f.write('\n'.join(image_ids))
                        print(f"Created {split} split with {len(image_ids)} images")
                    
    # --------------  build kitti dataset ----------------
    if cfg['type'] == 'kitti':
        train_set = KITTI(root_dir=cfg['root_dir'], split='trainval', cfg=cfg)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=cfg['num_workers'],
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        val_set = KITTI(root_dir=cfg['root_dir'], split='val', cfg=cfg)
        val_loader = DataLoader(dataset=val_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=cfg['num_workers'],
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=cfg['drop_last_val'])
        test_set = KITTI(root_dir=cfg['root_dir'], split='test', cfg=cfg)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=cfg['num_workers'],
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        return train_loader, val_loader, test_loader

    elif cfg['type'] == 'waymo':
        train_set = Waymo(root_dir=cfg['root_dir'], split='train', cfg=cfg)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=cfg['num_workers'],
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        test_set = Waymo(root_dir=cfg['root_dir'], split='test', cfg=cfg)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=cfg['num_workers'],
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        return train_loader, train_loader, test_loader

    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

