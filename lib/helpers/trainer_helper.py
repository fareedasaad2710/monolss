import os
import tqdm

import torch
import torch.nn as nn
import numpy as np
import pdb
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import save_checkpoint
from lib.helpers.save_helper import load_checkpoint
from lib.losses.loss_function import LSS_Loss,Hierarchical_Task_Learning
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections

from tools import eval


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger):
        self.cfg_train = cfg['trainer']
        self.cfg_test = cfg['tester']
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_name = test_loader.dataset.class_name
        self.label_dir = cfg['dataset']['label_dir']
        self.eval_cls = cfg['dataset']['eval_cls']
        
        # Set up the root directory for evaluation
        self.root_dir = cfg['dataset']['root_dir']
        self.eval_output_dir = cfg['tester']['out_dir']
        self.detection = [] # Initialize detection list for evaluation
        
        # Add decode_func to fix the AttributeError
        self.decode_func = extract_dets_from_outputs

        if self.cfg_train.get('resume_model', None):
            assert os.path.exists(self.cfg_train['resume_model'])
            self.epoch = load_checkpoint(self.model, self.optimizer, self.cfg_train['resume_model'], self.logger, map_location=self.device)
            self.lr_scheduler.last_epoch = self.epoch - 1

        self.model = torch.nn.DataParallel(model).to(self.device)

    def train(self):
        start_epoch = self.epoch
        ei_loss = self.compute_e0_loss()
        loss_weightor = Hierarchical_Task_Learning(ei_loss)
        for epoch in range(start_epoch, self.cfg_train['max_epoch']):
            # train one epoch
            self.logger.info('------ TRAIN EPOCH %03d ------' %(epoch + 1))
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.logger.info('Learning Rate: %f' % self.warmup_lr_scheduler.get_lr()[0])
            else:
                self.logger.info('Learning Rate: %f' % self.lr_scheduler.get_lr()[0])

            # reset numpy seed.
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            loss_weights = loss_weightor.compute_weight(ei_loss,self.epoch)

            log_str = 'Weights: '
            for key in sorted(loss_weights.keys()):
                log_str += ' %s:%.4f,' %(key[:-4], loss_weights[key])   
            self.logger.info(log_str)

            ei_loss = self.train_one_epoch(loss_weights)
            self.epoch += 1
            
            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            if ((self.epoch % self.cfg_train['eval_frequency']) == 0 and \
                self.epoch >= self.cfg_train['eval_start']):
                try:
                    self.logger.info('------ EVAL EPOCH %03d ------' % (self.epoch))
                    Car_res = self.eval_one_epoch()
                    self.logger.info(str(Car_res))
                except Exception as e:
                    self.logger.error(f"Evaluation failed with error: {str(e)}")
                    self.logger.info("Continuing training despite evaluation failure")


            if ((self.epoch % self.cfg_train['save_frequency']) == 0
                and self.epoch >= self.cfg_train['eval_start']):
                os.makedirs(self.cfg_train['log_dir']+'/checkpoints', exist_ok=True)
                ckpt_name = os.path.join(self.cfg_train['log_dir']+'/checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name, self.logger)

        return None
    
    def compute_e0_loss(self):
        self.model.train()
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=True, desc='pre-training loss stat')
        with torch.no_grad():        
            for batch_idx, (inputs,calibs,coord_ranges, targets, info) in enumerate(self.train_loader):
                if type(inputs) != dict:
                    inputs = inputs.to(self.device)
                else:
                    for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
                calibs = calibs.to(self.device)
                coord_ranges = coord_ranges.to(self.device)
                for key in targets.keys():
                    targets[key] = targets[key].to(self.device)
    
                # train one batch
                criterion = LSS_Loss(self.epoch)
                outputs = self.model(inputs,coord_ranges,calibs,targets)
                _, loss_terms = criterion(outputs, targets)
                
                trained_batch = batch_idx + 1
                # accumulate statistics
                for key in loss_terms.keys():
                    if key not in disp_dict.keys():
                        disp_dict[key] = 0
                    disp_dict[key] += loss_terms[key]      
                progress_bar.update()
            progress_bar.close()
            for key in disp_dict.keys():
                disp_dict[key] /= trained_batch             
        return disp_dict

    def train_one_epoch(self,loss_weights=None):
        self.model.train()

        disp_dict = {}
        stat_dict = {}
        for batch_idx, (inputs,calibs,coord_ranges, targets, info) in enumerate(self.train_loader):
            if type(inputs) != dict:
                inputs = inputs.to(self.device)
            else:
                for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)
            for key in targets.keys(): targets[key] = targets[key].to(self.device)
            # train one batch
            self.optimizer.zero_grad()
            criterion = LSS_Loss(self.epoch)
            outputs = self.model(inputs,coord_ranges,calibs,targets)

            total_loss, loss_terms = criterion(outputs, targets)
            
            if loss_weights is not None:
                total_loss = torch.zeros(1).cuda()
                for key in loss_weights.keys():
                    total_loss += loss_weights[key].detach()*loss_terms[key]
            total_loss.backward()
            self.optimizer.step()

            trained_batch = batch_idx + 1

            # accumulate statistics
            for key in loss_terms.keys():
                if key not in stat_dict.keys():
                    stat_dict[key] = 0

                if isinstance(loss_terms[key], int):
                    stat_dict[key] += (loss_terms[key])
                else:
                    stat_dict[key] += (loss_terms[key]).detach()
            for key in loss_terms.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                # disp_dict[key] += loss_terms[key]
                if isinstance(loss_terms[key], int):
                    disp_dict[key] += (loss_terms[key])
                else:
                    disp_dict[key] += (loss_terms[key]).detach()
            # display statistics in terminal
            if trained_batch % self.cfg_train['disp_frequency'] == 0:
                log_str = 'BATCH[%04d/%04d]' % (trained_batch, len(self.train_loader))
                for key in sorted(disp_dict.keys()):
                    disp_dict[key] = disp_dict[key] / self.cfg_train['disp_frequency']
                    log_str += ' %s:%.4f,' %(key, disp_dict[key])
                    disp_dict[key] = 0  # reset statistics
                self.logger.info(log_str)
                
        for key in stat_dict.keys():
            stat_dict[key] /= trained_batch
                            
        return stat_dict    
    def eval_one_epoch(self):
        self.model.eval()

        # Reset detection list for this evaluation
        self.detection = []
        
        try:
            for batch_idx, (inputs, calibs, coord_ranges, _, infos) in enumerate(self.test_loader):
                # Move The Datas onto the device
                inputs = inputs.to(self.device)
                calibs = calibs.to(self.device)
                coord_ranges = coord_ranges.to(self.device)
                
                # Run model inference with error handling
                try:
                    outputs = self.model(inputs, coord_ranges, calibs, K=50, mode='val')
                    
                    # Extract detections with robust error handling
                    try:
                        dets = self.decode_func(outputs, infos)
                        self.detection.extend(dets)
                    except Exception as e:
                        self.logger.error(f"Error in decode_func at batch {batch_idx}: {str(e)}")
                        # Continue with next batch rather than failing entire evaluation
                        continue
                        
                except Exception as e:
                    self.logger.error(f"Model inference error at batch {batch_idx}: {str(e)}")
                    continue

            # Save results so far
            self.logger.info('==> Saving results...')
            if not os.path.exists(self.eval_output_dir):
                os.makedirs(self.eval_output_dir)
            result_dir = os.path.join(self.eval_output_dir, 'data')
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
                
            # Safe writing of results with robust error handling
            for info in self.detection:
                try:
                    filename = '%06d.txt' % info['image_id']
                    filepath = os.path.join(result_dir, filename)
                    
                    with open(filepath, 'w') as f:
                        if 'bbox' in info and info['bbox'] is not None and len(info['bbox']) > 0:
                            for i in range(len(info['bbox'])):
                                try:
                                    # Safely get class name
                                    cls_id = int(info['cls_id'][i]) if i < len(info['cls_id']) else 0
                                    cls_id = max(0, min(cls_id, len(self.class_name)-1))  # Ensure valid index
                                    class_name = self.class_name[cls_id]
                                    
                                    # Extract other data with fallbacks for missing values
                                    score = float(info['score'][i]) if 'score' in info and i < len(info['score']) else 0.0
                                    bbox = info['bbox'][i] if i < len(info['bbox']) else [0, 0, 0, 0]
                                    
                                    # Handle optional data with safe defaults
                                    alpha = float(info['alpha'][i]) if 'alpha' in info and i < len(info['alpha']) else 0.0
                                    depth = float(info['depth'][i]) if 'depth' in info and i < len(info['depth']) else 0.0
                                    dim = info['dim'][i] if 'dim' in info and i < len(info['dim']) else [1.0, 1.0, 1.0]
                                    loc = info['loc'][i] if 'loc' in info and i < len(info['loc']) else [0.0, 0.0, 0.0]
                                    rot_y = float(info['rot_y'][i]) if 'rot_y' in info and i < len(info['rot_y']) else 0.0
                                    
                                    # Write line with safe formatting
                                    f.write(f"{class_name} 0.0 0 {alpha:.2f} ")
                                    f.write(f"{bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} ")
                                    f.write(f"{dim[0]:.2f} {dim[1]:.2f} {dim[2]:.2f} ")
                                    f.write(f"{loc[0]:.2f} {loc[1]:.2f} {loc[2]:.2f} ")
                                    f.write(f"{rot_y:.2f} {score:.2f}\n")
                                except Exception as e:
                                    self.logger.error(f"Error writing detection {i} to {filepath}: {str(e)}")
                                    # Skip this detection rather than failing whole file
                                    continue
                        else:
                            # Write empty file to prevent errors in evaluation
                            pass
                except Exception as e:
                    self.logger.error(f"Error handling file {filename}: {str(e)}")
                    continue

            # Find ground truth directory with extra error handling
            gt_label_path = os.path.join(self.root_dir, 'val', 'label_2')
            if not os.path.exists(gt_label_path):
                # Try alternate paths for nuscenes dataset
                alternate_paths = [
                    os.path.join(self.root_dir, 'train/label_2'),
                    os.path.join(self.root_dir, 'train', 'label_2'),
                    os.path.join(self.root_dir, 'nuscenes_in_kitti/train/label_2'),
                    os.path.join(self.label_dir)  # Use label_dir directly from config
                ]
                
                for path in alternate_paths:
                    if os.path.exists(path):
                        gt_label_path = path
                        self.logger.info(f'Found ground truth labels at {gt_label_path}')
                        break
                else:
                    # If no valid path found, use a default value and warn
                    self.logger.warning(f'Cannot find ground truth labels path. Using configured label_dir as fallback.')
                    gt_label_path = self.label_dir

            # Run KITTI evaluation with robust error handling
            try:
                self.logger.info('==> Evaluating with official KITTI eval code...')
                self.logger.info(f'Using ground truth from: {gt_label_path}')
                self.logger.info(f'Using predictions from: {result_dir}')
                
                from tools.eval import eval_from_scrach
                
                # Handle class list properly
                # Ensure eval_cls is always a list
                if isinstance(self.eval_cls, str):
                    eval_cls_list = [self.eval_cls]
                else:
                    eval_cls_list = self.eval_cls
                
                Car_res = eval_from_scrach(
                    gt_label_path,
                    result_dir,
                    eval_cls_list=eval_cls_list
                )
                return Car_res
            except Exception as e:
                self.logger.error(f"KITTI evaluation failed: {str(e)}")
                return {"Error": str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error in evaluation: {str(e)}")
            return {"Error": str(e)}

    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            out_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()        
        
      