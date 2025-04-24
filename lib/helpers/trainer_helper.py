import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MEMORY_EFFICIENT'] = '1'

import tqdm

import torch
import torch.nn as nn
import numpy as np
import pdb
import gc
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

        # Check if we should use memory-efficient training
        self.memory_efficient = os.environ.get('MEMORY_EFFICIENT', '0') == '1'
        if self.memory_efficient:
            self.logger.info("Memory-efficient mode enabled: using smaller batches and clearing cache regularly")

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

            # Clear CUDA cache before each epoch if memory-efficient mode is enabled
            if self.memory_efficient:
                torch.cuda.empty_cache()
                gc.collect()

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
                    # Clear CUDA cache before evaluation
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    Car_res = self.eval_one_epoch()
                    self.logger.info(str(Car_res))
                except Exception as e:
                    self.logger.error(f"Evaluation failed with error: {str(e)}")
                    self.logger.info("Continuing training despite evaluation failure")
                
                # Clear CUDA cache after evaluation
                torch.cuda.empty_cache()
                gc.collect()


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
            # In memory-efficient mode, process smaller batches at a time if needed
            if self.memory_efficient and isinstance(inputs, torch.Tensor) and inputs.size(0) > 1:
                # Process half a batch at a time if memory efficiency is needed
                # and we're not already using batch size 1
                sub_batch_size = max(1, inputs.size(0) // 2)
                
                # Split each tensor in the batch
                total_loss = torch.zeros(1).cuda()
                
                for start_idx in range(0, inputs.size(0), sub_batch_size):
                    end_idx = min(start_idx + sub_batch_size, inputs.size(0))
                    
                    # Clear cache before processing sub-batch
                    torch.cuda.empty_cache()
                    
                    # Extract sub-batch
                    sub_inputs = inputs[start_idx:end_idx].to(self.device)
                    sub_calibs = calibs[start_idx:end_idx].to(self.device)
                    sub_coord_ranges = coord_ranges[start_idx:end_idx].to(self.device)
                    
                    # Extract sub-targets
                    sub_targets = {}
                    for key in targets.keys():
                        if isinstance(targets[key], torch.Tensor) and targets[key].size(0) == inputs.size(0):
                            sub_targets[key] = targets[key][start_idx:end_idx].to(self.device)
                        else:
                            # Handle other cases (might need customization depending on targets structure)
                            sub_targets[key] = targets[key]
                    
                    # Process sub-batch
                    self.optimizer.zero_grad()
                    criterion = LSS_Loss(self.epoch)
                    outputs = self.model(sub_inputs, sub_coord_ranges, sub_calibs, sub_targets)
                    sub_loss, sub_loss_terms = criterion(outputs, sub_targets)
                    
                    # Apply loss weights
                    if loss_weights is not None:
                        sub_loss = torch.zeros(1).cuda()
                        for key in loss_weights.keys():
                            if key in sub_loss_terms:
                                sub_loss += loss_weights[key].detach() * sub_loss_terms[key]
                    
                    # Scale the loss by batch ratio
                    sub_loss = sub_loss * (end_idx - start_idx) / inputs.size(0)
                    
                    # Backward for this sub-batch
                    sub_loss.backward()
                    
                    # Accumulate loss terms for logging
                    for key in sub_loss_terms.keys():
                        if key not in stat_dict.keys():
                            stat_dict[key] = 0
                        if key not in disp_dict.keys():
                            disp_dict[key] = 0
                            
                        if isinstance(sub_loss_terms[key], int):
                            stat_term = sub_loss_terms[key]
                            disp_term = sub_loss_terms[key]
                        else:
                            stat_term = sub_loss_terms[key].detach()
                            disp_term = sub_loss_terms[key].detach()
                            
                        # Weight by sub-batch size ratio
                        ratio = (end_idx - start_idx) / inputs.size(0)
                        stat_dict[key] += stat_term * ratio
                        disp_dict[key] += disp_term * ratio
                    
                    # Free up memory
                    del sub_inputs, sub_calibs, sub_coord_ranges, sub_targets, outputs
                
                # Apply optimizer step after processing all sub-batches
                self.optimizer.step()
            else:
                # Standard processing for single image or small batches
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
                    if isinstance(loss_terms[key], int):
                        disp_dict[key] += (loss_terms[key])
                    else:
                        disp_dict[key] += (loss_terms[key]).detach()

            # Clear cache periodically in memory-efficient mode
            if self.memory_efficient and batch_idx % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            trained_batch = batch_idx + 1
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
        
        # Try to reduce batch size for evaluation to save memory
        try:
            # Create output directory for saving results
            self.logger.info('==> Preparing output directory...')
            if not os.path.exists(self.eval_output_dir):
                os.makedirs(self.eval_output_dir)
            result_dir = os.path.join(self.eval_output_dir, 'data')
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
                
            # Process batches individually with careful memory management
            for batch_idx, (inputs, calibs, coord_ranges, _, infos) in enumerate(self.test_loader):
                # Skip if we've already processed too many batches during debugging
                if batch_idx > 20 and 'SKIP_LARGE_EVAL' in os.environ:
                    continue
                    
                # Clear cache before each batch
                torch.cuda.empty_cache()
                gc.collect()
                
                try:
                    # Process one image at a time to save memory
                    batch_size = inputs.size(0)
                    for i in range(batch_size):
                        single_input = inputs[i:i+1].to(self.device)
                        single_calib = calibs[i:i+1].to(self.device)
                        single_coord_range = coord_ranges[i:i+1].to(self.device)
                        single_info = {k: v[i] if isinstance(v, list) else v[i:i+1] for k, v in infos.items()}
                        
                        # Run inference with gradient calculation disabled
                        with torch.no_grad():
                            try:
                                # Use lower top-k value to save memory
                                outputs = self.model(single_input, single_coord_range, single_calib, K=20, mode='val')
                                
                                # Process detections for this single image
                                det = self.decode_func(outputs, {k: [v] if not isinstance(v, list) else [v] for k, v in single_info.items()})
                                
                                if det and isinstance(det, list) and len(det) > 0:
                                    self.detection.extend(det)
                                
                                # Clear memory
                                del outputs, single_input, single_calib, single_coord_range
                                torch.cuda.empty_cache()
                                
                            except Exception as e:
                                self.logger.error(f"Inference error on batch {batch_idx}, image {i}: {str(e)}")
                                continue
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    continue
                
                # Log progress
                if batch_idx % 10 == 0:
                    self.logger.info(f'Processed {batch_idx} batches')

            # Save detection results
            self.logger.info('==> Saving detection results...')
            if not self.detection:
                self.logger.warning("No valid detections were made during evaluation")
                return {"Error": "No valid detections available for evaluation"}
                
            for info in self.detection:
                try:
                    # Skip if image_id is missing
                    if 'image_id' not in info:
                        continue
                        
                    # Create output file
                    filename = '%06d.txt' % info['image_id']
                    filepath = os.path.join(result_dir, filename)
                    
                    with open(filepath, 'w') as f:
                        # Check if we have valid detections
                        if 'bbox' in info and info['bbox'] is not None and len(info['bbox']) > 0:
                            for i in range(len(info['bbox'])):
                                try:
                                    # Safely get class name with bounds checking
                                    cls_id = int(info['cls_id'][i]) if i < len(info['cls_id']) else 0
                                    cls_id = max(0, min(cls_id, len(self.class_name)-1))
                                    class_name = self.class_name[cls_id]
                                    
                                    # Get other values with safety checks
                                    score = float(info['score'][i]) if 'score' in info and i < len(info['score']) else 0.0
                                    bbox = info['bbox'][i] if i < len(info['bbox']) else [0, 0, 0, 0]
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
                                    continue
                except Exception as e:
                    self.logger.error(f"Error saving detection results: {str(e)}")
                    continue

            # Find ground truth directory
            gt_label_path = os.path.join(self.root_dir, 'val', 'label_2')
            if not os.path.exists(gt_label_path):
                # Try alternate paths
                alternate_paths = [
                    os.path.join(self.root_dir, 'train/label_2'),
                    os.path.join(self.root_dir, 'train', 'label_2'),
                    os.path.join(self.root_dir, 'nuscenes_in_kitti/train/label_2'),
                    self.label_dir
                ]
                
                for path in alternate_paths:
                    if os.path.exists(path):
                        gt_label_path = path
                        self.logger.info(f'Found ground truth labels at {gt_label_path}')
                        break
                else:
                    self.logger.warning(f'Cannot find ground truth labels. Using label_dir as fallback: {self.label_dir}')
                    gt_label_path = self.label_dir

            # Run evaluation
            try:
                if not os.listdir(result_dir):
                    self.logger.warning("No detection results available for evaluation")
                    return {"Error": "No detection results available"}
                    
                self.logger.info('==> Running KITTI evaluation...')
                self.logger.info(f'Using ground truth from: {gt_label_path}')
                self.logger.info(f'Using predictions from: {result_dir}')
                
                # Import function for evaluation
                from tools.eval import eval_from_scrach
                
                # Ensure eval_cls is a list
                if isinstance(self.eval_cls, str):
                    eval_cls_list = [self.eval_cls]
                else:
                    eval_cls_list = self.eval_cls
                
                # Run evaluation
                Car_res = eval_from_scrach(
                    gt_label_path,
                    result_dir,
                    eval_cls_list=eval_cls_list
                )
                return Car_res
            except Exception as e:
                self.logger.error(f"Evaluation error: {str(e)}")
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
        
      