import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import logging
import argparse

from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester


parser = argparse.ArgumentParser(description='implementation of MonoLSS')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test', action='store_true', help='evaluate model on test set')
parser.add_argument('--config', type=str, default='lib/kitti.yaml')
args = parser.parse_args()


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def main():
    # load cfg
    config_path = args.config
    
    # Try different path options if the direct path doesn't exist
    if not os.path.exists(config_path):
        # Check if we're in Kaggle environment
        if os.path.exists('/kaggle'):
            # Try with absolute path in Kaggle
            kaggle_path = os.path.join('/kaggle/working/monolss', config_path)
            if os.path.exists(kaggle_path):
                config_path = kaggle_path
            else:
                # For the case of --config kitti_custom.yaml or --config lib/kitti_custom.yaml
                if not config_path.startswith('lib/'):
                    kaggle_path = os.path.join('/kaggle/working/monolss/lib', config_path)
                    if os.path.exists(kaggle_path):
                        config_path = kaggle_path
        else:
            # Try path relative to ROOT_DIR
            root_relative_path = os.path.join(ROOT_DIR, config_path)
            if os.path.exists(root_relative_path):
                config_path = root_relative_path
    
    # Final check
    assert os.path.exists(config_path), f"Config file not found at {config_path}. Please provide the correct path. Current paths tried: {args.config}, {os.path.join('/kaggle/working/monolss', args.config)}"
    
    print(f"Using config file at: {config_path}")
    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    os.makedirs(cfg['trainer']['log_dir'], exist_ok=True)
    logger = create_logger(os.path.join(cfg['trainer']['log_dir'], 'train.log'))

    import shutil
    if not args.evaluate:
        if not args.test:
            if os.path.exists(os.path.join(cfg['trainer']['log_dir'], 'lib/')):
                shutil.rmtree(os.path.join(cfg['trainer']['log_dir'], 'lib/'))
        if not args.test:
            shutil.copytree('./lib', os.path.join(cfg['trainer']['log_dir'], 'lib/'))
        
    
    #  build dataloader
    train_loader, val_loader, test_loader = build_dataloader(cfg['dataset'])

    # build model
    model = build_model(cfg['model'], train_loader.dataset.cls_mean_size)

    # evaluation mode
    if args.evaluate:
        tester = Tester(cfg['tester'], cfg['dataset'], model, val_loader, logger)
        tester.test()
        return

    if args.test:
        tester = Tester(cfg['tester'], cfg['dataset'], model, test_loader, logger)
        tester.test()
        return

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr & bnm scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg,
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=val_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger)
    trainer.train()


if __name__ == '__main__':
    main()
