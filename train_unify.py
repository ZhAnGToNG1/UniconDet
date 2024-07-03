import argparse
import importlib
import torch
import os
import shutil
import numpy as np
from dataset.data_loader import make_data_loader
from model.model_adapter import create_model, load_model, save_model
from train.optimizer import make_optimizer
from train.scheduler import make_lr_scheduler
from train.recorder import make_recorder
from train.trainer_adapter import make_trainer
from train.loss_adapter import make_losser
from val.evaluator_adapter import make_evaler
from utils.tool import save_result

def parse_args():
    parser = argparse.ArgumentParser(description='Training ContourDet')
    parser.add_argument('config_file')
    parser.add_argument("--device", default=0, type=int, help='device idx')
    args = parser.parse_args()
    return args

def get_cfg(args):
    cfg = importlib.import_module('config.' + args.config_file)
    return cfg

def main(cfg, args):
    print(cfg.common.model_dir)
    if os.path.exists(cfg.common.model_dir):
        print('######################################################')
        print('The task has been created, confirm whether to continue!')
        print('######################################################')
    os.system('mkdir -p {}'.format(cfg.common.model_dir))
    shutil.copy('./config/' + args.config_file.replace('.', '/') + '.py', cfg.common.model_dir + '/' + args.config_file.split('.')[-1] + '.py')
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.cuda.manual_seed_all(cfg.train.seed)
    torch.backends.cudnn.benchmark = True
    if cfg.train.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_dataloader, val_dataloader = make_data_loader(is_train=True, cfg=cfg)
    print("\nStarting training ...")
    print('fd:%s, ns:%s'%(cfg.train.fd, cfg.train.ns))

    model = create_model(cfg)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_lr_scheduler(optimizer, cfg)
    recorder = make_recorder(cfg)
    losser = make_losser(cfg)
    trainer = make_trainer(cfg, model, losser)
    evaluator = make_evaler(cfg, val_dataloader)

    start_epoch = 0
    if cfg.train.resume:
        start_epoch = load_model(cfg, model, optimizer, scheduler, recorder)


    mAP_log = os.path.join(cfg.common.recorder_dir, 'map.txt')
    loss_log = os.path.join(cfg.common.recorder_dir, 'loss.txt')

    best = 0
    for epoch in range(start_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        trainer.train(train_dataloader, optimizer, recorder, loss_log)
        scheduler.step()

        save_model(model, optimizer, scheduler, recorder, epoch,
                   cfg.common.model_dir, save_last=True)

        if cfg.train.save_every_ep:
            save_model(model, optimizer, scheduler, recorder, epoch + 1,
                       cfg.common.model_dir, save_last=False, save_best=False)
        else:
            if (epoch + 1) in list(cfg.train.save_ep):
                save_model(model, optimizer, scheduler, recorder, epoch + 1,
                           cfg.common.model_dir, save_last=False, save_best=False)

        if (epoch + 1) % cfg.train.eval_eps == 0 and epoch > cfg.train.start_eval_ep:
            voc_05map, print_coco, ap50 = evaluator.test_in_training(model)
            if ap50 > best:
                print('Best AP50:', ap50)
                best = ap50
                save_model(model, optimizer, scheduler, recorder, epoch,
                           cfg.common.model_dir, save_last=False, save_best=True)
            print("================evaluation...===============")
            save_result(cfg, mAP_log, epoch, print_coco, voc_05map)
            print("============================================")
    return

if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.device)
    cfg = get_cfg(args)
    main(cfg, args)


