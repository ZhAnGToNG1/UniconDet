from config.base import common, data, model, train, test, val, distill
from config.model.backbone import resnet
from config.model.head import fd_rmax, fd_rmax_cascade
from config.model.neck import deconv, fpn, augview
from config.model.loss import contourdet_loss

# common setting
data.dataset = 'dior'
common.task = 'unify1D'
common.task_name = 'MVCA_ResNet_CCRL'  # neck + backbone + XXX(improvement) + version
common.model_dir = './experiments/' + data.dataset + '_' + common.task + '/' + common.task_name
common.recorder_dir = common.model_dir + '/log'
val.inter_save_path = common.model_dir + '/eval'


# training parameters
train.batch_size = 8
train.num_workers = 12
train.dp_training = False
train.start_eval_ep = 80
train.gpus = [0]
train.eval_eps = 4
train.save_ep = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
train.epoch = 140
train.fd = 10
train.ns = 90
train.trainer = 'contourunify'
train.scale_range = [0.6, 1.4]
train.optimizer = {'name': 'adamw', 'lr': 1.25e-4,
                   'weight_decay': 5e-4,
                   'milestones': [50, 80, 100],
                   'gamma': 0.5}

train.multi_output_layers = True
train.output_layers = [4, 8, 16]

train.resume = None
model.resume_path = None


# backbone
model.backbone = resnet
model.backbone.num_layer = 50
model.backbone.output_five_feas = False
model.name = 'fpn_augview_cascade_network'

# neck
model.neck = augview
model.neck.stn_view_number = 2

# head
model.head = fd_rmax
model.head.heads = {'hm': 20, 'fourier': 2 * (2 * train.fd + 1), 'reg': 2, 'rmax': 1}
model.head.head_conv = 64

# loss
model.loss = contourdet_loss_region
model.loss.fd_weight = 5
model.loss.hm_weight = 1
model.loss.rmax_weight = 1
model.loss.reg_weight = 1

# dataset setting
data.num_classes = 20
data.class_name = ['windmill', 'tenniscourt', 'baseballfield', 'vehicle', 'stadium', 'groundtrackfield', 'airport', 'overpass',
                    'storagetank', 'harbor',
                    'ship', 'bridge', 'basketballcourt', 'Expressway-Service-area', 'golffield', 'airplane', 'dam',
                    'trainstation', 'Expressway-toll-station', 'chimney']

data.gt_path = '/root/data/Rotate_data/DIOR/test/val_unify.json'
data.imagelist = None
data.input_w = 800
data.input_h = 800

# testing and evaling
test.enable = True
test.visible = True
test.test_scales = [1]
test.patch_size = 800
test.patch_overlap = 0

val.valer = 'val_unify1D'

distill.distill_type = 'linear'
distill.teacher_weights = '/root/Aworkspace/code/crop_mae/pretrain/DIOR/image-level/checkpoint-800.pth'

class config(object):
    common = common
    data = data
    model = model
    train = train
    test = test
    distill = distill
