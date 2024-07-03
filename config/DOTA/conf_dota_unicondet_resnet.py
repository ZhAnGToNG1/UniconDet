from config.base import common, data, model, train, test, val
from config.model.backbone import resnet
from config.model.head import fd_rmax_ccrl
from config.model.neck import fpn,augview
from config.model.loss import contourdet_loss_region

# common setting
data.dataset = 'dota_obb'
common.task = 'unify1D'
common.task_name = 'mvca_resnet101_unicondet'  # neck + backbone + XXX(improvement) + version
common.model_dir = './experiments/Comparison/' + data.dataset + '_' + common.task  + '_' + common.task_name
common.recorder_dir = common.model_dir + '/log'
val.inter_save_path = common.model_dir + '/eval'
common.dota_test_res_dir = common.model_dir + '/test_results'


# training parameters
train.batch_size = 10
train.num_workers = 8
train.dp_training = True
train.start_eval_ep = 80
train.gpus = [0, 1]
train.eval_eps = 3
train.save_ep = [5, 10, 20, 30, 40, 50, 90, 100, 110, 120, 130, 140, 150]
train.save_every_ep = True
train.epoch = 160
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
train.seed = 317
train.deterministic = False

train.resume = False
model.resume_path = None

# backbone
model.backbone = resnet
model.backbone.num_layer = 101
model.backbone.output_five_feas = False
model.name = 'fpn_augview_cascade_network'

# neck
model.neck = augview
model.neck.stn_view_number = 2

# head
model.head = fd_rmax_ccrl
model.head.heads = {'hm': 15, 'fourier': 2 * (2 * train.fd + 1), 'reg': 2, 'rmax': 1}
model.head.head_conv = 64

# loss
model.loss = contourdet_loss_region
model.loss.fd_weight = 5
model.loss.hm_weight = 1
model.loss.rmax_weight = 1
model.loss.reg_weight = 1

# dataset setting
data.num_classes = 15
data.class_name =  ['plane',
                    'baseball-diamond',
                    'bridge',
                    'ground-track-field',
                    'small-vehicle',
                    'large-vehicle',
                    'ship',
                    'tennis-court',
                    'basketball-court',
                    'storage-tank',
                    'soccer-ball-field',
                    'roundabout',
                    'harbor',
                    'swimming-pool',
                    'helicopter']

data.gt_path = '/data/ZG/dataset/Unify_dataset/DOTA_unify/val_wholeimage/val_wholeimage.json'
data.imagelist = None
data.input_w = 640
data.input_h = 640

# testing and evaling
test.enable = True
test.visible = True
test.test_scales = [1.5, 1, 0.5]
test.patch_size = 1024
test.patch_overlap = 256

val.valer = 'val_unify1D'

class config(object):
    common = common
    data = data
    model = model
    train = train
    test = test
