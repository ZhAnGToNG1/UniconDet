import numpy as np

class common(object):
    task = 'det'
    task_name = None
    down_ratio = 4
    recorder_dir = './experiments/'
    model_dir = './experiments/'
    device = 'cuda'
    dota_test_res_dir = None

class train(object):
    epoch = 50
    save_ep = [10, 20, 30, 40, 50, 60, 70, 80]
    save_every_ep = False
    eval_eps = 5
    start_eval_ep = 30
    dp_training = False
    gpus = None
    batch_size = 4
    num_workers = 0
    resume = False

    trainer = 'contourdet'

    optimizer = {'name': 'adam', 'lr': 1e-4,
                 'weight_decay': 5e-4,
                 'milestones': [50, 80, ],
                 'gamma': 0.5}

    seed = 317
    deterministic = True

    # augmentation
    random_scale = True
    scale_range = [0.6, 1.4]

    random_shift = True
    shift_distance = 128

    random_rotate = True
    rotate_angles = [0, 90, 180, 270]

    horz_flip = 0.5

    random_color = True

    multi_output_layers = False
    output_layers = [4, 8, 16]

    fd = 15  # fourier degree TODO: assign different fd for different layers
    ns = 180  # ns means number_samples


class data(object):

    dataset = 'nv10'
    num_classes = 10
    class_name = ["airplane", "ship", "storage-tank", "baseball-diamond",
                  "tennis-court", "basketball-court", "ground-track-field",
                  "harbor", "bridge", "vehicle"]

    gt_path = None
    imagelist = None

    input_w = 512
    input_h = 512
    mean = np.array([0.339, 0.360, 0.358],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.181, 0.185, 0.192],
                   dtype=np.float32).reshape(1, 1, 3)

    data_rng = np.random.RandomState(123)
    eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                       dtype=np.float32)
    eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)

    down_ratio = common.down_ratio


class model(object):
    # parameters
    name = None
    backbone = None
    neck = None
    head = None
    loss = None

    # path
    resume_path = None


class test(object):

    test_scales = [1]
    patch_size = 512
    patch_overlap = 128
    K = 100
    nms_T = 0.2
    threshold = 0.1
    visible = False
    enable = True


class val(object):

    valer = 'val_det'
    K = 100
    nms_T = 0.2
    threshold = 0.1
    inter_save_path = None

class distill(object):
    distill_type = 'linear'
    teacher_weights = None





