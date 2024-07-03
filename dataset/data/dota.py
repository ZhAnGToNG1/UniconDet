import pycocotools.coco as coco
import torch.utils.data as data

class DOTA(data.Dataset):
    num_classes = 15

    def __init__(self, cfg, split):
        super(DOTA, self).__init__()
        self.cfg = cfg

        if split == 'val':
            self.img_path = '/data/ZG/dataset/Unify_dataset/DOTA_unify/val_wholeimage/images'
            self.ann_path = '/data/ZG/dataset/Unify_dataset/DOTA_unify/val_wholeimage/val_wholeimage.json'

        if split == 'train':
            self.img_path = '/data/ZG/dataset/Rotate_dataset/DOTA/trainval640/images'
            self.ann_path = '/data/ZG/dataset/Rotate_dataset/DOTA/trainval640/train_contour.json'

        self.max_objs = 350
        self.class_name = [
            '__background__',
            'plane',
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
        self._valid_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.split = split

        print('\nInitializing DOTA_obb {} data.'.format(split))
        self.coco = coco.COCO(self.ann_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def __len__(self):
        return self.num_samples