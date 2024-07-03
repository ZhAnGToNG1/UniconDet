import pycocotools.coco as coco
import torch.utils.data as data

class DIOR_HBB(data.Dataset):
    num_classes = 20

    def __init__(self, cfg, split):
        super(DIOR_HBB, self).__init__()
        self.cfg = cfg

        if split == 'val':
            self.img_path = '/data/ZG/dataset/HBB_dataset/DIOR/test/images'
            self.ann_path = '/data/ZG/dataset/HBB_dataset/DIOR/test/test_contour.json'

        if split == 'train':
            self.img_path = '/data/ZG/dataset/HBB_dataset/DIOR/train/images'
            self.ann_path = '/data/ZG/dataset/HBB_dataset/DIOR/train/train_contour.json'

        self.max_objs = 350
        self.class_name = [
            '__background__',
            'windmill', 'tenniscourt', 'baseballfield', 'vehicle', 'stadium', 'groundtrackfield', 'airport', 'overpass',
            'storagetank', 'harbor',
            'ship', 'bridge', 'basketballcourt', 'Expressway-Service-area', 'golffield', 'airplane', 'dam',
            'trainstation', 'Expressway-toll-station', 'chimney']
        self._valid_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.split = split

        print('\nInitializing DIOR_HBB {} data.'.format(split))
        self.coco = coco.COCO(self.ann_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def __len__(self):
        return self.num_samples