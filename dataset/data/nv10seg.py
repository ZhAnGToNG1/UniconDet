import pycocotools.coco as coco
import torch.utils.data as data

class NV10seg(data.Dataset):
    num_classes = 10

    def __init__(self, cfg, split):
        super(NV10seg, self).__init__()
        self.cfg = cfg

        if split == 'val':
            self.img_path = '/data/ZG/dataset/Rotate_dataset/NWPUVHR-10_Seg/test/images'
            self.ann_path = '/data/ZG/dataset/Rotate_dataset/NWPUVHR-10_Seg/test/test_contour.json'

        if split == 'train':
            # self.img_path = '/data/ZG/dataset/Rotate_dataset/NWPUVHR-10_Seg/debug/images'
            # self.ann_path = '/data/ZG/dataset/Rotate_dataset/NWPUVHR-10_Seg/debug/test_contour.json'
            #
            self.img_path = '/data/ZG/dataset/Rotate_dataset/NWPUVHR-10_Seg/train/images'
            self.ann_path = '/data/ZG/dataset/Rotate_dataset/NWPUVHR-10_Seg/train/train_contour.json'

        self.max_objs = 80
        self.class_name = ['__background__', "airplane", "ship", "storage-tank", "baseball-diamond",
                           "tennis-court", "basketball-court", "ground-track-field", "harbor", "bridge", "vehicle"]
        self._valid_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.split = split

        print('\nInitializing NV10Seg {} data.'.format(split))
        self.coco = coco.COCO(self.ann_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def __len__(self):
        return self.num_samples