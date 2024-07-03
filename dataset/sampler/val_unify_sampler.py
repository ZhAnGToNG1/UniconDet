import torch.utils.data as data
import numpy as np
import cv2
import os

class val_unify_sampler(data.Dataset):
    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)
        height, width = img.shape[0], img.shape[1]

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = np.array([img.shape[1], img.shape[0]], dtype=np.float32)

        meta = {'img_id': img_id, 'center': c, 'scale': s}

        ret = {'input': image_path,
               'meta': meta,
               'height': height,
               'width': width}

        return ret
