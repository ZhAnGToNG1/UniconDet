import torch.utils.data as data
import os


class val_sampler(data.Dataset):

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids = [img_id])[0]['file_name']
        image_path = os.path.join(self.img_path, file_name)
        ret = {'input': image_path, 'img_name': file_name}

        return ret






