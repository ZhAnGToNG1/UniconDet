import torch.utils.data as data
import os
import math
import cv2
import numpy as np
from ..utils.data_preprocess import Random_rotate, get_affine_transform, color_aug, affine_transform, guassian_radius, draw_umich_gaussian
from ..utils.fourier_process import contour2fourier


class SegDataset(data.Dataset):

    def __getitem__(self, index):
        img_id = self.images[index]
        img_path = os.path.join(self.img_path, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]

        if height != self.cfg.data.input_h or width != self.cfg.data.input_w:
            if height > self.cfg.data.input_h:
                height = self.cfg.data.input_h
                img = img[:self.cfg.data.input_h, :, :]
            if width > self.cfg.data.input_w:
                width = self.cfg.data.input_w
                img = img[:, :self.cfg.data.input_w, :]
            img_format = np.zeros([self.cfg.data.input_h, self.cfg.data.input_w, 3])
            img_format[:height, :width, :] = img
            img = img_format
            height, width = img.shape[0], img.shape[1]

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0

        assert height == self.cfg.data.input_h and width == self.cfg.data.input_w, 'predefined width(height) is not equtal to the width(height) of this image' + img_path

        if self.cfg.train.random_scale:
            s = s * np.random.choice(np.arange(self.cfg.train.scale_range[0], self.cfg.train.scale_range[1], 0.1))

        if self.cfg.train.random_shift:
            c[0] = np.random.randint(low=128, high=img.shape[1] - 128)
            c[1] = np.random.randint(low=128, high=img.shape[0] - 128)

        rot_ins = None
        if self.cfg.train.random_rotate:
            angle = np.random.choice(self.cfg.train.rotate_angles)
            rot_ins = Random_rotate(angle, height, width)
            img = rot_ins.rotate_img(img)

        fliped_flag = False
        if np.random.random() < float(self.cfg.train.horz_flip):
            fliped_flag = True
            img = img[:, ::-1, :]
            c[0] = width - c[0] - 1

        affine_matrix = get_affine_transform(c, s, 0, (self.cfg.data.input_w, self.cfg.data.input_h))
        inp = cv2.warpAffine(img, affine_matrix, (self.cfg.data.input_w, self.cfg.data.input_h), flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)

        if self.cfg.train.random_color:
            color_aug(self.cfg.data.data_rng, inp, self.cfg.data.eig_val, self.cfg.data.eig_vec)

        inp = (inp - self.cfg.data.mean) / self.cfg.data.std
        inp = inp.transpose(2, 0, 1)

        ret = []
        if self.cfg.train.multi_output_layers:
            for down_ratio in self.cfg.train.output_layers:
                ret.append(self._obtain_ms_regression(down_ratio, img, img_id, c, s, height, width, rot_ins, fliped_flag, inp, affine_matrix))

        else:
            ret.append(
                self._obtain_ms_regression(self.cfg.common.down_ratio, img, img_id, c, s, height, width, rot_ins, fliped_flag, inp,
                                           affine_matrix))

        return ret


    def _obtain_ms_regression(self,down_ratio, img, img_id, c, s, height, width, rot_ins, fliped_flag, inp, affine_matrix):
        meta = {'img_id': img_id, 'center': c, 'scale': s}

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        out_h, out_w = height // down_ratio, width // down_ratio

        hm = np.zeros((self.num_classes, out_h, out_w))
        fourier_coding = np.zeros((self.max_objs, (2 * self.cfg.train.fd + 1) * 2), dtype=np.float32)  # 26 = (2 * fd + 1) *2
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        matrix_output = get_affine_transform(c, s, 0, (out_w, out_h))

        for k in range(num_objs):
            ann = anns[k]
            cls_id = int(self.cat_ids[ann['category_id']])

            if self.cfg.train.random_rotate:
                rx0, ry0, rx1, ry1, rx2, ry2, rx3, ry3 = rot_ins.rotate_label(ann['bbox'])
            else:
                poly = np.array(ann['segmentation'][0]).reshape(-1, 2)

            if fliped_flag:
                poly[:, 0] = width - poly[:, 0] - 1

            for i in range(len(poly)):
                poly[i] = affine_transform(poly[i], matrix_output)

            #----------- poly -------
            poly = poly.astype('float32')
            rect = cv2.minAreaRect(poly)
            center = np.array(list(rect[0])).astype('float32')
            wh = np.array(list(rect[1])).astype('float32')
            #------------------------

            if 0 <= center[0] <= out_w - 1 and 0 <= center[1] <= out_h - 1 and len(poly) > (2 * self.cfg.train.fd + 1) * 2:
                radius = max(0, int(guassian_radius((math.ceil(wh[1]), math.ceil(wh[0])))))
                ct = np.array([center[0], center[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(hm[cls_id-1], ct_int, radius)

                ind[k] = ct_int[1] * out_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1

                fourier_coding[k] = contour2fourier([center[0], center[1]], poly, fd=self.cfg.train.fd)

        ret = {'input': inp,
               'hm': hm,
               'fourier': fourier_coding,
               'reg': reg,
               'reg_mask': reg_mask,
               'ind': ind,
               'meta': meta}
        return ret



