import torch.utils.data as data
import os
import math
import cv2
import numpy as np
from ..utils.data_preprocess import Random_rotate, get_affine_transform, color_aug, affine_transform, guassian_radius, draw_umich_gaussian
from ..utils.fourier_process import fourPoint2fourier


class DetDataset(data.Dataset):

    def __getitem__(self, index):
        img_id = self.images[index]
        img_path = os.path.join(self.img_path, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]


        assert height == self.cfg.data.input_h and width == self.cfg.data.input_w, 'predefined width(height) is not equtal to the width(height) of this image' + img_path

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0


        # data augmentation
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

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        out_h, out_w = height // down_ratio, width // down_ratio

        hm = np.zeros((self.num_classes, out_h, out_w))
        fourier_coding = np.zeros((self.max_objs, (2 * self.cfg.train.fd + 1) * 2), dtype=np.float32)  # 26 = (2 * fd + 1) *2
        rmax = np.zeros((self.max_objs, 1), dtype=np.float32)  # 26 = (2 * fd + 1) *2
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        matrix_output = get_affine_transform(c, s, 0, (out_w, out_h))

        for k in range(num_objs):
            ann = anns[k]
            cls_id = int(self.cat_ids[ann['category_id']])

            poly = np.array(ann['bbox']).reshape(-1, 2)

            if self.cfg.train.random_rotate:
                poly = rot_ins.rotate_label(poly)
            else:
                poly = poly
            rx0, ry0, rx1, ry1, rx2, ry2, rx3, ry3 = poly.reshape(-1,8).squeeze().tolist()
            # if self.cfg.train.random_rotate:
            #     rx0, ry0, rx1, ry1, rx2, ry2, rx3, ry3 = rot_ins.rotate_label(ann['bbox'])
            # else:
            #     rx0, ry0, rx1, ry1, rx2, ry2, rx3, ry3 = ann['bbox']

            if fliped_flag:
                rx0, rx1, rx2, rx3 = width - rx0 - 1, width - rx1 - 1, width - rx2 - 1, width - rx3 - 1

            rx0, ry0 = affine_transform([rx0, ry0], matrix_output)
            rx1, ry1 = affine_transform([rx1, ry1], matrix_output)
            rx2, ry2 = affine_transform([rx2, ry2], matrix_output)
            rx3, ry3 = affine_transform([rx3, ry3], matrix_output)

            #----------- poly -------
            points = np.array([(rx0, ry0), (rx1, ry1), (rx2, ry2), (rx3, ry3)], dtype=np.float32)
            coors = cv2.boxPoints(cv2.minAreaRect(points))
            coors = np.squeeze(coors.reshape(1, -1), axis=0).tolist()
            rx0, ry0, rx1, ry1, rx2, ry2, rx3, ry3 = coors
            #------------------------

            xmin, ymin, xmax, ymax = min(rx0, rx1, rx2, rx3), min(ry0, ry1, ry2, ry3), max(rx0, rx1, rx2, rx3), max(ry0, ry1, ry2, ry3)

            # center point (x,y) and width and height (w,h): used for calculating the gaussian kernel
            center_x, center_y = (xmax + xmin) / 2, (ymax + ymin) / 2
            h = ymax - ymin
            w = xmax - xmin

            if 0 <= center_x <= out_w - 1 and 0 <= center_y <= out_h - 1 and h > 2 and w > 2:
                radius = max(0, int(guassian_radius((math.ceil(h), math.ceil(w)))))
                ct = np.array([center_x, center_y], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(hm[cls_id], ct_int, radius)

                ind[k] = ct_int[1] * out_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1

                fourier_signture, r_max, _ = fourPoint2fourier([center_x, center_y],
                                                               [rx0, ry0, rx1, ry1, rx2, ry2, rx3, ry3],
                                                               fd=self.cfg.train.fd,
                                                               ns=self.cfg.train.ns)
                fourier_coding[k] = fourier_signture
                rmax[k] = r_max

        ret = {'input': inp,
               'hm': hm,
               'fourier': fourier_coding,
               'rmax': rmax,
               'reg': reg,
               'reg_mask': reg_mask,
               'ind': ind}
        return ret



