import torch.utils.data as data
import os
import math
import cv2
import numpy as np
from ..utils.data_preprocess import Random_rotate, get_affine_transform, color_aug, affine_transform, guassian_radius, \
    draw_umich_gaussian,draw_mask
from ..utils.fourier_process import Unify2fourier1D, sample_contour_theta_rmax, Unify2Cheby
from shapely.geometry import Polygon
import pycocotools.mask as mask_utils


def coco_poly_to_rle(poly, h, w):
    rle_ = []
    for i in range(len(poly)):
        rles = mask_utils.frPyObjects([poly[i].reshape(-1)], h, w)
        rle = mask_utils.merge(rles)
        rle['counts'] = rle['counts'].decode('utf-8')
        rle_.append(rle)
    return rle_


class UnifyDataset_Cheby(data.Dataset):
    def __getitem__(self, index):
        img_id = self.images[index]
        img_path = os.path.join(self.img_path, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        org_h, org_w = height, width
        if height != self.cfg.data.input_h and width != self.cfg.data.input_w:
            img = cv2.resize(img, (self.cfg.data.input_h, self.cfg.data.input_w))
            height, width = img.shape[0], img.shape[1]

        assert height == self.cfg.data.input_h and width == self.cfg.data.input_w, 'predefined width(height) is not equtal to the width(height) of this image' + img_path

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0

        # data augmentation
        if self.cfg.train.random_scale:
            s = s * np.random.choice(np.arange(self.cfg.train.scale_range[0], self.cfg.train.scale_range[1], 0.1))

        if self.cfg.train.random_shift:
            c[0] = np.random.randint(low=self.cfg.data.input_h // 4, high=img.shape[1] - self.cfg.data.input_h // 4)
            c[1] = np.random.randint(low=self.cfg.data.input_w // 4, high=img.shape[0] - self.cfg.data.input_w // 4)

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
                ret.append(self._obtain_ms_regression(down_ratio, img, img_id, c, s, height, width, rot_ins, fliped_flag, inp, org_w, org_h))

        else:
            ret.append(
                self._obtain_ms_regression(self.cfg.common.down_ratio, img, img_id, c, s, height, width, rot_ins, fliped_flag, inp,
                                           org_w, org_h))
        return ret

    def _obtain_ms_regression(self, down_ratio, img, img_id, c, s, height, width, rot_ins, fliped_flag, inp, org_w,org_h):

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        out_h, out_w = height // down_ratio, width // down_ratio

        hm = np.zeros((self.num_classes, out_h, out_w))
        fourier_coding = np.zeros((self.max_objs, self.cfg.train.fd), dtype=np.float32)  # 26 = (2 * fd + 1) *2
        rmax = np.zeros((self.max_objs, 1), dtype=np.float32)  # 26 = (2 * fd + 1) *2
        region = np.zeros((self.num_classes, out_h, out_w))

        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        matrix_output = get_affine_transform(c, s, 0, (out_w, out_h))

        # image = cv2.warpAffine(img, matrix_output,
        #                     (out_w, out_h),
        #                     flags=cv2.INTER_LINEAR)

        for k in range(num_objs):
            ann = anns[k]
            cls_id = int(self.cat_ids[ann['category_id']]) - 1

            if self.cfg.data.input_h != org_h or self.cfg.data.input_w != org_w:
                poly = np.array(ann['segmentation'][0]).reshape(-1, 2) * (self.cfg.data.input_w / org_w)
            else:
                poly = np.array(ann['segmentation'][0]).reshape(-1, 2)

            if self.cfg.train.random_rotate:
                poly = rot_ins.rotate_label(poly)
            else:
                poly = poly

            if fliped_flag:
                poly[:, 0] = width - poly[:, 0] - 1

            for i in range(len(poly)):
                poly[i] = affine_transform(poly[i], matrix_output)

            ns, _ = poly.shape
            pts = poly.reshape(-1, ns, 2)
            rles = coco_poly_to_rle(pts, out_h, out_w)
            m = mask_utils.decode(rles).squeeze(-1)
            mask = np.zeros((out_h, out_w), dtype=np.float32)
            m = m.astype(np.int32) * 1
            mask[m > 0] = m[m > 0]

            #----------- poly -------
            poly = poly.astype('float32')
            rect = cv2.minAreaRect(poly)
            center = np.array(list(rect[0])).astype('float32')
            wh = np.array(list(rect[1])).astype('float32')
            #------------------------

            # from tool import plot_points
            # plot_points(image, poly)
            # cv2.circle(image, tuple(center), 1, (0, 255, 0), 4)
            # coors = cv2.boxPoints(rect)
            # coors = np.squeeze(coors.reshape(1, -1), axis=0).tolist()
            # rx0, ry0, rx1, ry1, rx2, ry2, rx3, ry3 = coors
            #
            # cv2.line(image, (int(rx0), int(ry0)), (int(rx1), int(ry1)), [0,255,255], 1)
            # cv2.line(image, (int(rx1), int(ry1)), (int(rx2), int(ry2)), [0,255,255], 1)
            # cv2.line(image, (int(rx2), int(ry2)), (int(rx3), int(ry3)), [0,255,255], 1)
            # cv2.line(image, (int(rx3), int(ry3)), (int(rx0), int(ry0)), [0,255,255], 1)
            # cv2.circle(image,(int(center[0]),int(center[1])),2,(0,0,255),2)
            #---------------
            if len(poly) == 4:
                poly = Polygon(poly)
                theta_d = sample_contour_theta_rmax(poly, center, num_samples=self.cfg.train.ns)
            else:
                poly = Polygon(poly).buffer(0.001)
                theta_d = sample_contour_theta_rmax(poly, center, num_samples=self.cfg.train.ns)

            if 0 <= center[0] <= out_w - 1 and 0 <= center[1] <= out_h - 1 and wh[0] > 1 and wh[1] > 1 and len(theta_d) >= self.cfg.train.ns:
                radius = max(0, int(guassian_radius((math.ceil(wh[1]), math.ceil(wh[0])))))
                ct = np.array([center[0], center[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(hm[cls_id], ct_int, radius)

                ind[k] = ct_int[1] * out_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1

                fourier_signture, r_max = Unify2Cheby(theta_d, self.cfg.train.ns, self.cfg.train.fd)

                fourier_coding[k] = fourier_signture
                rmax[k] = r_max


                mask[mask > 0] = 1
                draw_mask(region[cls_id], mask)

                # --------------------------------
                # Reconstruct cheby modeling
                # from fourier_process import Polar2Cartesian, reconstruct_cheby
                #
                # res_poly = reconstruct_cheby(fourier_signture,sample_pts=self.cfg.train.ns, num_coefs=self.cfg.train.fd)
                #
                # #
                # before_points = res_poly * r_max / 10
                # #
                # after_points = Polar2Cartesian((center[0], center[1]), before_points, self.cfg.train.ns)
                # #
                # from tool import plot_points
                # plot_points(image, after_points)

                #  vis for contour - center
                # from fourier_process import fourier2poly
                # fd = self.cfg.train.fd
                # real_part = fourier_signture[0:2*fd+1]
                # image_part = fourier_signture[2*fd+1:]
                #
                # c = real_part + image_part * 1j
                # c = np.array(c).reshape(1, -1)
                #
                # res_poly = fourier2poly(c, recon_points=180).reshape(-1, 2)
                #
                # res_poly = res_poly + center
                # from tool import plot_points
                # plot_points(image, res_poly)
        #
        #
        #
        #
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()
        # # # # # plt.imshow(region[9])
        # # # # # plt.show()
        # print('s')


        ret = {'input': inp,
               'hm': hm,
               'fourier': fourier_coding,
               'rmax': rmax,
               'reg': reg,
               'reg_mask': reg_mask,
               'ind': ind,
               'region': region}
        return ret



