import numpy as np
import cv2
import random
import pycocotools.mask as mask_utils

class Random_rotate():

    def __init__(self,angle,default_h,default_w):
        self.angle = angle
        self.pi_angle = -angle * np.pi / 180
        self.default_h = default_h
        self.default_w = default_w

    def rotate_img(self, img):
        '''

        :param img:
        :return: out_img
        '''

        if len(img.shape) == 3:
            h, w, _ = img.shape
        else:
            h, w = img.shape
        assert h == self.default_h and w == self.default_w, 'the h of img is different with default_h or ' \
                                                            'the w of img is different with default_w'
        M = cv2.getRotationMatrix2D((w / 2, h / 2), self.angle, 1)
        out_img = cv2.warpAffine(img, M, (w, h))
        return out_img

    def rotate_label(self, bbox):
        '''
            :param bbox:  orginal object bounding box
            :return: out_bbox
        '''

        a = self.default_w / 2
        b = self.default_h / 2

        new_bbox = []
        for coor in bbox:
            x, y = coor
            x_new = (x - a) * np.cos(self.pi_angle) - (y - b) * np.sin(self.pi_angle) + a
            y_new = (x - a) * np.sin(self.pi_angle) + (y - b) * np.cos(self.pi_angle) + b
            new_bbox.append([x_new, y_new])

        new_bbox = np.array(new_bbox)
        return new_bbox

    def rotate_label_bak(self, bbox):
        '''
        :param bbox:  orginal object bounding box
        :return: out_bbox
        '''

        x0, y0, x1, y1, x2, y2, x3, y3 = bbox

        a = self.default_w / 2
        b = self.default_h / 2

        X0 = (x0 - a) * np.cos(self.pi_angle) - (y0 - b) * np.sin(self.pi_angle) + a
        Y0 = (x0 - a) * np.sin(self.pi_angle) + (y0 - b) * np.cos(self.pi_angle) + b

        X1 = (x1 - a) * np.cos(self.pi_angle) - (y1 - b) * np.sin(self.pi_angle) + a
        Y1 = (x1 - a) * np.sin(self.pi_angle) + (y1 - b) * np.cos(self.pi_angle) + b

        X2 = (x2 - a) * np.cos(self.pi_angle) - (y2 - b) * np.sin(self.pi_angle) + a
        Y2 = (x2 - a) * np.sin(self.pi_angle) + (y2 - b) * np.cos(self.pi_angle) + b

        X3 = (x3 - a) * np.cos(self.pi_angle) - (y3 - b) * np.sin(self.pi_angle) + a
        Y3 = (x3 - a) * np.sin(self.pi_angle) + (y3 - b) * np.cos(self.pi_angle) + b
        out_bbox = [X0,Y0,X1,Y1,X2,Y2,X3,Y3]
        return out_bbox




def transform_preds(coords,center,scale,output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center,scale,0,output_size,inv=1)
    for p in range(coords.shape[0]):
        target_coords[p,0:2] = affine_transform(coords[p,0:2],trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale,np.ndarray) and not isinstance(scale,list):
        scale = np.array([scale, scale],dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.pi * rot/180
    src_dir = get_dir([0,src_w * -0.5],rot_rad)
    dst_dir = np.array([0,dst_w * -0.5],np.float32)

    src = np.zeros((3,2),dtype=np.float32)
    dst = np.zeros((3,2),dtype=np.float32)

    src[0,:] = center + scale_tmp * shift
    src[1,:] = center + src_dir + scale_tmp * shift

    dst[0,:] = [dst_w * 0.5, dst_h * 0.5]
    dst[1,:] = np.array([dst_w * 0.5,dst_h * 0.5],np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform_seg_eval(pt, t):
    """pt: [n, 2]"""
    new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
    return new_pt

def affine_transform(pt,t):
    new_pt = np.array([pt[0], pt[1], 1.],dtype=np.float32).T
    new_pt = np.dot(t,new_pt)
    return new_pt[:2]

def coco_poly_to_rle(poly, h, w):
    rle_ = []
    for i in range(len(poly)):
        rles = mask_utils.frPyObjects([poly[i].reshape(-1)], h, w)
        rle = mask_utils.merge(rles)
        rle['counts'] = rle['counts'].decode('utf-8')
        rle_.append(rle)
    return rle_



def get_3rd_point(a,b):
    direct = a - b
    return b + np.array([-direct[1],direct[0]],dtype=np.float32)



def get_dir(src_point,rot_rad):
    sn , cs = np.sin(rot_rad) , np.cos(rot_rad)
    src_result = [0,0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result



def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)



def guassian_radius(det_size, min_overlap= 0.7):
    #print("运行guassian_radius")
    height , width = det_size

    a1 = 1
    b1 = (height + width)
    # 面积 * 交并比
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 *c1)
    r1 = (b1 + sq1) /2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 *c2)
    r2 = (b2 + sq2 ) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_mask(masks,mask):
    masks[mask > 0] = mask[mask > 0]

    return mask

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap