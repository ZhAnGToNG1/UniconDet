import torch
import cv2
import numpy as np

color_category = np.array([[0,0,255],[0,255,0],[2,124,50],[0,128,0],[255,0,0],[128,0,255],[0,255,0],[255,128,0],[128,255,0],
                           [255,0,255],[255,0,128],[0,255,0],[2,202,202],[22,22,255],[0,255,128],[145,240,2],[245,90,150],[211,4,4],
                           [222,111,22],[234,12,33]])

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    # feat:[128,128,2]  ->  be same with gt  [self.max_obj,2]
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def plot_points(img,res_poly):

    c = (255,0,0)
    for point in res_poly:
        #c += 2
        point = point.astype(int)
        cv2.circle(img, tuple(point), 1, c, 1)

    return img


def draw_bbox_for_seg(contour, score, image, cat, cat_list, show_Txt = True):

    contour = np.array(contour, dtype=np.float32)
    poly = contour.reshape(-1, 2)
    score = str(round(float(score), 2))
    x1, y1 = poly[0]
    cat = int(cat)
    label = cat_list[cat]
    cv2.putText(image, score + '_' + label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1,
                lineType=cv2.LINE_AA)

    c = color_category[cat]
    c = c.tolist()

    for point in poly:
        point = point.astype('int16')
        cv2.circle(image, tuple(point), 1, c, 1)
    return image

def draw_bbox_for_hbbobb(contour, score, image, cat, cat_list, show_Txt = True):

    contour = np.array(contour, dtype=np.float32)
    poly = contour.reshape(-1, 2)
    score = str(round(float(score), 2))
    x1, y1 = poly[0]
    cat = int(cat)
    label = cat_list[cat]
    # cv2.putText(image, score + '_' + label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1,
    #             lineType=cv2.LINE_AA)

    c = color_category[cat]
    c = c.tolist()

    rect = cv2.minAreaRect(poly)
    bbox = cv2.boxPoints(rect)

    cv2.line(image, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), c, 2)
    cv2.line(image, (bbox[1][0], bbox[1][1]), (bbox[2][0], bbox[2][1]), c, 2)
    cv2.line(image, (bbox[2][0], bbox[2][1]), (bbox[3][0], bbox[3][1]), c, 2)
    cv2.line(image, (bbox[3][0], bbox[3][1]), (bbox[0][0], bbox[0][1]), c, 2)

    return image

import json
def save_result(cfg, mAP_log, epoch, print_coco, voc_05map):
    with open(mAP_log, 'a') as fp:
        log = {}
        log['epoch'] = str(epoch)

        list = print_coco.split('\n')
        for each in list[:3]:
            iou = each.split('IoU=')[-1].split('|')[0]
            value = each.split('=')[-1]
            log[iou.rstrip()] = value

        for index in range(len(cfg.data.class_name)):
            log[cfg.data.class_name[index]] = str(voc_05map[index])

        output = ''
        for key in log:
            output = output + key + ":" + log[key] + ' '
        output = output + '\n'
        fp.write(output)