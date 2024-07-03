from detect.detector_adapter import make_valer
import os
from tqdm import tqdm
import numpy as np
import json
from data_preprocess import affine_transform_seg_eval, get_affine_transform, coco_poly_to_rle
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

class eval_unify(object):
    def __init__(self, cfg, dataloder):
        self.cfg = cfg
        self.save_path = cfg.val.inter_save_path
        self.dataloader = dataloder
        self.validator = make_valer(self.cfg)

    def test_in_training(self, model):

        dst_dir = self.save_path
        if os.path.exists(dst_dir):
            for file in os.listdir(dst_dir):
                os.remove(os.path.join(dst_dir, file))
        else:
            os.makedirs(dst_dir)

        evalator = Evaluator(self.cfg, self.cfg.val.inter_save_path, self.cfg.data.gt_path)
        model.eval()

        images_num = len(self.dataloader)
        for iter_id, batch in enumerate(tqdm(self.dataloader)):
            if iter_id > images_num:
                break
            img_path = batch['input'][0]
            ret = self.validator.run(img_path, model)
            for i in ret['results']:
                poly_score = ret['results'][i]
                if len(poly_score) < 0:
                    continue
                label = np.ones([poly_score.shape[0], 1]) * i
                if i == 1:
                    output = np.concatenate([poly_score, label], axis=-1)
                else:
                    output = np.concatenate([output, np.concatenate([poly_score, label], axis=-1)], axis=0)
            evalator.evaluate(output, batch)
        return evalator.summarize()


class Evaluator:
    def __init__(self, cfg, result_dir, anno_dir):
        self.results = []
        self.img_ids = []
        self.aps = []
        self.cfg = cfg
        self.result_dir = result_dir
        ann_file = anno_dir
        self.coco = coco.COCO(ann_file)

    def evaluate(self, output, batch):
        score = output[:, 2 * self.cfg.train.ns]
        label = output[:, 2 * self.cfg.train.ns + 1].astype(int)
        py = output[:, :2 * self.cfg.train.ns]

        if len(py) == 0:
            return

        img_id = int(batch['meta']['img_id'])
        center = batch['meta']['center'].detach().cpu().numpy()
        scale = batch['meta']['scale'].detach().cpu().numpy()

        h, w = batch['height'], batch['width']
        trans_output_inv = get_affine_transform(center[0], scale[0], 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']
        py = py.reshape(-1, self.cfg.train.ns, 2)
        py = [affine_transform_seg_eval(py_, trans_output_inv) for py_ in py]
        rles = coco_poly_to_rle(py, ori_h, ori_w)

        # -- vis the mask
        # import pycocotools.mask as mask_utils
        # import numpy as np
        # m = mask_utils.decode(rles)
        # mt = np.sum(m, axis=-1)
        # import matplotlib.pyplot as plt
        # plt.imshow(mt)
        # plt.show()

        # -- vis the poly
        # import cv2
        # img_dir = '/root/data/Rotate_data/iSAID/Devkit/preprocess/datasetpart/iSAID_filter/train/images'
        # img_path = os.path.join(img_dir, img['file_name'])
        # image = cv2.imread(img_path)
        # from utils.add_rotate_bbox import draw_bbox_for_seg
        # for i in range(len(label)):
        #     img = draw_bbox_for_seg(output[i][:361], image, label[i]-1)
        # # cv2.imwrite('/root/a.png', img)

        #
        coco_dets = []
        for i in range(len(rles)):
            detection = {
                'image_id': img_id,
                'category_id': int(label[i]),
                'segmentation': rles[i],
                'score': float('{:.2f}'.format(score[i]))
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        coco_eval = COCOeval(self.coco, coco_dets, 'segm')
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        stats, print_coco, ap50 = coco_eval.summarize()
        class_index = []
        for cat_id in sorted(self.coco.cats):
            class_index.append(self.coco.cats[cat_id]['name'])

        voc_05map = []
        for i in range(len(class_index)):
            stats, _, _ = coco_eval.summarize(catId=i)
            voc_05map.append("%s: %.5f" % (class_index[i], stats[1]))

        print("COCO results:\n", print_coco)
        for class_ap in voc_05map:
            print(class_ap)

        return voc_05map, print_coco, ap50