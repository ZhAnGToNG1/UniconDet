from detect.detector_adapter import make_valer
import os
from NV10_val_eval import nv10_main


class eval_nv10(object):
    def __init__(self, cfg, dataloder):
        self.cfg = cfg
        self.save_path = cfg.val.inter_save_path
        self.dataloader = dataloder

    def test_in_training(self, model):

        dst_dir = self.save_path
        if os.path.exists(dst_dir):
            for file in os.listdir(dst_dir):
                os.remove(os.path.join(dst_dir, file))
        else:
            os.mkdir(dst_dir)

        validator = make_valer(self.cfg)
        model.eval()

        images_num = len(self.dataloader)
        for iter_id, batches in enumerate(self.dataloader):
            if iter_id > images_num:
                break
            img_path = batches['input'][0]
            img_name = batches['img_name'][0].split('.jpg')[0]
            ret = validator.run(img_path, model)
            results = ret['results']
            for cls_id in results:
                if len(results[cls_id]) == 0:
                    continue
                category = self.cfg.data.class_name[cls_id-1]
                dst_path = os.path.join(dst_dir, 'Task1_' + category + '.txt')
                if os.path.exists(dst_path):
                    fo = open(dst_path, 'a')
                else:
                    fo = open(dst_path, 'w')

                for obj in results[cls_id]:
                    x1, y1, x2, y2, x3, y3, x4, y4, score = obj
                    fo.write("%s %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % (
                        img_name, score, x1, y1, x2, y2, x3, y3, x4, y4))
                fo.close()

        return nv10_main(dst_dir, gt_path=self.cfg.data.gt_path,
                         imagesetfile=self.cfg.data.imagelist,
                         classnames=self.cfg.data.class_name)