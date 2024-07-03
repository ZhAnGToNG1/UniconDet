from .losses import FocalLoss_hm, SmoothL1Loss, FourierLoss_rmax, Polarmask_loss,FourierLoss_rmax_weight
import torch
import torch.nn as nn
from utils.tool import _sigmoid

class ContourPoint_loss(nn.Module):
    def __init__(self, cfg):
        super(ContourPoint_loss,self).__init__()
        self.hm_func = FocalLoss_hm()
        self.reg_func = SmoothL1Loss()
        self.contourpoint_func = SmoothL1Loss()
        self.cfg = cfg

    def forward(self, outputs, batches):

        scalar_status = {}

        hm_loss, reg_loss, contour_loss, rmax_loss = 0, 0, 0, 0
        num_layer = 0

        for i in range(len(outputs)):
            output = outputs[i]
            batch = batches[i]

            num_layer += 1

            hm_loss += self.hm_func(_sigmoid(output['hm']), batch['hm'])
            reg_loss += self.reg_func(output['reg'], batch['reg_mask'], batch['ind'], batch['reg'])
            contour_loss += self.contourpoint_func(output['contour'],batch['reg_mask'],batch['ind'], batch['contour'])
            #rmax_loss += self.rmax_func(output['rmax'], batch['reg_mask'], batch['ind'], batch['rmax'])

        scalar_status.update({'hm_loss'     : hm_loss/num_layer})
        scalar_status.update({'reg_loss'    : reg_loss/num_layer})
        scalar_status.update({'contour_loss': contour_loss/num_layer})
       # scalar_status.update({'rmax_loss'   : rmax_loss/num_layer})

        loss = self.cfg.model.loss.hm_weight * scalar_status['hm_loss'] + \
               self.cfg.model.loss.fd_weight * scalar_status['contour_loss'] + \
               self.cfg.model.loss.reg_weight * scalar_status['reg_loss'] #+ \
               #self.cfg.model.loss.rmax_weight * scalar_status['rmax_loss']

        scalar_status.update({'loss': loss})
        return loss, scalar_status