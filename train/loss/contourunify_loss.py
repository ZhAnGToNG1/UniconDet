from .losses import FocalLoss_hm, SmoothL1Loss, FourierLoss, Polarmask_loss,FourierLoss_weighted
import torch
import torch.nn as nn
from utils.tool import _sigmoid

class ContourUnify_loss(nn.Module):
    def __init__(self, cfg):
        super(ContourUnify_loss,self).__init__()
        self.hm_func = FocalLoss_hm()
        self.reg_func = SmoothL1Loss()
        self.fourier_func = FourierLoss(cfg)
        self.cfg = cfg

    def forward(self, outputs, batches):

        scalar_status = {}
        hm_loss, reg_loss, fourier_loss= 0, 0, 0
        num_layer = 0

        for i in range(len(outputs)):
            output = outputs[i]
            batch = batches[i]

            num_layer += 1

            hm_loss += self.hm_func(_sigmoid(output['hm']), batch['hm'])
            reg_loss += self.reg_func(output['reg'], batch['reg_mask'], batch['ind'], batch['reg'])
            fourier_loss += self.fourier_func(output['fourier'], batch['reg_mask'], batch['ind'], batch['fourier'])

        scalar_status.update({'hm_loss'     : hm_loss/num_layer})
        scalar_status.update({'reg_loss'    : reg_loss/num_layer})
        scalar_status.update({'fourier_loss': fourier_loss/num_layer})

        loss = self.cfg.model.loss.hm_weight * scalar_status['hm_loss'] + \
               self.cfg.model.loss.fd_weight * scalar_status['fourier_loss'] + \
               self.cfg.model.loss.reg_weight * scalar_status['reg_loss']


        scalar_status.update({'loss': loss})
        return loss, scalar_status