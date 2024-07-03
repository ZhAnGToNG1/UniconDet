import torch
import torch.nn as nn
from utils.tool import _tranpose_and_gather_feat
import torch.nn.functional as F
import numpy as np
import math

def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss_hm(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss_seg(pred, gt):

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss_weight(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        print('num_pos is zero')
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss)
    return loss,num_pos

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    num_pos = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -= all_loss
    return loss


def _slow_reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class FocalLoss_seg(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss_seg, self).__init__()
        self.neg_loss = _neg_loss_seg

    def forward(self, out, target):
        return self.neg_loss(out, target)



class FocalLoss_hm(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss_hm, self).__init__()
        self.neg_loss = _neg_loss_hm

    def forward(self, out, target):
        return self.neg_loss(out, target)



class FocalLoss_weight(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss_weight, self).__init__()
        self.neg_loss = _neg_loss_weight

    def forward(self, out, target):
        return self.neg_loss(out, target)

class RegLoss(nn.Module):
    '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        # print(ind)
        pred = _tranpose_and_gather_feat(output, ind)

        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        # loss = loss.cpu().detach().numpy()
        # loss = torch.from_numpy(loss)
        return loss



class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        # print(ind)
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.smooth_l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss



# --------------------------------------------------------
def fourier2poly(fourier_coeff,fourier_degree,num_sample):
    fourier_coeff = fourier_coeff.view(-1, 2 * (2 * fourier_degree + 1))
    real_maps = fourier_coeff[:, :2*fourier_degree +1]
    imag_maps = fourier_coeff[:, 2*fourier_degree+1:]

    device = real_maps.device
    k_vect = torch.arange(
        -fourier_degree,
        fourier_degree + 1,
        dtype=torch.float,
        device=device).view(-1, 1)
    i_vect = torch.arange(
        0, num_sample, dtype=torch.float, device=device).view(1, -1)

    transform_matrix = 2 * np.pi / num_sample * torch.mm(
        k_vect, i_vect)

    x1 = torch.einsum('ak, kn-> an', real_maps,
                      torch.cos(transform_matrix))
    x2 = torch.einsum('ak, kn-> an', imag_maps,
                      torch.sin(transform_matrix))
    y1 = torch.einsum('ak, kn-> an', real_maps,
                      torch.sin(transform_matrix))
    y2 = torch.einsum('ak, kn-> an', imag_maps,
                      torch.cos(transform_matrix))

    x_maps = x1 - x2
    y_maps = y1 + y2

    return x_maps, y_maps

from numpy.fft import ifft

def fourier2poly_v2(fourier_coeff, recon_points):
    a = np.zeros((len(fourier_coeff), recon_points), dtype='complex')
    k = (len(fourier_coeff[0]) - 1) // 2
    a[:, 0:k + 1] = fourier_coeff[:, k:]
    a[:, -k:] = fourier_coeff[:, :k]

    poly_complex = ifft(a) * recon_points
    polygon = np.zeros((len(fourier_coeff), recon_points, 2))
    polygon[:, :, 0] = poly_complex.real
    polygon[:, :, 1] = poly_complex.imag
    return polygon


class FourierLoss(nn.Module):
    def __init__(self, cfg):
        super(FourierLoss, self).__init__()
        self.fd = cfg.train.fd
        self.ns = cfg.train.ns

    def forward(self, output, mask, ind, target):
        mask = mask.view(-1)
        pred = _tranpose_and_gather_feat(output, ind)
        ft_x, ft_y = fourier2poly(target, fourier_degree=self.fd, num_sample=self.ns)
        ft_x_pre, ft_y_pre = fourier2poly(pred, fourier_degree=self.fd, num_sample=self.ns)
        mask = mask.unsqueeze(-1).expand_as(ft_x)
        loss_x = F.smooth_l1_loss(ft_x_pre * mask, ft_x * mask, size_average=False)
        loss_y = F.smooth_l1_loss(ft_y_pre * mask, ft_y * mask, size_average=False)
        loss = (loss_x + loss_y) / 2
        loss = loss / (mask.sum() + 1e-4)
        return loss

class FourierLoss_weighted(nn.Module):
    def __init__(self, cfg):
        super(FourierLoss_weighted, self).__init__()
        self.fd = cfg.train.fd
        self.ns = cfg.train.ns

    def forward(self, output, mask, ind, target, weighted):
        mask = mask.view(-1)
        # weighted = weighted.view(-1)
        pred = _tranpose_and_gather_feat(output, ind)
        ft_x, ft_y = fourier2poly(target, fourier_degree=self.fd, num_sample=self.ns)
        ft_x_pre, ft_y_pre = fourier2poly(pred, fourier_degree=self.fd, num_sample=self.ns)
        mask = mask.unsqueeze(-1).expand_as(ft_x)
        # weighted = weighted.unsqueeze(-1).expand_as(ft_x)
        loss_x = F.smooth_l1_loss(ft_x_pre * mask , ft_x * mask , size_average=False)
        loss_y = F.smooth_l1_loss(ft_y_pre * mask , ft_y * mask , size_average=False)
        loss = (loss_x + loss_y) / 2
        loss = loss / (mask.sum() + 1e-4)
        return loss





class FourierLoss_rmax_weight(nn.Module):
    def __init__(self, cfg):
        super(FourierLoss_rmax_weight, self).__init__()
        self.fd = cfg.train.fd
        self.ns = cfg.train.ns
        self.relu = nn.ReLU()

    def forward(self, output, mask, ind, target, weight):
        mask = mask.view(-1)
        _, fd, _, _ = output.shape
        w = weight.sum(dim=1).unsqueeze(dim=1).repeat(1, fd, 1, 1).float()
        output = w * output
        pred = _tranpose_and_gather_feat(output, ind)
        ft_x, _ = fourier2poly(target, fourier_degree=self.fd, num_sample=self.ns)
        ft_x_pre, _ = fourier2poly(pred, fourier_degree=self.fd, num_sample=self.ns)
        ft_x_pre = self.relu(ft_x_pre)
        mask = mask.unsqueeze(-1).expand_as(ft_x)
        loss = F.smooth_l1_loss(ft_x_pre * mask, ft_x * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class FourierLoss_rmax(nn.Module):
    def __init__(self, cfg):
        super(FourierLoss_rmax, self).__init__()
        self.fd = cfg.train.fd
        self.ns = cfg.train.ns

    def forward(self, output, mask, ind, target):
        mask = mask.view(-1)
        pred = _tranpose_and_gather_feat(output, ind)
        ft_x, _ = fourier2poly(target, fourier_degree=self.fd, num_sample=self.ns)
        ft_x_pre, _ = fourier2poly(pred, fourier_degree=self.fd, num_sample=self.ns)
        mask = mask.unsqueeze(-1).expand_as(ft_x)
        loss = F.smooth_l1_loss(ft_x_pre * mask, ft_x * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss



from fourier_process import f_series

class ChebyLoss_rmax(nn.Module):
    def __init__(self, cfg):
        super(ChebyLoss_rmax, self).__init__()
        self.coefs = cfg.train.fd
        self.ns = cfg.train.ns

    def forward(self, output, mask, ind, target):
        mask = mask.view(-1)
        pred = _tranpose_and_gather_feat(output, ind)

        # reconstruct cheby
        theta = torch.linspace(-1, 1, self.ns)[:-1].cuda()
        fi = f_series(theta, self.coefs - 1)

        pred = pred.reshape(-1, self.coefs)
        target = target.reshape(-1, self.coefs)
        # 0, ..., num_coefs-1 term
        pred_r = torch.mm(pred[:, :self.coefs], fi)
        gt_r = torch.mm(target[:, :self.coefs], fi)

        mask = mask.unsqueeze(-1).expand_as(gt_r)
        loss = F.smooth_l1_loss(pred_r * mask, gt_r * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class FourierLoss_rmax_R2C(nn.Module):
    def __init__(self, cfg):
        super(FourierLoss_rmax_R2C, self).__init__()
        self.fd = cfg.train.fd
        self.ns = cfg.train.ns

    def forward(self, output, mask, ind, target):
        mask = mask.view(-1)
        pred = _tranpose_and_gather_feat(output, ind)
        ft_x, _ = fourier2poly(target, fourier_degree=self.fd, num_sample=self.ns)
        ft_x_pre, _ = fourier2poly(pred, fourier_degree=self.fd, num_sample=self.ns)
        mask = mask.unsqueeze(-1).expand_as(ft_x)
        loss = F.smooth_l1_loss(ft_x_pre * mask, ft_x * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class ContrastiveFourierLoss(nn.Module):
    def __init__(self, cfg):
        super(ContrastiveFourierLoss, self).__init__()
        self.fd = cfg.train.fd
        self.ns = cfg.train.ns

    def forward(self, output, mask, ind, output1):
        mask = mask.view(-1)
        pred = _tranpose_and_gather_feat(output, ind)
        pred1 = _tranpose_and_gather_feat(output1, ind)
        ft_x, _ = fourier2poly(pred, fourier_degree=self.fd, num_sample=self.ns)
        ft_x1, _ = fourier2poly(pred1, fourier_degree=self.fd, num_sample=self.ns)
        mask = mask.unsqueeze(-1).expand_as(ft_x)
        loss = F.smooth_l1_loss(ft_x1 * mask, ft_x * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class Polarmask_loss(nn.Module):
    def __init__(self, cfg):
        super(Polarmask_loss, self).__init__()
        self.fd = cfg.train.fd
        self.ns = cfg.train.ns
        self.relu = nn.ReLU()

    def forward(self, output, mask, ind, target):
        mask = mask.view(-1)
        mask_one_channel = mask
        pred = _tranpose_and_gather_feat(output, ind)
        ft_x, _ = fourier2poly(target, fourier_degree=self.fd, num_sample=self.ns)
        ft_x_pre, _ = fourier2poly(pred, fourier_degree=self.fd, num_sample=self.ns)
        ft_x_pre = self.relu(ft_x_pre)
        ft_x_pre_2 = ft_x_pre.pow(2)
        ft_x_2 = ft_x.pow(2)

        total = torch.stack([ft_x_pre_2, ft_x_2], -1)
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0]

        loss = (l_max.sum(dim=1) / (l_min.sum(dim=1) + 1e-4)).log() * mask_one_channel
        loss = loss.sum() / (mask_one_channel.sum() + 1e-4)

        return loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.mse_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class CrossEntroy(nn.Module):
    def __init__(self):
        super(CrossEntroy, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)

        pred_remain = pred[mask == 1]
        target_remain = target[mask == 1]

        t = target_remain.long()
        p = pred_remain

        loss = F.cross_entropy(p, t, size_average = False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class DistributionFocalLoss(nn.Module):
    def __init__(self):
        super(DistributionFocalLoss,self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)

        pred_remain = pred[mask == 1]
        label_remain = target[mask == 1]

        disl = label_remain.long()
        disr = disl + 1

        wl = disr.float() - label_remain
        wr = label_remain - disl.float()

        loss = F.cross_entropy(pred_remain,disl,reduction='none') * wl \
               + F.cross_entropy(pred_remain,disr,reduction='none') * wr
        loss = loss.mean()
        return loss

class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_lfloss(pred * mask, target * mask, reduction='elementwise_mean')
        return loss




def _nll_loss(x,mu,sigma,sigma_const = 0.3):
    pi = torch.tensor(np.pi)
    Z = (2 * pi * (sigma + sigma_const) ** 2) ** 0.5
    probability_density = torch.exp(-0.5 * (x - mu) ** 2 / ((sigma + sigma_const) ** 2)) / Z
    nll = -torch.log(probability_density + 1e-7)
    return nll





class PolarLoss(nn.Module):
    def __init__(self):
        super(PolarLoss,self).__init__()

    def forward(self,output,mask,ind,target):
        [output_angle,output_wh,output_direction] = output
        [gt_angle,gt_wh,gt_direction] = target


        #1. get the angle of GT or Pred

        pred_angle = _tranpose_and_gather_feat(output_angle, ind)
        mask = mask.unsqueeze(2).expand_as(pred_angle).float()
        pred_angle = pred_angle * 10
        pred_direction = _tranpose_and_gather_feat(output_direction, ind)
        pred_direction = torch.argmax(pred_direction, dim=2)
        pred_direction = pred_direction.unsqueeze_(dim=-1)
        pred_angle_res = torch.zeros_like(pred_angle)
        pred_angle_res[pred_direction == 0] = 90 - pred_angle[pred_direction == 0]
        pred_angle_res[pred_direction == 1] = pred_angle[pred_direction == 1] + 90
        pred_angle_res[pred_direction == 2] = pred_angle[pred_direction == 2]
        pred_angle_res = pred_angle_res * mask


        gt_angle = gt_angle * 10
        gt_direction = _tranpose_and_gather_feat(gt_direction, ind)
        gt_direction = torch.argmax(gt_direction, dim=2)
        gt_direction = gt_direction.unsqueeze_(dim=-1)
        gt_angle_res = torch.zeros_like(gt_angle)
        gt_angle_res[gt_direction == 0] = 90 - gt_angle[gt_direction == 0]
        gt_angle_res[gt_direction == 1] = gt_angle[gt_direction == 1] + 90
        gt_angle_res[gt_direction == 2] = gt_angle[gt_direction == 2]
        gt_angle_res = gt_angle_res * mask

        # 2. get wh of GT and Pred
        output_w, output_h = output_wh[:,0,:,:].unsqueeze(dim=1),output_wh[:,1,:,:].unsqueeze(dim=1)
        gt_w, gt_h = gt_wh[:,:,0].unsqueeze(dim=-1),gt_wh[:,:,1].unsqueeze(dim=-1)
        pred_w = _tranpose_and_gather_feat(output_w,ind) * mask
        pred_h = _tranpose_and_gather_feat(output_h,ind) * mask


        # 3. calculate euclidean distance
        w_ed = (pred_w - gt_w) * (pred_w - gt_w) + (pred_angle_res - gt_angle_res) * (pred_angle_res - gt_angle_res)
        w_loss = torch.log(w_ed + 1)

        h_ed = (pred_h - gt_h) * (pred_h - gt_h) + (pred_angle_res - gt_angle_res) * (pred_angle_res - gt_angle_res)
        h_loss = torch.log(h_ed + 1)

        losses = torch.sum((w_loss + h_loss) / 2) / (mask.sum() + 1e-4)
        return losses
















