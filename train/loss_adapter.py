from .loss.contourdet_loss import ContourDet_loss
from .loss.contourdet_loss_cheby import ContourDet_loss_cheby
from .loss.contourseg_loss import ContourSeg_loss
from .loss.contourunify_loss import ContourUnify_loss
from .loss.contourdet_loss_wonorm import ContourDet_loss_wonorm
from .loss.contourdet_loss_polarmaskloss import ContourDet_polariouloss
from .loss.contourunify_loss_supfd import ContourUnify_loss_sup_fd
from .loss.contourdet_loss_Region import ContourDet_loss_region
from .loss.contourdet_loss_hmweighted import ContourDet_loss_weighted
from .loss.contourpoint_loss import ContourPoint_loss

_losser_factory = {
    'contourpoint_loss': ContourPoint_loss,
    'contourdet_loss': ContourDet_loss,
    'contourdet_loss_cheby': ContourDet_loss_cheby,
    'contourdet_loss_wonorm': ContourDet_loss_wonorm,
    'contourseg_loss': ContourSeg_loss,
    'contourunify_loss': ContourUnify_loss,
    'contourdet_loss_polarmask': ContourDet_polariouloss,
    'contourunfiy_loss_supfd': ContourUnify_loss_sup_fd,
    'contourdet_loss_region': ContourDet_loss_region,
    'contourdet_loss_hmweighted': ContourDet_loss_weighted
}

def make_losser(cfg):
    get_losser = _losser_factory[cfg.model.loss.__name__]
    losser = get_losser(cfg)
    return losser