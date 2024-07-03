from .trainer.contourdet import DetTrainer
from .trainer.contourunify import UnifyTrainer
from .trainer.contourseg import SegTrainer
from .trainer.contourunify_distll import UnifyTrainer_Distill
from .trainer.contourpoint import PointTrainer

_trainer_factory = {
    'contourdet': DetTrainer,
    'contourseg': SegTrainer,
    'contourunify': UnifyTrainer,
    'contourunify_distll': UnifyTrainer_Distill,
    'contourpoint': PointTrainer

}


def make_trainer(cfg, model, loss):
    get_trainer = _trainer_factory[cfg.train.trainer]
    trainer = get_trainer(cfg, model, loss)
    return trainer