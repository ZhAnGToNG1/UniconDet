from .evaluator.eval_nv10 import eval_nv10
from .evaluator.eval_isaid import eval_isaid
from .evaluator.eval_unify import eval_unify
from .evaluator.eval_dota import eval_dota

_evaler_factory = {
    'nv10': eval_unify,
    'nv10seg': eval_unify,
    'nv10obb': eval_unify,
    'nv10unify': eval_unify,
    'isaid': eval_isaid,
    'dota_obb': eval_unify,
    'dior': eval_unify,
    'dior_hbb':eval_unify,
    'dota_unify': eval_unify,
    'dota_ablation': eval_unify,
    'hrsid': eval_unify,
    'isaidc': eval_unify
}


def make_evaler(cfg, dataloader):
    get_evaler = _evaler_factory[cfg.data.dataset]
    evaler = get_evaler(cfg, dataloader)
    return evaler