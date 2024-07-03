from .sampler.det_sampler import DetDataset
from .sampler.seg_sampler import SegDataset
#
from .sampler.unify_sampler import UnifyDataset
from .sampler.unify_sampler_1d import UnifyDataset_1D
from .sampler.unify_sampler_1d_region_wo_norm import UnifyDataset_1DRegion_wonorm
from .sampler.unify_sampler_1d_region_cheby import UnifyDataset_Cheby
from .sampler.unify_sampler_1d_region import UnifyDataset_1DRegion
from .sampler.contourpoint_sampler import ContourPointDataset
#
from .sampler.val_sampler import val_sampler
from .sampler.val_seg_sampler import val_seg_sampler
from .sampler.val_unify_sampler import val_unify_sampler
from .data.nv10 import NV10
from .data.nv10seg import NV10seg
from .data.nv10obb import NV10obb
from .data.isaid_bak import isaid
from .data.isaid import iSAID_c
from .data.dota import DOTA
from .data.dota_unfiy import DOTA_unify
from .data.dota_ablation import DOTA_ablation
from .data.dior import DIOR
from .data.dior_hbb import DIOR_HBB
from .data.hrsid import HRSID
from .data.nv10unify import NV10unify
import torch

dataset_factory = {
    'hrsid': HRSID,
    'nv10': NV10,
    'nv10unify': NV10unify,
    'nv10seg': NV10seg,
    'nv10obb': NV10obb,
    'isaid': isaid,
    'dota_unify': DOTA_unify,
    'dota_ablation': DOTA_ablation,
    'dior': DIOR,
    'isaidc': iSAID_c,
    'dior_hbb': DIOR_HBB,
    'dota_obb': DOTA,
}


_sample_factory = {
    'det': DetDataset,
    'val_det': val_sampler,
    'seg': SegDataset,
    'val_seg': val_seg_sampler,
    'unify': UnifyDataset,
    'unify1D': UnifyDataset_1DRegion,
    'cheby': UnifyDataset_Cheby,
    'unify1Dwonorm': UnifyDataset_1DRegion_wonorm,
    'val_unify': val_unify_sampler,
    'val_unify1D': val_unify_sampler,
    'val_unify1Dwonorm': val_unify_sampler,
    'val_cheby': val_unify_sampler,
    'ContourPoint': ContourPointDataset,
    'val_ContourPoint': val_unify_sampler
}


def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass
    return Dataset


def make_train_loader(cfg):
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    shuffle = True
    drop_last = True
    dataset_name = cfg.data.dataset
    task_name = cfg.common.task
    Dataset = get_dataset(dataset_name, task_name)

    data_loader = torch.utils.data.DataLoader(
        Dataset(cfg, 'train'),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )
    return data_loader


def make_val_loader(cfg):
    batch_size = 1
    num_workers = cfg.train.num_workers
    shuffle = False
    drop_last = False
    dataset_name = cfg.data.dataset
    task_name = 'val_' + cfg.common.task
    Dataset = get_dataset(dataset_name, task_name)

    data_loader = torch.utils.data.DataLoader(
        Dataset(cfg, 'val'),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )
    return data_loader


def make_data_loader(is_train=True, cfg=None):
    if is_train:
        return make_train_loader(cfg), make_val_loader(cfg)
    else:
        return make_val_loader(cfg)
    