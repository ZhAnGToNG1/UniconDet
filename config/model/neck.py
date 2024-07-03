class deconv(object):
    num_layers = 3
    dim = 256
    backbone_output_dim = 2048


class spatialtrans(object):
    dim = 256

class fpn(object):
    in_channels = [256, 512, 1024, 2048]
    out_channels = 256
    num_outs = 4

class augview(object):
    stn_view_number = 1
    in_channels = [256, 512, 1024, 2048]
    out_channels = 256
    num_outs = 4


class fpn_fuse(object):
    in_channels = [256, 512, 1024, 2048]
    out_channels = 256
    num_outs = 4
