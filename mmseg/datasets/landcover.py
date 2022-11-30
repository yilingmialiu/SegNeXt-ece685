# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class LandcoverDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background','building', 'woodland','water', 'road')
    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]]
    #PALETTE = [[0], [75], [92], [180], [255]]

    def __init__(self, split, **kwargs):
        super(LandcoverDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='_m.png', 
            split=split,reduce_zero_label=False,
            #classes=('background','building', 'woodland','water', 'road'),
            #palette=[[0, 0, 0], [75, 75, 75], [92, 92, 92], [180, 180, 180], [255, 255, 255]],
            #palette=[[0], [75], [92], [180], [255]],
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
