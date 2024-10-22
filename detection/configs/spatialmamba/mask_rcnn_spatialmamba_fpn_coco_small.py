_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        type='MM_SpatialMamba',
        out_indices=(0, 1, 2, 3),
        pretrained="",
        dims=64,
        d_state=1,
        depths=(2, 4, 21, 5),
        drop_path_rate=0.3,
    ),
    neck=dict(in_channels=[64, 128, 256, 512]),
)
