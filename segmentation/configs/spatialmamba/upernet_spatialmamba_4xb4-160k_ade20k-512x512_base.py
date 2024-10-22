_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        type='MM_SpatialMamba',
        out_indices=(0, 1, 2, 3),
        pretrained="",
        dims=96,
        d_state=1,
        depths=(2, 4, 21, 5),
        drop_path_rate=0.5,
    ),
)


