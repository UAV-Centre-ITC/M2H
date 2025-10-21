from functools import partial

import numpy as np
import torch
import torch.nn as nn

from m2h_core.models import CenterPadding, FullHead as MLTHead, FullModel as DepthEncoderDecoder

min_depth, max_depth = 0.001, 10
arch_name="vit_small" # vit_base

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

mlt_head = MLTHead(
                    channels=256,
                    embed_dims=backbone.embed_dim,
                    post_process_channels=[backbone.embed_dim // 2 ** (3 - i) for i in range(4)],
                    readout_type="project",
                    min_depth=min_depth,
                    max_depth=max_depth,
                    num_classes = 41,
                    act_layer=nn.GELU
                )

out_index = {
                "vit_small": [2, 5, 8, 11],
                "vit_base": [2, 5, 8, 11],
                "vit_large": [4, 11, 17, 23],
                "vit_giant2": [9, 19, 29, 39],
            }[arch_name]

model = DepthEncoderDecoder(backbone=backbone, mlt_head=mlt_head )
            
model.backbone.forward = partial(
                backbone.get_intermediate_layers,
                n=out_index,
                reshape=True,
                return_class_token=True,
                norm=False,
            )

model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone.patch_size)(x[0]))
##############
device = torch.device("cuda")
            
model = model.to(device)
model.eval()

dummy_input = torch.randn(1, 3,224,224, dtype=torch.float).to(device)

# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 100
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(mean_syn)
torch.cuda.empty_cache()
