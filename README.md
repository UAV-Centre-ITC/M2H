# Multi-Mono-Hydra (M2H)

**Multi-Task Learning with Efficient Window-Based Cross-Task Attention for Monocular Spatial Perception**  
📄 [Read the paper on arXiv (2510.17363)](https://arxiv.org/pdf/2510.17363)

**Authors:** U.V.B.L. Udugama, George Vosselman, and Francesco Nex

This repository hosts the ROS integration of **M2H**, a unified perception framework that performs monocular spatial understanding through joint prediction of semantic segmentation, depth, surface normals, and edge maps from a single RGB image. M2H combines an efficient transformer-based architecture with multi-task learning to improve geometric and semantic reasoning while maintaining real-time performance.

## Features
- RGB-only inference for dense depth and semantic segmentation.
- Configurable output topics and inference cadence.
- Launch files for quick integration with existing camera drivers.
- Window-based cross-task attention and gated feature merging for efficient multi-task feature sharing.

## Main Contributions
- **Unified multi-task framework** that jointly predicts semantics, depth, edges, and surface normals from one monocular image.
- **Window-Based Cross-Task Attention (WMCA)** exchanges information between tasks within localized windows to capture inter-task dependencies without heavy computation.
- **Global Gated Feature Merging (GGFM)** injects global context via a gating strategy, improving scene consistency across tasks.
- **Cross-task consistency learning** (depth–normal and edge–semantic losses) encourages geometric and semantic alignment.
- **Performance and efficiency:** achieves ≈30 FPS on an RTX 3080 while delivering state-of-the-art results on NYUDv2, Hypersim, and Cityscapes.

## Highlights
## Architecture
![M2H Overall Model](images/M2H_TC.png)

M2H Overall Model

The M2H architecture is built around a shared Vision Transformer (ViT) encoder based on DINOv2, followed by a set of lightweight, task-specific decoders for semantic segmentation, depth, surface normals, and edges. A Window-Based Cross-Task Attention (WMCA) module enables efficient feature interaction across tasks within localized spatial windows, while the Global Gated Feature Merging (GGFM) block integrates global context through channel-wise gating. Together, these components allow M2H to learn both local geometric cues and global semantic relationships in a unified manner. The model efficiently balances shared representation learning and task specialization, resulting in robust and coherent multi-modal scene understanding from a single RGB input.

- Builds on the DINOv2 ViT encoder as a shared backbone for all tasks.
- Employs multi-scale token reassembly to recover spatial structure from transformer features.
- Uses Dynamic Weight Averaging (DWA) to balance multi-task training adaptively.
- Demonstrates consistent improvements over both single-task and prior multi-task baselines.

## Results Summary
**NYUDv2 (validation) — multi-task comparison**

| Method | Semseg mIoU ↑ | Depth RMSE ↓ | Normals mean ↓ | Boundary odsF ↑ |
| --- | --- | --- | --- | --- |
| MTMamba++ [11] | 57.01 | 0.4818 | 18.27 | 79.40 |
| SwinMTL [12] | 58.14 | 0.5179 | — | — |
| M2H-small (ours) | 58.05 | 0.4396 | 14.04 | 74.44 |
| **M2H (ours)** | **61.54** | **0.4196** | **13.81** | **85.27** |

**Parameters & GFLOPs (NYUDv2)**

| Method | #Params | GFLOPs |
| --- | --- | --- |
| TaskPrompter [29] | 373 M | 416 |
| SwinMTL [12] | 87.38 M | 65 |
| MTMamba++ [11] | 315 M | 524 |
| M2H-small (ours) | 33.7 M | 59 |
| **M2H (ours)** | **81 M** | **488** |

**3D Mapping Test (ITC dataset) with Mono-Hydra framework**

| Model | 2nd Floor ME (m) ↓ | 2nd Floor SD (m) ↓ | 3rd Floor ME (m) ↓ | 3rd Floor SD (m) ↓ | FPS ↑ |
| --- | --- | --- | --- | --- | --- |
| DistDepth[36]+HRNet[37] | 0.19 | 0.18 | 0.21 | 0.16 | 15 |
| MTMamba++ [11] | 0.21 | 0.22 | 0.18 | 0.19 | 18 |
| M2H-small (ours) | 0.16 | 0.18 | 0.15 | 0.17 | 42 |
| **M2H (ours)** | **0.11** | **0.14** | **0.10** | **0.13** | **30** |


M2H produces accurate and temporally stable predictions suitable for real-time robotic perception and mapping pipelines.

## Features
- RGB-only inference for dense depth and semantic segmentation.
- Configurable output topics and inference cadence.
- Launch files for quick integration with existing camera drivers.

## Requirements
- ROS Noetic (or later) with a configured catkin workspace.
- Python 3.8+ with `torch`, `torchvision`, and `mmcv`.
- CUDA-capable GPU recommended; CPU mode supported with reduced throughput.

## Setup
1. Clone into your catkin workspace `src/` directory.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt  # TODO: create file
   ```
3. Retrieve model weights (instructions pending).
4. Build the workspace:
   ```bash
   catkin build
   source devel/setup.bash
   ```

## Quickstart
```bash
roslaunch m2h m2h.launch image_topic:=/camera/image_raw
```

The node subscribes to the RGB stream and publishes semantically-colored segmentation maps and depth images. Remap the topics in the launch file or via command-line arguments to match your setup.

## Parameters
| Name | Default | Description |
| --- | --- | --- |
| `image_topic` | `/camera/image_raw` | Source RGB topic. |
| `image_depth_topic` | `/camera/image_depth` | Depth prediction output topic. |
| `image_semantic_topic` | `/camera/image_segmented` | Semantic prediction output topic. |
| `model_path` | `$(find m2h)/scripts/checkpoints/val_model_epoch_14_rmse_0.5422_miou_0.6969.pt` | Path to the checkpoint weights. |
| `feed_width` / `feed_height` | `224` | Network input resolution prior to inference. |
| `skip_frequency` | `7` | Process every `n`th frame (set to `1` to use every frame). |
| `arch_name` | `vit_small` | DINOv2 backbone variant (`vit_small`, `vit_base`, `vit_large`, `vit_giant2`). |
| `min_depth` / `max_depth` | `0.001` / `10.0` | Depth head output bounds. |

## Repository Layout
- `launch/`: ROS launch files.
- `scripts/`: Runtime nodes and model definitions (`m2h.py`, lightweight `m2h_perf.py`).
- `scripts/tools/`: Optional helper utilities for plane fitting, point cloud export, and debugging.
- `config/`: YAML configuration for fusion, camera, and label mappings.
- `data/`: Color maps and small auxiliary resources.

## Roadmap
- Add automated weight download tooling.
- Provide example bag / RViz configuration.
- Document CPU-only deployment tips.


## Citation
If you use this repository or the M2H model in your research, please cite:

```bibtex
@article{Udugama2025M2H,
  title={Multi-Task Learning with Efficient Window-Based Cross-Task Attention for Monocular Spatial Perception},
  author={Udugama, U.V.B.L. and Vosselman, George and Nex, Francesco},
  journal={arXiv preprint arXiv:2510.17363},
  year={2025}
}
```
