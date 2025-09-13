# Comfyui-MoGe2

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes to use [MoGe2](https://github.com/microsoft/MoGe) prediction.

![](./example_workflows/MoGe2.jpg)

Original repo: https://github.com/microsoft/MoGe

Huggingface demo: https://huggingface.co/spaces/Ruicheng/MoGe-2

## Updates

- [2025-07-29]  Support `Ruicheng/moge-2-vitl-normal` and `Ruicheng/moge-vitl` model.

## Features

|version|model|3D|depth_map|normal_map|
|---|---|---|---|---|
|v1|[Ruicheng/moge-vitl](https://huggingface.co/Ruicheng/moge-vitl/tree/main)|✅|✅|❌|
|v2|[Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal/tree/main)|✅|✅|✅|

> Using `v1` model to export `normal` will return black image instead of normal map. `Ruicheng/moge-vitl` does not support normal map.

## How to Use

### ComfyUI-Manager

Run ComfyUI → `Manager` → `Custom Nodes Manager` → search and install `Comfyui-MoGe2`

### Git Clone

1. Clone this repo to `ComfyUI/custom_nodes` 
2. Install requirements: `pip install -r requirements.txt`

### Panorama Mode (Metric)

- Use the new `MoGe2Panorama` node to process equirectangular (360°) panoramas with metric-scale preservation.
- Requirements:
  - A local copy of the MoGe-2 v2 checkpoint folder. Set `model_path` to that local path (e.g., `C:/models/Ruicheng/moge-2-vitl-normal`). No network download is attempted.
  - Input must be an equirectangular panorama. Output resolution matches the input.
- Parameters:
  - `model_path`: local path to the model folder.
  - `face_resolution`: icosahedron per-view resolution (e.g., 512).
  - `resolution_level`: model resolution/tokens (Low/Medium/High/Ultra).
  - `merge_method`: z_buffer (metric-preserving) merge.
  - `output_pcl`: exports merged point cloud as `.ply` with input colors.
  - `filename_prefix`: output path/prefix for exports.
- Outputs:
  - `depth`: panorama depth visualization (closer is brighter in the preview).
  - `normal`: panorama normal visualization (merged and rotated to world).
  - `pcl_path`: saved point cloud path (string).

Example workflow: `example_workflows/MoGe2Panorama.json`


## Model Support

- [x] [Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal/tree/main)
- [x] [Ruicheng/moge-vitl](https://huggingface.co/Ruicheng/moge-vitl/tree/main)

## Acknowledgements

I would like to thank the contributors to the [MoGe](https://github.com/microsoft/MoGe), [ComfyUI-MoGe](https://github.com/kijia), for their open research.
