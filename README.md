# Comfyui-MoGe2

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes to use [MoGe2](https://github.com/microsoft/MoGe) prediction.

![](./example_workflows/MoGe2.jpg)

Original repo: https://github.com/microsoft/MoGe

Huggingface demo: https://huggingface.co/spaces/Ruicheng/MoGe-2

## Updates

- [2025-09-15] Panorama mode: default model `v2` with normals; default weighted merge for smoother seams; GLB extra rotation option; hole filling; per-view exports; synthetic validation script; fixed segment misalignment (rotation order).
- [2025-07-29] Support `Ruicheng/moge-2-vitl-normal` and `Ruicheng/moge-vitl` model.

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

- Use the `MoGe2Panorama` node to process equirectangular (360°) panoramas with metric-scale preservation and stitched normals.
- Requirements:
  - A local copy of the MoGe checkpoint folder(s). The node resolves a local path from the selected version or uses your `model_path` override. No network download is attempted.
  - Input must be an equirectangular panorama. Output resolution matches the input.
- Model selection:
  - `model` (v1/v2): Choose the MoGe version. Default is `v2` (includes normals).
  - `model_path` (optional): Local override path. If set and exists, it is used instead of the version mapping. If neither exists, the node raises an error.
- Parameters (brief):
  - `face_resolution`: Per-view split resolution (icosahedron faces). Higher = finer coverage; more VRAM/time.
  - `resolution_level`: Internal model token resolution (Low/Medium/High/Ultra). Higher = better, slower.
  - `view_fov_x_deg`: Virtual view FOV; increase (e.g., 100–110) for more overlap and smoother seams.
  - `merge_method`: `weighted` (default) blends overlapping views by angle and optional depth; `z_buffer` picks nearest distance.
  - `angle_power`: Angle weighting exponent for `weighted` merge (weight ~ cos(theta)^p).
  - `depth_alpha`: Optional depth factor for `weighted` merge (weight ~ 1 / distance^alpha).
  - `apply_mask`: Apply model validity mask to ignore unreliable pixels.
  - `horizontal_wrap`: Horizontal wrap at the pano seam (useful for equirectangular edge).
  - `fill_holes`: Fill small holes after merge by averaging neighbors.
  - `hole_iters`: Number of fill iterations.
  - `horizontal_wrap`: Use horizontal wrap border when remapping per-view data to the panorama grid. Keep off unless debugging edge cases.
  - `export_per_view`: Export each per-view prediction for debugging.
  - `per_view_export_format`: `ply`, `glb`, or `both` for per-view exports.
  - `per_view_prefix`: Prefix/path for per-view exports under ComfyUI’s output directory.
  - `output_pcl`: Export merged point cloud as `.ply` with panorama colors.
  - `output_glb`: Export textured mesh as `.glb` built over the panorama grid.
  - `glb_rotate_x_deg`: Extra clockwise rotation around X (red) axis for GLB export.
  - `filename_prefix`: Output prefix under ComfyUI’s output directory.
  - `use_fp16`: Use half precision to reduce VRAM and improve speed.
- Outputs:
  - `depth`: Panorama depth visualization (closer appears brighter in the preview).
  - `normal`: Panorama normal visualization (world-space, stitched from views).
  - `pcl_path`: Saved point cloud path (STRING, `.ply`).
  - `glb_path`: Saved textured mesh path (STRING, `.glb`).

#### Misalignment Fix (Rotation Order)

- Fixed per-segment misalignment due to a rotation-order bug when converting camera-frame outputs to world space.
- Now uses row-vector convention `x_world = x_cam @ R` (where `R` is world→camera rotation) to rotate points/normals into the global panorama frame.
- Seam alignment and metric consistency across views should be preserved.

#### Synthetic Validation

- A synthetic validation script verifies the rotation and seam alignment logic without model weights.
- Run: `python scripts/validate_panorama.py`
- Outputs under `outputs/validate_panorama/`:
  - `diff_heatmap.png`: normalized 3D error heatmap (should be ~0 on a unit sphere test).
  - `mask.png`: merged valid mask.

Example workflow: `example_workflows/MoGe2Panorama.json`


## Model Support

- [x] [Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal/tree/main)
- [x] [Ruicheng/moge-vitl](https://huggingface.co/Ruicheng/moge-vitl/tree/main)

## Acknowledgements

I would like to thank the contributors to the [MoGe](https://github.com/microsoft/MoGe), [ComfyUI-MoGe](https://github.com/kijia), for their open research.
