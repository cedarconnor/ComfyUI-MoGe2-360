# Comfyui-MoGe2

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes to use [MoGe2](https://github.com/microsoft/MoGe) prediction.

![](./example_workflows/MoGe2.jpg)

Original repo: https://github.com/microsoft/MoGe

Huggingface demo: https://huggingface.co/spaces/Ruicheng/MoGe-2

## Updates

- [2025-09-15] Panorama mode: default model `v2` with normals; default z-buffer merge (ray distance) for strongest metric consistency; GLB extra rotation option; hole filling; per-view exports; synthetic validation script; fixed segment misalignment (rotation order).
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
  - `face_resolution`: Per-view split resolution (icosahedron faces). Higher = finer coverage; more VRAM/time. Typical 512–1024; up to 4096 if VRAM allows.
  - `resolution_level`: Internal model token resolution (Low/Medium/High/Ultra). Higher = better, slower.
  - `view_fov_x_deg`: Virtual view FOV (default 110) for more overlap and smoother seams.
- `merge_method`: `z_buffer` (default) picks nearest distance; `weighted` blends overlapping views by angle and optional depth; `affine_depth` aligns and blends depth per slice (scale+bias) in the pano; `poisson_depth` fuses depth via gradient-domain Poisson integration.
- `zbuffer_mode`: `ray` (default) uses distance along the panorama ray; `radial` uses ||P||.
 - `mesh_wrap_x`: Close the panorama seam by connecting x=0 and x=W-1 with duplicated UVs. Prevents gaps and keeps texture seams stable.
 - `export_depth` + `depth_format` + `depth_prefix`: Save fused depth as 16‑bit PNG (mm), EXR float, or both.
- `mask_image` (optional): label/mask IMAGE at panorama resolution. Unique colors (RGB) or intensities denote labels; 0 is background by default.
- `multi_glb_from_mask`: if true (with `mask_image`), exports one GLB per label region.
- `mask_ignore_zero`: ignore label 0 when exporting per-label GLBs.
- `min_label_area_ratio`: minimum fraction of pixels a label must occupy to export (default 0.5%).
- `multi_glb_prefix`: output prefix for per-label GLBs under ComfyUI’s output directory.
  - `angle_power`: Angle weighting exponent for `weighted` merge (weight ~ cos(theta)^p).
  - `depth_alpha`: Optional depth factor for `weighted` merge (weight ~ 1 / distance^alpha).
  - `apply_mask`: Apply model validity mask to ignore unreliable pixels.
  - `horizontal_wrap`: Horizontal wrap at the pano seam (useful for equirectangular edge).
  - `skip_small_masks` + `min_mask_ratio`: ignore views with too few valid pixels.
  - `wrap_consistency`: enforce left/right seam consistency after depth fusion.
  - `polar_smooth` + `polar_cap_ratio` + `polar_blur_ks`: stabilize zenith/nadir regions.
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
- `glb_path`: Saved textured mesh path (STRING, `.glb`). If multiple per-label GLBs are exported, this string contains multiple lines: the main GLB (if enabled) and then one path per label GLB.
 - `depth_file`: Path(s) to exported depth files (one per line if multiple formats were chosen).

### Example Workflow

See `example_workflows/MoGe2Panorama_LabelExport.json` for a simple setup that:
- Loads a panorama and a label mask,
- Runs `MoGe2Panorama` with `multi_glb_from_mask` enabled,
- Saves the depth visualization, and exports per‑label GLBs.

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
