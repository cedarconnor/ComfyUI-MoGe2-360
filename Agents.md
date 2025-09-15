# Agent Spec: Modify zade23 ComfyUI MoGe2 Node to Support Panorama Mode with Correct Metric Scale

## Overview

The goal is to extend the zade23 ComfyUI node for MoGe2 so that it can process **equirectangular / spherical panoramic images** (360°), using the infer_panorama workflow, **and** preserve the metric scale output that MoGe2 is capable of when processing single-perspective images. Current infer_panorama implementation often loses metric scale in the merged output. This spec describes features of infer_panorama.py, identifies where scale loss likely occurs, and outlines exactly what to modify / add in the ComfyUI node implementation to fix these issues.

---

Update 2025-09-15: Implemented the panorama node with default model v2 (normals), default weighted merge for smoother seams, GLB orientation control, simple hole filling, optional per-view exports (PLY/GLB), and a synthetic validation script. Also fixed a rotation-order bug that caused per-segment misalignment.

## Background & Key Features of MoGe-2 & infer_panorama.py

Taken from MoGe-2 paper, README, and code:

- MoGe-2 outputs **point map** (H × W × 3), **depth map**, **normal map** (if using the “-normal” model), **mask**, and **intrinsics**. For MoGe-2, the point map is in **metric scale** (i.e. real-world units) when using a single perspective input. :contentReference[oaicite:0]{index=0}  
- The `infer_panorama` script is an experimental extension. It:  
  1. Splits an equirectangular panorama into multiple perspective views. :contentReference[oaicite:1]{index=1}  
  2. Runs MoGe (or MoGe2) inference on each view separately. :contentReference[oaicite:2]{index=2}  
  3. Combines / merges the output maps (depth, point, mask, possibly normals) back into a panorama depth map / point map. :contentReference[oaicite:3]{index=3}  
- According to the MoGe-2 CHANGELOG, panorama inference scripts were added. :contentReference[oaicite:4]{index=4}  

---

## Known Issues / Scale Loss

From GitHub issues and user reports:

- After merging panoramic views, the **metric scale** that is present in single-view outputs often seems lost; the merged panorama behaves similarly (in scale) to MoGe-1 which is affine / relative only. (Issue #97) :contentReference[oaicite:5]{index=5}  
- Some views in the panorama splits may have *no valid mask pixels*, causing parts of focal / shift recovery or reprojection to either fail or produce degenerate / default values. This could lead to mis-estimation in merging or normalization steps. :contentReference[oaicite:6]{index=6}  

---

## Modification Plan: What to Change in zade23 ComfyUI Node Implementation

Here are steps / features / patches the agent should implement to ensure correct panorama processing with metric scale preserved.

| Component | Proposed Changes / Additions | Rationale / Ensuring Metric Scale |
|---|---|---|
| **Panorama Handling Node (Splitter)** | • Add a node that accepts an equirectangular (360°) image <br> • Split into perspective views (e.g. cube faces or more view patches) <br> • For each view compute and output its **intrinsic matrix** (focal length in pixels, principal point) and **extrinsic pose** (rotation relative to pano center) <br> • Use field of view consistent with how splits are done (e.g. cube faces → 90° FOV) | Having correct intrinsics & extrinsics is essential so that later conversion of depth → 3D, and merging into global/world coordinates preserve correct units and geometry |
| **Inference per View** | • Use MoGeModel.infer (MoGe-2) for each view, retrieving: metric point map, depth, mask, normals (if available), and intrinsics (if model reports) <br> • Ensure that FOV/intrinsics passed to the inference correspond to the virtual camera used for that view <br> • Handle views with empty valid mask: skip or mark as invalid so they don’t corrupt merging metrics | Ensures that each view's output is in metric units and aligned relative to its camera, avoiding scale drift or fallback to non-metric behavior when invalid inputs present |
| **Transform to Global / World Coordinates** | • For each view’s point map, convert depth & pixel coordinates into local 3D camera frame:  
> \[x = (u − cx)/fx * depth, y = (v − cy)/fy * depth, z = depth\] <br> • Apply extrinsic (rotation) to bring into a common coordinate system (e.g. placing view cameras around the pano center) <br> • Also rotate normals appropriately if provided | This ensures that metric point positions are consistent across views, with no per-view relative scaling differences, so merging uses true metric points |
| **Merging / Stitching Logic** | • Create a merge node that takes all transformed point maps (and optionally normals) and generates a pano output: depth map, point map, mask <br> • For each pixel in the panorama (in equirectangular), compute which perspective view(s) cover it, and select / blend the contribution(s) based on mask and depth (e.g. closest depth wins, or weighted average) <br> • In overlapping view regions, ensure merging logic does not rescale or normalize depth or points but chooses the value that preserves metric relationships <br> • For pixels with no valid view, mark them invalid in mask or interpolate carefully if required | Without this, merging might collapse scale or perform averaging that reduces metric fidelity; choosing consistent metric distances ensures global consistency |
| **Scale Preservation Fixes** | • Remove or modify any normalization / shift / scale adjustments in merge pipeline; ensure that global scale factor (if predicted by model) is applied uniformly across all views' point maps before merging <br> • Ensure that if the model uses decoupled scale prediction (MoGe-2: via CLS-token MLP), that predicted scale is used to scale the affine-invariant point maps produced per view <br> • In code where focal-shift or shift recovery is done, guard against missing / empty mask; propagate correct intrinsics so that those corrections do not break metric properties <br> • Avoid resizing of inputs in a way that changes pixel-to-metric conversion unless also adjusting intrinsics accordingly | These are exactly where metric scale tends to be lost: unintended normalization or mismatched conversions; using proper focal, using global scale, using depth correctly will preserve metric units |
| **UI / Node API Changes** | • Add node parameters/options:  
>   – Panorama mode toggle <br>   – Split strategy (cube faces, patch grid, view count) <br>   – Provision for user-supplied pano intrinsics (if any) <br>   – Face resolution <br>   – Merge method (pick, average, weighted) <br> • Display / export final outputs: pano point map (metric), pano depth map, full pano point cloud (with metric units) <br> • Visualization of normals and masking to assess invalid / boundary regions | Users need to control splitting / merging, inspect results, ensure correctness; making parameters visible helps debugging of scale issues |

---

## Suggested File / Code Path Patches (Locations)

Here are likely places in the zade23 node / MoGe2 code that must be modified or patched:

- In the implementation of `infer_panorama.py` (or wherever panorama splitting & merging is done), inspect how **intrinsics** are computed for each view; patch if needed to compute correct focal length (in pixels) based on face dimensions.  
- In the merge code that reprojects point maps into pano, locate normalization / resizing steps; ensure that any resizing of images / depths is matched by correct scaling of intrinsics / depth values.  
- In the node wrapper in zade23, which may assume only single image inference, add logic for batching / looping over view patches + storing per-view intrinsics & extrinsics.  
- In post-processing or export stages, ensure that exported point cloud uses the global transformed points (world frame) rather than view-local; and that output format includes units (if applicable).  

---

## Test Cases & Validation

To ensure that modifications work correctly, include unit / integration test cases:

1. **Synthetic panorama** with known geometry: e.g. render a cube room or sphere with known dimensions, generate an equirectangular image. Run the panorama node, export point cloud, measure distances between known points; check that distances correspond to real units as expected.

2. **Single view vs panorama consistency**: Pick a region of the pano corresponding to a single perspective view; compare the MoGe2 single-view output vs the panorama merged output in that region: depth & point maps should match (within numerical tolerance) and both be metric.

3. **Scale prediction consistency**: Use model versions with scale prediction; check that the predicted scale (global) is identical (or nearly) across views when merged, and matches that from single-view inference.

4. **Edge / overlap seam smoothness**: Check in overlapping edges between view patches that depth / point transitions are smooth, no discontinuities or “jumps” due to inconsistent depth scale or orientation.

5. **Handling invalid / masked views**: Create panoramas where some faces have blank or low signal (e.g. uniform sky). Ensure that code handles mask = False well (skips, does not compute focal shift incorrectly, etc.), and that merging does not collapse scale because of missing data.

---

## Sample agents.md Tasks / Steps

Here’s a task breakdown that Codex / agent might perform in order:

1. Read and understand existing zade23 ComfyUI node code: its inference flow, what inputs/outputs it uses, how it handles single image vs any panorama support (if any).

2. Read `moge/scripts/infer_panorama.py` from MoGe-2, inspect its splitting / merging logic and where Intrinsics / extrinsic / depth → 3D → panorama reprojection happens. Identify code paths where normalization or default shifts may remove metric scaling.

3. Create a new node or extend existing node in zade23:

   - New “PanoSplitter” node: input pano image → list of view images + intrinsics + extrinsics
   - Modify inference node to accept view + intrinsics + extrinsics and produce metric point maps etc.

4. Create “PanoMerger” node: input per-view outputs + metadata → produce pano depth / point map / normals etc, respecting metric units, merging appropriately.

5. Modify API / node UI to expose parameters (split strategy, resolution, merge behaviour, user FOV / intrinsics etc.)

6. Write test scripts / sample workflows, covering single view vs pano, synthetic scenes, scale checks

7. Documentation: Update README / node doc to explain panorama mode, when metric scale is preserved, assumptions (intrinsics, view splits etc.)

---

### References

- MoGe-2 paper: Metric scale geometry estimation, decoupled scale + relative geometry. :contentReference[oaicite:7]{index=7}  
- MoGe-2 CHANGELOG: Inference scripts for panorama images, user-supplied FOV. :contentReference[oaicite:8]{index=8}  
- MoGe README: Definition of maps including point map metric units for MoGe-2. :contentReference[oaicite:9]{index=9}  

---

## Summary

- The core issue is that infer_panorama’s merge / reprojection logic does not always maintain the metric scale from MoGe-2’s single view outputs.  
- To fix: maintain correct intrinsics & extrinsics per view; convert depths into proper 3D points; apply model’s predicted global scale; reproject to global/world frame; merge without normalization or scale collapsing; handle edge / mask cases gracefully.  
- Implement as a set of nodes in zade23 ComfyUI: splitting, inference, transform, merge, export.  
- Add UI / parameters to control split strategy, face resolution, merging method.  
- Validate with synthetic & real test cases to confirm metric accuracy.

---

## Implementation Notes (Planned in this repo)

- Single Node MVP: Implement a new `MoGe2Panorama` node that takes an equirectangular panorama and outputs panorama depth (visualized), panorama normals (visualized), and optional exports for point cloud (PLY) and textured mesh (GLB).
- Split Strategy: Use existing icosahedron-based virtual cameras from `moge/utils/panorama.py#get_panorama_cameras` with 90° FOV and zero translation (all virtual cameras at panorama center). Output panorama resolution matches the input panorama resolution.
- Per-View Inference: For each split view, run `MoGe-2` (v2) `infer` with `fov_x=90`, `force_projection=True`, and `apply_mask=True`. The node supports a `model` dropdown (v1/v2) with a local default mapping and an optional `model_path` override; it loads weights locally only (no network). If the resolved path is missing, it raises a clear error. Views with empty valid masks are skipped.
- Metric-Preserving Merge (Z-Buffer):
  - For each panorama pixel, compute the ray direction in world coordinates via `spherical_uv_to_directions`.
  - For each view, project that direction to the view image to sample the predicted per-view point map (in camera frame) and mask.
  - Rotate sampled camera-space points and normals into world-space using the view rotation (from extrinsics), then compute radial distance `t = ||P_world||`.
  - Across views, pick the valid contribution with the smallest `t` (z-buffer) per panorama pixel. Do not apply log-distance blending, affine normalization, or gradient Poisson solves in this path to preserve metric scale.
- Normals: Merge normals by selecting the same winning view per pixel, after rotating normals to world space. Visualize with the existing `colorize_normal` utility.
- Exports: Optionally export a world-space point cloud (PLY) and a textured mesh (GLB) built over the panorama grid. Exports use ComfyUI's `folder_paths` with a configurable `filename_prefix`.
- Parameters:
  - `model` (enum: v1/v2, default v2): select MoGe version (panorama prefers v2 for metric scale and normals).
  - `model_path` (string, optional, local): local checkpoint path that overrides the version mapping when present.
  - `face_resolution` (int, default 512): resolution of each virtual view (icosahedron faces).
  - `resolution_level` (enum): forwarded to the model to control token count.
  - `merge_method` (enum): default `z_buffer` (metric preserving). Optionally expose `poisson` later.
  - `apply_mask` (bool): apply model’s validity mask.
  - `view_fov_x_deg` (int): per-view FOV to control overlap (smoother seams with more overlap).
  - `merge_method` (enum): `weighted` (default) or `z_buffer`.
  - `angle_power`, `depth_alpha`: weights for `weighted` blend.
  - `fill_holes` (bool) and `hole_iters` (int): simple neighbor-average hole filling.
  - `horizontal_wrap` (bool): use horizontal wrap border in pano remap for edge-case debugging.
  - `export_per_view` (bool): export each per-view prediction for debugging.
  - `per_view_export_format` (enum): `ply`, `glb`, or `both` for per-view exports.
  - `per_view_prefix` (string): prefix/path for per-view exports.
  - `output_pcl` (bool): whether to write out a PLY point cloud.
  - `output_glb` (bool): whether to write out a textured GLB mesh.
  - `glb_rotate_x_deg` (int): extra rotation around X axis for GLB viewers.
  - `filename_prefix` (string): export prefix/path.
  - `use_fp16` (bool): enable FP16 inference for speed/VRAM.
- Outputs: panorama depth visualization (IMAGE), panorama normal visualization (IMAGE), point cloud path (STRING, .ply), and GLB path (STRING, .glb).
- Future Work: Add alternative split strategies (cube faces), expose pano intrinsics, and add advanced seam blending that respects metric scale (e.g., confidence-weighted z-buffer) if needed.

---

## Implemented Fix: Segment Misalignment (Rotation Order)

- utils3d extrinsics follow world→camera: `x_cam = R x_world + t`.
- Converting camera-space predictions to world for zero-translation virtual cameras requires: `x_world = R^T x_cam`.
- With row-vector math used in this repo, this becomes `x_world_row = x_cam_row @ R`.
- Previous code used `@ R.T`, causing per-view outputs to be rotated incorrectly and misaligned across seams.
- nodes.py updated to use `@ R` for points and normals in the pano z-buffer merge and per-view exports.

---

## Validation

- Added `scripts/validate_panorama.py` which simulates a unit sphere panorama, generates per-view camera-space predictions, merges using the same z-buffer world-frame logic, and reports mean absolute 3D error.
- Expected: near-zero error and clean seams, confirming rotation handling and projection are consistent.

Run:
- `python scripts/validate_panorama.py`
- Outputs under `outputs/validate_panorama/`: `diff_heatmap.png`, `mask.png`.

---
