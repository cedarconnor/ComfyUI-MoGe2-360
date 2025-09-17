import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import logging
from .moge.model import import_model_class_by_version
import cv2
import utils3d
from .moge.utils.vis import colorize_depth, colorize_normal
from .moge.utils.panorama import (
    get_panorama_cameras,
    split_panorama_image,
    spherical_uv_to_directions,
)
import torch
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from typing import Dict, Tuple, List, Optional, Union
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))

log = logging.getLogger(__name__)

class RunMoGe2Process:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["v1","v2"], {"default": "v2"}),
                "image": ("IMAGE",),
                "max_size": ("INT", {"default": 800, "min": 100, "max": 1000, "step": 100}),
                "resolution_level": (["Low", "Medium", "High", "Ultra"], {"default": "High"}),
                "remove_edge": ("BOOLEAN", {"default": True}),
                "apply_mask": ("BOOLEAN", {"default": True}),
                "output_glb": ("BOOLEAN", {"default": True}),  # 新增的开关按钮
                "filename_prefix": ("STRING", {"default": "3D/MoGe"}),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "IMAGE","STRING")
    RETURN_NAMES = ("depth", "normal", "glb_path")
    FUNCTION = "process"
    CATEGORY = "MoGe2"
    OUTPUT_NODE = True
    DESCRIPTION = "Runs the MoGe2 model on the input image. \n v1: Ruicheng/moge-vitl \n v2: Ruicheng/moge-2-vitl-normal"
    
    def process(self, model: str, image, max_size: int, resolution_level: str, remove_edge: bool, apply_mask: bool, output_glb: bool, filename_prefix: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
        
        model_version = model
        
        if model_version == "v1":
            pretrained_model_name_or_path = "Ruicheng/moge-vitl"
        elif model_version == "v2":
            pretrained_model_name_or_path = "Ruicheng/moge-2-vitl-normal"
        
        model_instance = import_model_class_by_version(model_version).from_pretrained(pretrained_model_name_or_path).cuda().eval()
        
        # Convert ComfyUI tensor to numpy array if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            
        # Remove batch dimension if present
        if len(image.shape) == 4:
            image = image[0]
        
        # Ensure image is in the range [0, 255] and convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        larger_size = max(image.shape[:2])
        if larger_size > max_size:
            scale = max_size / larger_size
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        height, width = image.shape[:2]
        resolution_level_int = {'Low': 0, 'Medium': 5, 'High': 9, 'Ultra': 30}.get(resolution_level, 9)
        
        # Convert image to tensor and format it properly for the model
        use_fp16 = True
        image_tensor = torch.tensor(image, dtype=torch.float32 if not use_fp16 else torch.float16, device=torch.device('cuda')).permute(2, 0, 1) / 255
        
        output = model_instance.infer(image_tensor, apply_mask=apply_mask, resolution_level=resolution_level_int, use_fp16=use_fp16)
        output = {k: v.cpu().numpy() for k, v in output.items()}
        
        points = output['points']
        depth = output['depth']
        mask = output['mask']
        normal = output.get('normal', None)
        
        # mask
        if remove_edge:
            mask_cleaned = mask & ~utils3d.numpy.depth_edge(depth, rtol=0.04)
        else:
            mask_cleaned = mask
        
        # normal visualization
        if normal is not None:
            normal_vis = colorize_normal(normal)
        else:
            normal_vis = np.zeros_like(image)
        
        # depth visualization
        depth_for_vis = depth.copy()
        
        masked_depth = depth_for_vis[mask]
        
        if masked_depth.size == 0:
            # If nothing is detected, create a black image
            depth_normalized = np.zeros_like(depth_for_vis)
        else:
            # Normalize the depth values in the masked region to the [0, 1] range
            min_val = masked_depth.min()
            max_val = masked_depth.max()
            
            # Avoid division by zero if depth is constant (e.g., a flat plane)
            if max_val > min_val:
                depth_normalized = (depth_for_vis - min_val) / (max_val - min_val)
            else:
                depth_normalized = np.ones_like(depth_for_vis) * 0.5 # Mid-gray for flat depth

        # Invert the depth map: closer objects become brighter
        depth_inverted = 1.0 - depth_normalized
        depth_inverted[~mask] = 0
        depth_gray_uint8 = (depth_inverted * 255).astype(np.uint8)

        # Convert the single-channel grayscale image to a 3-channel RGB image for ComfyUI compatibility
        depth_gray_rgb = cv2.cvtColor(depth_gray_uint8, cv2.COLOR_GRAY2RGB)

        # Convert numpy array to ComfyUI tensor
        def numpy_to_tensor(img_np):
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8)
            img_np = img_np.astype(np.float32) / 255.0
            if len(img_np.shape) == 3:
                img_np = img_np[None, ...] # Add batch dimension
            return torch.from_numpy(img_np)

        # Convert final visualization to tensor
        depth_tensor = numpy_to_tensor(depth_gray_rgb)
        normal_vis_tensor = numpy_to_tensor(normal_vis)
        
        # mesh
        if normal is None:
            faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                points,
                image.astype(np.float32) / 255,
                utils3d.numpy.image_uv(width=width, height=height),
                mask=mask_cleaned,
                tri=True
            )
            vertex_normals = None
        else:
            faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
                points,
                image.astype(np.float32) / 255,
                utils3d.numpy.image_uv(width=width, height=height),
                normal,
                mask=mask_cleaned,
                tri=True
            )
        vertices = vertices * np.array([1, -1, -1], dtype=np.float32) 
        vertex_uvs = vertex_uvs * np.array([1, -1], dtype=np.float32) + np.array([0, 1], dtype=np.float32)
        if vertex_normals is not None:
            vertex_normals = vertex_normals * np.array([1, -1, -1], dtype=np.float32)
        
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        relative_path = "" # Initialize to empty string
        
        if output_glb:
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_normals=vertex_normals,
                visual = trimesh.visual.texture.TextureVisuals(
                    uv=vertex_uvs,
                    material=trimesh.visual.material.PBRMaterial(
                        baseColorTexture=Image.fromarray(image),
                        metallicFactor=0.5,
                        roughnessFactor=1.0
                    )
                ),
                process=False
            )

            output_glb_path = Path(full_output_folder) / f'{filename}_{counter:05}_.glb'
            output_glb_path.parent.mkdir(exist_ok=True, parents=True)
            mesh.export(output_glb_path)
            relative_path = str(Path(subfolder) / f'{filename}_{counter:05}_.glb')
        else:
            relative_path = "GLB export disabled"

        return (depth_tensor, normal_vis_tensor, relative_path)

class MoGe2Panorama:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["v1", "v2"], {"default": "v2", "tooltip": "Select MoGe version. v2 outputs normals and metric scale; v1 has no normals. Use v2 unless you specifically need v1 behavior."}),
                "model_path": ("STRING", {"default": "C:/models/Ruicheng/moge-2-vitl-normal", "tooltip": "Local checkpoint folder for the selected model. If set and exists, overrides the version mapping. No network download is attempted."}),
                "image": ("IMAGE", {"tooltip": "Input equirectangular panorama (H×W×3). Output resolution matches this input exactly."}),
                "face_resolution": ("INT", {"default": 768, "min": 128, "max": 1536, "step": 64, "tooltip": "Per‑view render size (pixels) for each virtual camera. Higher → sharper details and smoother seams, but more VRAM/time. Typical: 512–1024 for 4096×2048 panos."}),
                "resolution_level": (["Low", "Medium", "High", "Ultra"], {"default": "Ultra", "tooltip": "Internal token budget for MoGe2 (affects accuracy). Low≈fastest, Ultra≈best. Use Ultra on A600 if VRAM allows."}),
                "view_fov_x_deg": ("INT", {"default": 110, "min": 60, "max": 120, "step": 5, "tooltip": "Horizontal FOV for virtual views. 90≈cube faces, 100–115 gives overlap for seam smoothing. Larger FOV increases overlap and robustness at the cost of distortion."}),
                "merge_method": (["z_buffer", "weighted", "affine_depth", "poisson_depth"], {"default": "z_buffer", "tooltip": "How to combine slices: z_buffer=nearest along ray (best metric fidelity); weighted=blend 3D points; affine_depth=align per‑slice scale/bias in the pano then blend; poisson_depth=gradient‑domain depth fusion."}),
                "zbuffer_mode": (["ray", "radial"], {"default": "ray", "tooltip": "Distance used by z‑buffer: ray=distance along the panorama ray (recommended), radial=||P|| from origin. Ray usually yields cleaner seams."}),
                "angle_power": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.5, "tooltip": "Weighted/affine/poisson only. View‑angle weight exponent (w∝cosθ^p). 0 disables angle weight. Typical 2–4. Higher favors view centers and reduces seams."}),
                "depth_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Weighted/affine/poisson only. Optional distance weight (w∝1/d^α). 0 disables. Typical 0.3–0.7 to limit far‑view bleed across seams."}),
                "fill_holes": ("BOOLEAN", {"default": True, "tooltip": "After merge, fill small invalid gaps by averaging 3×3 neighbors (points and normals). Helps minor cracks without over‑smoothing."}),
                "hole_iters": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1, "tooltip": "Hole‑fill passes. 1–2 is usually enough; >2 increases blur and runtime."}),
                "apply_mask": ("BOOLEAN", {"default": True, "tooltip": "Respect MoGe’s validity mask per view. Strongly recommended to avoid corrupt predictions being merged."}),
                "horizontal_wrap": ("BOOLEAN", {"default": True, "tooltip": "Use horizontal wrap when remapping onto the pano grid (u=0/1). Keep ON for equirectangular images to avoid seam artifacts."}),
                "skip_small_masks": ("BOOLEAN", {"default": True, "tooltip": "Skip any view whose valid mask area is tiny (e.g., blank sky). Prevents noisy slices from polluting the merge."}),
                "min_mask_ratio": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.1, "step": 0.001, "tooltip": "Minimum valid fraction for a view to be used. Typical 0.005–0.02 (0.5%–2%)."}),
                "wrap_consistency": ("BOOLEAN", {"default": True, "tooltip": "After depth fusion in image space, enforce left/right seam consistency by averaging boundary columns where both sides are valid."}),
                "align_in_disparity": ("BOOLEAN", {"default": True, "tooltip": "affine_depth only. Fit alignment in disparity (1/depth) instead of depth. Improves cross‑slice consistency and reduces scale drift."}),
                "polar_smooth": ("BOOLEAN", {"default": True, "tooltip": "Apply mild smoothing near the zenith/nadir to avoid spikes from equirectangular distortion."}),
                "polar_cap_ratio": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.2, "step": 0.005, "tooltip": "Amount of top/bottom (as H fraction) to smooth. Typical 0.04–0.10."}),
                "polar_blur_ks": ("INT", {"default": 7, "min": 3, "max": 31, "step": 2, "tooltip": "Odd Gaussian kernel size for polar smoothing. Typical 5–13."}),
                "export_per_view": ("BOOLEAN", {"default": False, "tooltip": "Export each virtual view’s prediction as PLY/GLB for QA (slow; use for debugging coverage/masks)."}),
                "per_view_export_format": (["ply", "glb", "both"], {"default": "ply", "tooltip": "Per‑view export format. PLY is lighter; GLB builds a textured triangulated mesh per view."}),
                "per_view_prefix": ("STRING", {"default": "3D/MoGe_Pano_Views", "tooltip": "Output prefix for per‑view exports (relative to ComfyUI outputs)."}),
                "output_pcl": ("BOOLEAN", {"default": True, "tooltip": "Export merged world‑space point cloud (.ply). Useful for quick inspection; large for big panos."}),
                "output_glb": ("BOOLEAN", {"default": False, "tooltip": "Export textured GLB of the merged panorama mesh. Enable when you need a viewer‑ready 3D file."}),
                "mesh_wrap_x": ("BOOLEAN", {"default": True, "tooltip": "Close the panorama seam (u=0/1) by connecting first/last columns with duplicated UVs. Prevents visible cracks in some viewers."}),
                "glb_rotate_x_deg": ("INT", {"default": 90, "min": -180, "max": 180, "step": 15, "tooltip": "Extra X rotation applied to the exported GLB to match viewer conventions. +90 often aligns Y‑up/Z‑forward viewers."}),
                "filename_prefix": ("STRING", {"default": "3D/MoGe_Pano", "tooltip": "Output prefix for PLY/GLB (relative to ComfyUI outputs)."}),
                "export_depth": ("BOOLEAN", {"default": False, "tooltip": "Save the fused depth map to disk (PNG16 in mm, EXR float, or both)."}),
                "depth_format": (["png16", "exr", "both"], {"default": "png16", "tooltip": "Depth output format(s): png16=16‑bit millimeters (0–65535), exr=32‑bit float meters, both=write both files."}),
                "depth_prefix": ("STRING", {"default": "3D/MoGe_Pano_Depth", "tooltip": "Output prefix for depth files (relative to ComfyUI outputs)."}),
                "use_fp16": ("BOOLEAN", {"default": True, "tooltip": "Use mixed precision for speed/VRAM. Turn off only if you see precision artifacts on unusual hardware."}),
            },
            "optional": {
                "mask_image": ("IMAGE", {"tooltip": "Optional label/mask image at panorama resolution. Grayscale: each unique intensity is a label. RGB: each unique color is a label."}),
                "multi_glb_from_mask": ("BOOLEAN", {"default": False, "tooltip": "If ON and a mask is provided, export one GLB per label region. Useful for separating furniture/architecture classes."}),
                "mask_ignore_zero": ("BOOLEAN", {"default": True, "tooltip": "Treat label 0 as background (skip exporting it). Turn OFF if you want a GLB for label 0."}),
                "min_label_area_ratio": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.5, "step": 0.001, "tooltip": "Minimum area for a label to be exported (fraction of all pixels). Typical 0.005–0.02."}),
                "multi_glb_prefix": ("STRING", {"default": "3D/MoGe_Pano_Label", "tooltip": "Output prefix for per‑label GLBs (relative to ComfyUI outputs)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("depth", "normal", "pcl_path", "glb_path", "depth_file")
    FUNCTION = "process"
    CATEGORY = "MoGe2"
    OUTPUT_NODE = True
    DESCRIPTION = "MoGe-2 panorama inference with metric-preserving merge (z-buffer default)."

    def _to_numpy_image(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.ndim == 4:
            image = image[0]
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return image

    def _numpy_to_tensor_image(self, img_np: np.ndarray) -> torch.Tensor:
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        img_np = img_np.astype(np.float32) / 255.0
        if img_np.ndim == 3:
            img_np = img_np[None, ...]
        return torch.from_numpy(img_np)

    def _get_resolution_level_int(self, resolution_level: str) -> int:
        return {'Low': 0, 'Medium': 5, 'High': 9, 'Ultra': 30}.get(resolution_level, 9)

    def _extract_rotation(self, extrinsic: np.ndarray) -> np.ndarray:
        if extrinsic.shape == (4, 4):
            return extrinsic[:3, :3].astype(np.float32)
        if extrinsic.shape == (3, 4):
            return extrinsic[:3, :3].astype(np.float32)
        if extrinsic.shape == (3, 3):
            return extrinsic.astype(np.float32)
        raise ValueError(f"Unexpected extrinsic shape: {extrinsic.shape}")

    def _remap_multi(self, arr: np.ndarray, mapx: np.ndarray, mapy: np.ndarray, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) -> np.ndarray:
        # Supports HxWxC arrays; returns remapped HxWxC
        return cv2.remap(arr, mapx, mapy, interpolation=interpolation, borderMode=borderMode)

    def _uv_to_pixel(self, uv: np.ndarray, width: int, height: int) -> np.ndarray:
        mapx = (np.clip(uv[..., 0], 0.0, 1.0) * (width - 1)).astype(np.float32)
        mapy = (np.clip(uv[..., 1], 0.0, 1.0) * (height - 1)).astype(np.float32)
        return np.stack([mapx, mapy], axis=-1)

    def _project_dirs(self, dirs_world: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
        # Project world-space unit directions into a camera (world->camera R), returning normalized uv in [0,1] and z depth
        if extrinsic.shape == (4, 4) or extrinsic.shape == (3, 4):
            R = extrinsic[:3, :3].astype(np.float32)
        else:
            R = extrinsic.astype(np.float32)
        K = intrinsic.astype(np.float32)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # Row-vector: v_cam = v_world @ R.T
        v_cam = dirs_world @ R.T
        z = v_cam[..., 2]
        x = v_cam[..., 0]
        y = v_cam[..., 1]
        eps = 1e-8
        u = fx * (x / (z + eps)) + cx
        v = fy * (y / (z + eps)) + cy
        uv = np.stack([u, v], axis=-1)
        return uv, z

    def _image_uv(self, width: int, height: int, dtype=np.float32) -> np.ndarray:
        u = np.linspace(0.0, 1.0, width, dtype=dtype)
        v = np.linspace(0.0, 1.0, height, dtype=dtype)
        uu, vv = np.meshgrid(u, v, indexing='xy')
        return np.stack([uu, vv], axis=-1)

    def _fill_holes_average(self, points: np.ndarray, normals: Optional[np.ndarray], mask: np.ndarray, iters: int = 1):
        if iters <= 0:
            return points, normals, mask
        k = np.ones((3, 3), dtype=np.float32)
        pts = points.copy()
        nrm = normals.copy() if normals is not None else None
        m = mask.copy()
        for _ in range(iters):
            m_f = m.astype(np.float32)
            cnt = cv2.filter2D(m_f, -1, k, borderType=cv2.BORDER_DEFAULT)
            sumx = cv2.filter2D(pts[..., 0] * m_f, -1, k, borderType=cv2.BORDER_DEFAULT)
            sumy = cv2.filter2D(pts[..., 1] * m_f, -1, k, borderType=cv2.BORDER_DEFAULT)
            sumz = cv2.filter2D(pts[..., 2] * m_f, -1, k, borderType=cv2.BORDER_DEFAULT)
            fill_mask = (~m) & (cnt > 0)
            if not np.any(fill_mask):
                break
            pts_new = np.zeros_like(pts)
            pts_new[..., 0] = sumx / (cnt + 1e-6)
            pts_new[..., 1] = sumy / (cnt + 1e-6)
            pts_new[..., 2] = sumz / (cnt + 1e-6)
            pts[fill_mask] = pts_new[fill_mask]
            m[fill_mask] = True
            if nrm is not None:
                sumnx = cv2.filter2D(nrm[..., 0] * m_f, -1, k, borderType=cv2.BORDER_DEFAULT)
                sumny = cv2.filter2D(nrm[..., 1] * m_f, -1, k, borderType=cv2.BORDER_DEFAULT)
                sumnz = cv2.filter2D(nrm[..., 2] * m_f, -1, k, borderType=cv2.BORDER_DEFAULT)
                nrm_new = np.zeros_like(nrm)
                nrm_new[..., 0] = sumnx / (cnt + 1e-6)
                nrm_new[..., 1] = sumny / (cnt + 1e-6)
                nrm_new[..., 2] = sumnz / (cnt + 1e-6)
                norm = np.linalg.norm(nrm_new, axis=-1, keepdims=True) + 1e-8
                nrm_new = nrm_new / norm
                nrm[fill_mask] = nrm_new[fill_mask]
        return pts, nrm, m

    def _mesh_from_pano_wrapx(self, points: np.ndarray, uv_grid: np.ndarray, normals: Optional[np.ndarray], mask: np.ndarray):
        """Build a seam-closed triangle mesh from a pano grid with horizontal wrap.
        Returns (faces, vertices, vertex_uvs, vertex_normals or None).
        """
        H, W = points.shape[:2]
        vid = -np.ones((H, W), dtype=np.int32)
        vertices = []
        vuv = []
        vnorm = [] if normals is not None else None
        # Base vertices
        for y in range(H):
            for x in range(W):
                if not mask[y, x]:
                    continue
                p = points[y, x]
                if not np.isfinite(p).all():
                    continue
                vid[y, x] = len(vertices)
                vertices.append(p)
                vuv.append(uv_grid[y, x])
                if normals is not None:
                    vnorm.append(normals[y, x])
        vertices = np.array(vertices, dtype=np.float32)
        vuv = np.array(vuv, dtype=np.float32)
        if normals is not None:
            vnorm = np.array(vnorm, dtype=np.float32)

        # Duplicate x=0 seam vertices with u=1 for wrap faces
        seam_dup_index = -np.ones((H,), dtype=np.int32)
        for y in range(H):
            if vid[y, 0] >= 0:
                seam_dup_index[y] = len(vertices)
                vertices = np.vstack([vertices, points[y, 0][None, :]])
                uv = uv_grid[y, 0].copy()
                uv[0] = 1.0
                vuv = np.vstack([vuv, uv[None, :]])
                if normals is not None:
                    vnorm = np.vstack([vnorm, normals[y, 0][None, :]])

        # Faces
        faces = []
        for y in range(H - 1):
            for x in range(W - 1):
                a = vid[y, x]
                b = vid[y, x + 1]
                c = vid[y + 1, x]
                d = vid[y + 1, x + 1]
                if a >= 0 and b >= 0 and c >= 0:
                    faces.append([a, c, b])
                if c >= 0 and d >= 0 and b >= 0:
                    faces.append([c, d, b])
            # wrap face between x=W-1 and x=0 using seam duplicates for the 0-side
            a = vid[y, W - 1]
            b = seam_dup_index[y]
            c = vid[y + 1, W - 1]
            d = seam_dup_index[y + 1]
            if a >= 0 and b >= 0 and c >= 0:
                faces.append([a, c, b])
            if c >= 0 and d >= 0 and b >= 0:
                faces.append([c, d, b])
        faces = np.array(faces, dtype=np.int32)
        if normals is not None:
            return faces, vertices, vuv, vnorm
        else:
            return faces, vertices, vuv, None

    def process(self,
                model: str,
                model_path: str,
                image,
                face_resolution: int,
                resolution_level: str,
                view_fov_x_deg: int,
                merge_method: str,
                zbuffer_mode: str,
                angle_power: float,
                depth_alpha: float,
                fill_holes: bool,
                hole_iters: int,
                apply_mask: bool,
                horizontal_wrap: bool,
                skip_small_masks: bool,
                min_mask_ratio: float,
                wrap_consistency: bool,
                align_in_disparity: bool,
                polar_smooth: bool,
                polar_cap_ratio: float,
                polar_blur_ks: int,
                export_per_view: bool,
                per_view_export_format: str,
                per_view_prefix: str,
                output_pcl: bool,
                output_glb: bool,
                mesh_wrap_x: bool,
                glb_rotate_x_deg: int,
                filename_prefix: str,
                export_depth: bool,
                depth_format: str,
                depth_prefix: str,
                use_fp16: bool,
                mask_image: Optional[Union[np.ndarray, torch.Tensor]] = None,
                multi_glb_from_mask: bool = False,
                mask_ignore_zero: bool = True,
                min_label_area_ratio: float = 0.005,
                multi_glb_prefix: str = "3D/MoGe_Pano_Label",
                ) -> Tuple[torch.Tensor, torch.Tensor, str, str, str]:

        # Resolve model path from version unless an override path is provided
        version_to_default_local = {
            "v1": "C:/models/Ruicheng/moge-vitl",
            "v2": "C:/models/Ruicheng/moge-2-vitl-normal",
        }
        model_version = model
        if isinstance(model_path, str) and len(model_path.strip()) > 0 and Path(model_path).exists():
            resolved_path = model_path
        else:
            resolved_path = version_to_default_local.get(model_version)
        if resolved_path is None or not Path(resolved_path).exists():
            raise FileNotFoundError(f"Could not find local weights for {model_version}. Checked: {resolved_path or '(none)'} and override '{model_path}'.")

        # Load model locally (no network)
        model_instance = import_model_class_by_version(model_version).from_pretrained(resolved_path).cuda().eval()

        # Prepare image (equirectangular pano)
        pano = self._to_numpy_image(image)
        pano_h, pano_w = pano.shape[:2]

        # Build virtual cameras (icosahedron, 90° FOV)
        extrinsics, intrinsics_list = get_panorama_cameras(fov_x_deg=float(view_fov_x_deg))

        # Split panorama into view images
        view_images = split_panorama_image(pano, extrinsics, intrinsics_list, resolution=face_resolution)

        # Run inference per view
        res_level_int = self._get_resolution_level_int(resolution_level)
        per_points: List[np.ndarray] = []
        per_normals: List[Optional[np.ndarray]] = []
        per_masks: List[np.ndarray] = []
        # For Poisson depth fusion (view-domain inputs)
        view_distance_maps: List[np.ndarray] = []
        view_pred_masks: List[np.ndarray] = []
        face_h, face_w = face_resolution, face_resolution
        for vi, img in enumerate(view_images):
            # prepare tensor
            img_t = torch.tensor(img, dtype=torch.float16 if use_fp16 else torch.float32, device=torch.device('cuda')).permute(2, 0, 1) / 255.0
            out = model_instance.infer(img_t, resolution_level=res_level_int, apply_mask=apply_mask, fov_x=float(view_fov_x_deg), use_fp16=use_fp16)
            out = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}

            points = out.get('points', None)
            depth = out.get('depth', None)
            mask = out.get('mask', None)
            normal = out.get('normal', None)

            # Ensure shapes
            if points is None or depth is None or mask is None:
                # view not usable
                per_points.append(np.full((face_h, face_w, 3), np.inf, dtype=np.float32))
                per_normals.append(None)
                per_masks.append(np.zeros((face_h, face_w), dtype=bool))
                continue

            # Convert types
            points = points.astype(np.float32)
            mask = mask.astype(bool)
            if normal is not None:
                normal = normal.astype(np.float32)

            # Optionally skip views with too few valid pixels
            if apply_mask and skip_small_masks:
                valid_px = int(mask.sum())
                min_px = int(face_resolution * face_resolution * float(min_mask_ratio))
                if valid_px < max(16, min_px):
                    per_points.append(None)
                    per_normals.append(None)
                    per_masks.append(None)
                    continue

            # Apply mask to invalidate
            if apply_mask:
                invalid = ~mask
                points[invalid] = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
                if normal is not None:
                    normal[invalid] = 0.0

            per_points.append(points)
            per_normals.append(normal)
            per_masks.append(mask)
            # Record per-view radial distance for Poisson fusion (sanitize invalid)
            dist_view = np.linalg.norm(points, axis=-1).astype(np.float32)
            if apply_mask:
                # Replace invalid distances with a benign positive value; masks will exclude them later
                dist_view[~mask] = 1.0
            view_distance_maps.append(dist_view)
            view_pred_masks.append(mask.astype(bool))

            # Optional: export per-view debug assets (PLY/GLB)
            if export_per_view:
                try:
                    R = self._extract_rotation(extrinsics[vi])  # world->camera rotation
                    # Transform per-view camera points to world frame using row-vector convention
                    pts_world_view = (points.reshape(-1, 3) @ R).reshape(points.shape)
                    # Export PLY
                    if per_view_export_format in ("ply", "both"):
                        # Export as point cloud (PLY) using masked points directly (no mesh dependency)
                        m = mask if mask is not None else np.ones(points.shape[:2], dtype=bool)
                        verts = pts_world_view[m]
                        cols = img[m]
                        verts = verts * np.array([1, -1, -1], dtype=np.float32)
                        pc = trimesh.PointCloud(verts, colors=cols)
                        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(per_view_prefix, folder_paths.get_output_directory())
                        out_path = Path(full_output_folder) / f"{filename}_{counter:05}_view{vi:02}.ply"
                        out_path.parent.mkdir(exist_ok=True, parents=True)
                        pc.export(out_path)
                    # Export GLB
                    if per_view_export_format in ("glb", "both"):
                        m = mask if mask is not None else np.ones(points.shape[:2], dtype=bool)
                        uv_grid_view = self._image_uv(width=points.shape[1], height=points.shape[0])
                        # Include normals if available
                        if normal is not None:
                            # Rotate normals to world like points
                            n_world_view = (normal.reshape(-1, 3) @ R).reshape(normal.shape)
                            faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
                                pts_world_view,
                                img.astype(np.float32) / 255.0,
                                uv_grid_view,
                                n_world_view.astype(np.float32),
                                mask=m,
                                tri=True,
                            )
                        else:
                            faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                                pts_world_view,
                                img.astype(np.float32) / 255.0,
                                uv_grid_view,
                                mask=m,
                                tri=True,
                            )
                            vertex_normals = None
                        # Orientation and UV convention; then extra X-rotation
                        vertices = vertices * np.array([1, -1, -1], dtype=np.float32)
                        vertex_uvs = vertex_uvs * np.array([1, -1], dtype=np.float32) + np.array([0, 1], dtype=np.float32)
                        if vertex_normals is not None:
                            vertex_normals = vertex_normals * np.array([1, -1, -1], dtype=np.float32)
                        if glb_rotate_x_deg != 0:
                            theta = np.deg2rad(glb_rotate_x_deg)
                            c, s = np.cos(theta), np.sin(theta)
                            Rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=np.float32)
                            vertices = (vertices @ Rx.T)
                            if vertex_normals is not None:
                                vertex_normals = (vertex_normals @ Rx.T)
                        mesh = trimesh.Trimesh(
                            vertices=vertices,
                            faces=faces,
                            vertex_normals=vertex_normals,
                            visual=trimesh.visual.texture.TextureVisuals(
                                uv=vertex_uvs,
                                material=trimesh.visual.material.PBRMaterial(
                                    baseColorTexture=Image.fromarray(img),
                                    metallicFactor=0.5,
                                    roughnessFactor=1.0,
                                ),
                            ),
                            process=False,
                        )
                        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(per_view_prefix, folder_paths.get_output_directory())
                        out_path = Path(full_output_folder) / f"{filename}_{counter:05}_view{vi:02}.glb"
                        out_path.parent.mkdir(exist_ok=True, parents=True)
                        mesh.export(out_path)
                except Exception as e:
                    log.exception("Per-view export failed for view %d: %s", vi, e)

        # Prepare panorama grid and projection maps
        uv = self._image_uv(width=pano_w, height=pano_h)
        spherical_dirs = spherical_uv_to_directions(uv)  # (H, W, 3) world-space unit directions

        num_views = len(per_points)
        export_depth_map = None
        if merge_method == 'z_buffer':
            best_dist = np.full((pano_h, pano_w), np.inf, dtype=np.float32)
            best_point_world = np.full((pano_h, pano_w, 3), np.nan, dtype=np.float32)
            best_normal_world = np.full((pano_h, pano_w, 3), 0, dtype=np.float32)
            final_mask = np.zeros((pano_h, pano_w), dtype=bool)
        elif merge_method == 'weighted':
            accum_w = np.zeros((pano_h, pano_w), dtype=np.float32)
            accum_pts = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
            accum_norm = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
            final_mask = np.zeros((pano_h, pano_w), dtype=bool)
        else:
            # depth-domain fusion containers
            dist_maps: List[np.ndarray] = []
            mask_maps: List[np.ndarray] = []
            weight_maps: List[np.ndarray] = []
            rotated_normals: List[Optional[np.ndarray]] = []
            # also collect per-view rotations for norm blending when needed
            final_mask = None  # will define later

        for i in range(num_views):
            pts = per_points[i]
            msk = per_masks[i]
            if pts is None or msk is None:
                continue

            # Project pano directions into this view
            projected_uv, projected_depth = self._project_dirs(spherical_dirs, extrinsics[i], intrinsics_list[i])
            proj_valid = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)

            # Compute pixel maps for remap
            proj_pixels = self._uv_to_pixel(projected_uv, width=pts.shape[1], height=pts.shape[0])

            # Remap points and mask to pano grid
            borderMode = cv2.BORDER_WRAP if horizontal_wrap else cv2.BORDER_REPLICATE
            remap_points = self._remap_multi(pts, proj_pixels[..., 0], proj_pixels[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=borderMode)
            remap_mask = self._remap_multi(msk.astype(np.uint8), proj_pixels[..., 0], proj_pixels[..., 1], interpolation=cv2.INTER_NEAREST, borderMode=borderMode) > 0
            valid_here = proj_valid & remap_mask & np.isfinite(remap_points).all(axis=-1)
            if not np.any(valid_here):
                continue

            # Rotate to world
            # utils3d extrinsics are world->camera: x_cam = R * x_world + t
            # Convert camera-space points to world: x_world = R^T * x_cam
            # With row vectors, this is x_world_row = x_cam_row @ R (not R.T)
            R = self._extract_rotation(extrinsics[i])  # world->camera rotation
            pts_cam = remap_points.reshape(-1, 3)
            pts_world = (pts_cam @ R).reshape(pano_h, pano_w, 3)
            if zbuffer_mode == 'ray':
                dist = np.sum(pts_world * spherical_dirs, axis=-1)
            else:
                dist = np.linalg.norm(pts_world, axis=-1)

            if merge_method == 'z_buffer':
                # Z-buffer update
                if zbuffer_mode == 'ray':
                    better = (dist > 0) & (dist < best_dist) & valid_here
                else:
                    better = (dist < best_dist) & valid_here
                if np.any(better):
                    best_dist[better] = dist[better]
                    best_point_world[better] = pts_world[better]
                    final_mask[better] = True

                    # normals
                    normal = per_normals[i]
                    if normal is not None:
                        remap_normal = self._remap_multi(normal, proj_pixels[..., 0], proj_pixels[..., 1], interpolation=cv2.INTER_LINEAR)
                        n_cam = remap_normal.reshape(-1, 3)
                        n_world = (n_cam @ R)
                        n_world = n_world / (np.linalg.norm(n_world, axis=-1, keepdims=True) + 1e-8)
                        n_world = n_world.reshape(pano_h, pano_w, 3)
                        best_normal_world[better] = n_world[better]
            elif merge_method == 'weighted':
                # Weighted merge by view angle (and optional distance)
                # Camera forward in world: third row of R
                forward_world = R[2, :]
                cosang = np.clip((spherical_dirs * forward_world[None, None, :]).sum(axis=-1), 0.0, 1.0)
                w = np.zeros_like(dist)
                # angle weighting
                if angle_power > 0:
                    w = cosang ** float(angle_power)
                else:
                    w = np.ones_like(dist)
                # depth weighting
                if depth_alpha > 0:
                    w = w / np.power(dist + 1e-6, float(depth_alpha))
                w = np.where(valid_here, w, 0.0)
                if np.any(w > 0):
                    accum_pts += (w[..., None] * pts_world)
                    final_mask |= (w > 0)
                    normal = per_normals[i]
                    if normal is not None:
                        remap_normal = self._remap_multi(normal, proj_pixels[..., 0], proj_pixels[..., 1], interpolation=cv2.INTER_LINEAR)
                        n_cam = remap_normal.reshape(-1, 3)
                        n_world = (n_cam @ R).reshape(pano_h, pano_w, 3)
                        n_world = n_world / (np.linalg.norm(n_world, axis=-1, keepdims=True) + 1e-8)
                        accum_norm += (w[..., None] * n_world)
                    accum_w += w
            else:
                # depth-domain fusion precompute: distances, weights, masks, normals
                forward_world = R[2, :]
                cosang = np.clip((spherical_dirs * forward_world[None, None, :]).sum(axis=-1), 0.0, 1.0)
                w = cosang ** float(angle_power) if angle_power > 0 else np.ones_like(dist)
                if depth_alpha > 0:
                    w = w / np.power(np.maximum(dist, 1e-6), float(depth_alpha))
                w = np.where(valid_here, w, 0.0).astype(np.float32)
                # Store maps for later affine/Poisson fusion
                dist_maps.append(dist.astype(np.float32))
                mask_maps.append(valid_here.astype(bool))
                weight_maps.append(w)
                normal = per_normals[i]
                if normal is not None:
                    remap_normal = self._remap_multi(normal, proj_pixels[..., 0], proj_pixels[..., 1], interpolation=cv2.INTER_LINEAR)
                    n_cam = remap_normal.reshape(-1, 3)
                    n_world = (n_cam @ R).reshape(pano_h, pano_w, 3)
                    n_world = n_world / (np.linalg.norm(n_world, axis=-1, keepdims=True) + 1e-8)
                    rotated_normals.append(n_world.astype(np.float32))
                else:
                    rotated_normals.append(None)

        # Consolidate weighted merge results and optional hole fill
        if merge_method == 'weighted':
            valid = accum_w > 1e-6
            best_point_world = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
            best_point_world[valid] = (accum_pts[valid] / accum_w[valid, None]).astype(np.float32)
            best_normal_world = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
            if np.any(accum_norm != 0):
                n = np.zeros_like(best_normal_world)
                n[valid] = (accum_norm[valid] / (accum_w[valid, None] + 1e-8))
                n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
                best_normal_world = n.astype(np.float32)
            final_mask = valid
            # Export depth along pano ray
            export_depth_map = np.maximum(np.sum(best_point_world * spherical_dirs, axis=-1), 0.0)
        elif merge_method in ('affine_depth', 'poisson_depth'):
            # Depth-domain fusion
            pano_depth = None
            if merge_method == 'affine_depth' and len(dist_maps) > 0:
                # Running baseline alignment and blending
                accum_depth = np.full((pano_h, pano_w), np.inf, dtype=np.float32)
                accum_w = np.zeros((pano_h, pano_w), dtype=np.float32)
                # helper to compute weighted LS a,b s.t. y ≈ a x + b
                def wls_affine(x, y, w):
                    x = x.astype(np.float32); y = y.astype(np.float32); w = w.astype(np.float32)
                    S = np.sum(w)
                    if S < 1e-6:
                        return 1.0, 0.0
                    Sx = np.sum(w * x)
                    Sy = np.sum(w * y)
                    Sxx = np.sum(w * x * x)
                    Sxy = np.sum(w * x * y)
                    denom = S * Sxx - Sx * Sx
                    if abs(denom) < 1e-6:
                        return 1.0, 0.0
                    a = (S * Sxy - Sx * Sy) / denom
                    b = (Sxx * Sy - Sx * Sxy) / denom
                    return float(a), float(b)
                # integrate views
                for vi in range(len(dist_maps)):
                    d = dist_maps[vi]
                    m = mask_maps[vi]
                    w = weight_maps[vi]
                    if pano_depth is None:
                        # initialize with first valid map
                        pano_depth = np.where(m, d, np.inf).astype(np.float32)
                        accum_depth = np.where(m, d * w, 0.0).astype(np.float32)
                        accum_w = np.where(m, w, 0.0).astype(np.float32)
                    else:
                        overlap = m & (accum_w > 0)
                        if np.any(overlap):
                            x_d = d[overlap]
                            y_d = (accum_depth[overlap] / (accum_w[overlap] + 1e-6))
                            w_overlap = w[overlap]
                            if align_in_disparity:
                                # work in disparity: s = 1/depth
                                x = 1.0 / np.maximum(x_d, 1e-6)
                                y = 1.0 / np.maximum(y_d, 1e-6)
                                a, b = wls_affine(x, y, w_overlap)
                                s = a * (1.0 / np.maximum(d, 1e-6)) + b
                                d_aligned = 1.0 / np.maximum(s, 1e-6)
                            else:
                                a, b = wls_affine(x_d, y_d, w_overlap)
                                d_aligned = a * d + b
                        else:
                            d_aligned = d
                        # blend by weights
                        accum_depth = np.where(m, accum_depth + d_aligned * w, accum_depth)
                        accum_w = np.where(m, accum_w + w, accum_w)
                        pano_depth = accum_depth / (accum_w + 1e-6)
                final_mask = accum_w > 1e-6
            elif merge_method == 'poisson_depth' and len(view_distance_maps) > 0:
                try:
                    from .moge.utils.panorama import merge_panorama_depth as _merge_panorama_depth
                    pano_depth, final_mask = _merge_panorama_depth(pano_w, pano_h, view_distance_maps, view_pred_masks, extrinsics, intrinsics_list)
                except Exception as e:
                    log.exception("Poisson depth merge failed, falling back to affine_depth: %s", e)
                    # Simple fallback: average with weights
                    accum_depth = np.zeros((pano_h, pano_w), dtype=np.float32)
                    accum_w = np.zeros((pano_h, pano_w), dtype=np.float32)
                    for d, m, w in zip(dist_maps, mask_maps, weight_maps):
                        accum_depth += np.where(m, d * w, 0.0)
                        accum_w += np.where(m, w, 0.0)
                    pano_depth = np.where(accum_w > 0, accum_depth / (accum_w + 1e-6), 0.0).astype(np.float32)
                    final_mask = accum_w > 0

            # Optional wrap seam consistency
            if pano_depth is None:
                pano_depth = np.zeros((pano_h, pano_w), dtype=np.float32)
                final_mask = np.zeros((pano_h, pano_w), dtype=bool)
            if wrap_consistency and np.any(final_mask):
                left = pano_depth[:, 0]
                right = pano_depth[:, -1]
                both = (final_mask[:, 0] & final_mask[:, -1])
                avg = (left + right) * 0.5
                pano_depth[:, 0][both] = avg[both]
                pano_depth[:, -1][both] = avg[both]

            # Optional polar smoothing
            if polar_smooth and np.any(final_mask):
                cap = max(1, int(polar_cap_ratio * pano_h))
                k = max(3, int(polar_blur_ks) | 1)  # ensure odd
                if cap > 0:
                    top = pano_depth[:cap]
                    bot = pano_depth[-cap:]
                    pano_depth[:cap] = cv2.GaussianBlur(top, (k, k), 0)
                    pano_depth[-cap:] = cv2.GaussianBlur(bot, (k, k), 0)

            # Convert pano depth to world points along rays
            best_point_world = spherical_dirs * pano_depth[..., None]
            export_depth_map = pano_depth.astype(np.float32)
            # Blend normals across views using the same weights collected
            if any(n is not None for n in rotated_normals):
                accum_n = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
                accum_w_n = np.zeros((pano_h, pano_w), dtype=np.float32)
                for n, w, m in zip(rotated_normals, weight_maps, mask_maps):
                    if n is None:
                        continue
                    ww = np.where(m, w, 0.0)
                    accum_n += ww[..., None] * n
                    accum_w_n += ww
                best_normal_world = np.zeros_like(accum_n)
                valid_n = accum_w_n > 1e-6
                best_normal_world[valid_n] = accum_n[valid_n] / (accum_w_n[valid_n, None] + 1e-6)
                nn = np.linalg.norm(best_normal_world, axis=-1, keepdims=True) + 1e-8
                best_normal_world = best_normal_world / nn
            else:
                best_normal_world = np.zeros((pano_h, pano_w, 3), dtype=np.float32)

        # For z-buffer, choose depth map to export
        if merge_method == 'z_buffer':
            if zbuffer_mode == 'ray':
                export_depth_map = best_dist.copy()
            else:
                export_depth_map = np.maximum(np.sum(best_point_world * spherical_dirs, axis=-1), 0.0)

        # Simple hole filling (optional)
        if fill_holes:
            best_point_world, best_normal_world, final_mask = self._fill_holes_average(best_point_world, best_normal_world, final_mask, iters=hole_iters)

        # Build visualization images
        if np.any(final_mask):
            if merge_method in ('weighted', 'affine_depth', 'poisson_depth'):
                dvals = np.linalg.norm(best_point_world, axis=-1)
            else:
                dvals = best_dist.copy()
            masked = dvals[final_mask]
            if masked.size > 0:
                dmin, dmax = masked.min(), masked.max()
                if dmax > dmin:
                    dnorm = (dvals - dmin) / (dmax - dmin)
                else:
                    dnorm = np.ones_like(dvals) * 0.5
            else:
                dnorm = np.zeros_like(dvals)
            dnorm_inv = 1.0 - dnorm
            dnorm_inv[~final_mask] = 0.0
            depth_vis = (dnorm_inv * 255).astype(np.uint8)
            depth_vis_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2RGB)
        else:
            depth_vis_rgb = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)

        normal_vis = colorize_normal(best_normal_world.astype(np.float32)) if np.any(final_mask) else np.zeros_like(pano)

        # Export point cloud if requested
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        pcl_relative_path = ""
        glb_relative_path = ""
        if output_pcl and np.any(final_mask):
            verts = best_point_world[final_mask]
            colors = pano[final_mask]
            pc = trimesh.PointCloud(verts, colors=colors)
            output_ply_path = Path(full_output_folder) / f"{filename}_{counter:05}_.ply"
            output_ply_path.parent.mkdir(exist_ok=True, parents=True)
            pc.export(output_ply_path)
            pcl_relative_path = str(Path(subfolder) / f"{filename}_{counter:05}_.ply")
        else:
            pcl_relative_path = "Point cloud export disabled or empty mask"

        # Optional GLB export (textured mesh over the pano grid)
        label_glb_paths: List[str] = []

        if output_glb and np.any(final_mask):
            try:
                # Build a triangulated image mesh using the pano grid
                uv_grid = self._image_uv(width=pano_w, height=pano_h)
                # If normals available, try to include; otherwise export without normals
                if mesh_wrap_x:
                    faces, vertices, vertex_uvs, vertex_normals = self._mesh_from_pano_wrapx(
                        best_point_world.astype(np.float32), uv_grid.astype(np.float32),
                        best_normal_world.astype(np.float32) if best_normal_world is not None else None,
                        final_mask,
                    )
                else:
                    if best_normal_world is not None and np.any(final_mask):
                        faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
                            best_point_world,
                            pano.astype(np.float32) / 255.0,
                            uv_grid,
                            best_normal_world.astype(np.float32),
                            mask=final_mask,
                            tri=True,
                        )
                    else:
                        faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                            best_point_world,
                            pano.astype(np.float32) / 255.0,
                            uv_grid,
                            mask=final_mask,
                            tri=True,
                        )
                        vertex_normals = None

                # Match orientation and UV; then apply user-specified extra X-rotation
                vertices = vertices * np.array([1, -1, -1], dtype=np.float32)
                vertex_uvs = vertex_uvs * np.array([1, -1], dtype=np.float32) + np.array([0, 1], dtype=np.float32)
                if vertex_normals is not None:
                    vertex_normals = vertex_normals * np.array([1, -1, -1], dtype=np.float32)

                # Extra clockwise rotation around X axis for GLB viewers
                if glb_rotate_x_deg != 0:
                    theta = np.deg2rad(glb_rotate_x_deg)
                    c, s = np.cos(theta), np.sin(theta)
                    Rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=np.float32)
                    vertices = (vertices @ Rx.T)
                    if vertex_normals is not None:
                        vertex_normals = (vertex_normals @ Rx.T)

                mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    vertex_normals=vertex_normals,
                    visual=trimesh.visual.texture.TextureVisuals(
                        uv=vertex_uvs,
                        material=trimesh.visual.material.PBRMaterial(
                            baseColorTexture=Image.fromarray(pano),
                            metallicFactor=0.5,
                            roughnessFactor=1.0,
                        ),
                    ),
                    process=False,
                )

                output_glb_path = Path(full_output_folder) / f"{filename}_{counter:05}_.glb"
                output_glb_path.parent.mkdir(exist_ok=True, parents=True)
                mesh.export(output_glb_path)
                glb_relative_path = str(Path(subfolder) / f"{filename}_{counter:05}_.glb")
            except Exception as e:
                log.exception("Failed to export GLB: %s", e)
                glb_relative_path = "GLB export failed"
        else:
            if output_glb:
                glb_relative_path = "GLB export skipped: empty mask"
            else:
                glb_relative_path = "GLB export disabled"

        # Optional: export multiple GLBs based on a label/mask image
        if multi_glb_from_mask and (mask_image is not None) and np.any(final_mask):
            try:
                mask_np = self._to_numpy_image(mask_image)
                if mask_np.ndim == 3 and mask_np.shape[2] == 3:
                    labels_map = (mask_np[..., 0].astype(np.int64) << 16) | (mask_np[..., 1].astype(np.int64) << 8) | (mask_np[..., 2].astype(np.int64))
                else:
                    if mask_np.ndim == 3 and mask_np.shape[2] == 1:
                        mask_np = mask_np[..., 0]
                    labels_map = mask_np.astype(np.int64)
                if labels_map.shape[:2] != (pano_h, pano_w):
                    labels_map = cv2.resize(labels_map, (pano_w, pano_h), interpolation=cv2.INTER_NEAREST)
                unique_labels = np.unique(labels_map)
                if mask_ignore_zero:
                    unique_labels = unique_labels[unique_labels != 0]
                total_px = pano_h * pano_w
                uv_grid = self._image_uv(width=pano_w, height=pano_h)

                for lbl in unique_labels:
                    region_mask = final_mask & (labels_map == int(lbl))
                    if np.count_nonzero(region_mask) < max(64, int(min_label_area_ratio * total_px)):
                        continue
                    # Build mesh for this label only
                    try:
                        if mesh_wrap_x:
                            faces, vertices, vertex_uvs, vertex_normals = self._mesh_from_pano_wrapx(
                                best_point_world.astype(np.float32), uv_grid.astype(np.float32),
                                best_normal_world.astype(np.float32) if best_normal_world is not None else None,
                                region_mask,
                            )
                        else:
                            if best_normal_world is not None and np.any(region_mask):
                                faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
                                    best_point_world,
                                    pano.astype(np.float32) / 255.0,
                                    uv_grid,
                                    best_normal_world.astype(np.float32),
                                    mask=region_mask,
                                    tri=True,
                                )
                            else:
                                faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                                    best_point_world,
                                    pano.astype(np.float32) / 255.0,
                                    uv_grid,
                                    mask=region_mask,
                                    tri=True,
                                )
                                vertex_normals = None

                        # Orientation and extra rotation
                        vertices = vertices * np.array([1, -1, -1], dtype=np.float32)
                        vertex_uvs = vertex_uvs * np.array([1, -1], dtype=np.float32) + np.array([0, 1], dtype=np.float32)
                        if vertex_normals is not None:
                            vertex_normals = vertex_normals * np.array([1, -1, -1], dtype=np.float32)
                        if glb_rotate_x_deg != 0:
                            theta = np.deg2rad(glb_rotate_x_deg)
                            c, s = np.cos(theta), np.sin(theta)
                            Rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=np.float32)
                            vertices = (vertices @ Rx.T)
                            if vertex_normals is not None:
                                vertex_normals = (vertex_normals @ Rx.T)

                        mesh = trimesh.Trimesh(
                            vertices=vertices,
                            faces=faces,
                            vertex_normals=vertex_normals,
                            visual=trimesh.visual.texture.TextureVisuals(
                                uv=vertex_uvs,
                                material=trimesh.visual.material.PBRMaterial(
                                    baseColorTexture=Image.fromarray(pano),
                                    metallicFactor=0.5,
                                    roughnessFactor=1.0,
                                ),
                            ),
                            process=False,
                        )

                        full_output_folder, filename2, counter2, subfolder2, _ = folder_paths.get_save_image_path(multi_glb_prefix, folder_paths.get_output_directory())
                        out_path = Path(full_output_folder) / f"{filename2}_{counter2:05}_label{int(lbl)}.glb"
                        out_path.parent.mkdir(exist_ok=True, parents=True)
                        mesh.export(out_path)
                        label_glb_paths.append(str(Path(subfolder2) / f"{filename2}_{counter2:05}_label{int(lbl)}.glb"))
                    except Exception as e:
                        log.exception("Per-label GLB export failed for label %s: %s", str(lbl), e)
            except Exception as e:
                log.exception("Label mask processing failed: %s", e)

        # Optional depth file export
        depth_relative_path = ""
        if export_depth and (export_depth_map is not None) and np.any(final_mask):
            try:
                full_output_folder_d, filename_d, counter_d, subfolder_d, _ = folder_paths.get_save_image_path(depth_prefix, folder_paths.get_output_directory())
                paths = []
                dep = export_depth_map.copy().astype(np.float32)
                dep[~final_mask] = 0.0
                if depth_format in ("png16", "both"):
                    d16 = np.clip(dep * 1000.0, 0, 65535).astype(np.uint16)
                    out_p = Path(full_output_folder_d) / f"{filename_d}_{counter_d:05}_.png"
                    out_p.parent.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(out_p), d16)
                    paths.append(str(Path(subfolder_d) / f"{filename_d}_{counter_d:05}_.png"))
                if depth_format in ("exr", "both"):
                    out_e = Path(full_output_folder_d) / f"{filename_d}_{counter_d:05}_.exr"
                    out_e.parent.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(out_e), dep.astype(np.float32))
                    paths.append(str(Path(subfolder_d) / f"{filename_d}_{counter_d:05}_.exr"))
                depth_relative_path = "\n".join(paths)
            except Exception as e:
                log.exception("Failed to export depth file(s): %s", e)
                depth_relative_path = "Depth export failed"
        elif export_depth:
            depth_relative_path = "Depth export skipped: empty mask"

        # Convert to tensors for ComfyUI
        depth_tensor = self._numpy_to_tensor_image(depth_vis_rgb)
        normal_tensor = self._numpy_to_tensor_image(normal_vis)

        # If multi GLBs were exported, append the list to the glb_path output for visibility
        if label_glb_paths:
            extra = "\n".join(label_glb_paths)
            if isinstance(glb_relative_path, str) and len(glb_relative_path) > 0:
                glb_relative_path = glb_relative_path + "\n" + extra
            else:
                glb_relative_path = extra

        return (depth_tensor, normal_tensor, pcl_relative_path, glb_relative_path, depth_relative_path)

NODE_CLASS_MAPPINGS = {
    "RunMoGe2Process": RunMoGe2Process,
    "MoGe2Panorama": MoGe2Panorama,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunMoGe2Process": "MoGe2 Process",
    "MoGe2Panorama": "MoGe2 Panorama",
}
