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
                "model": (["v1", "v2"], {"default": "v2", "tooltip": "Select MoGe version. Panorama prefers v2 for metric scale and normals."}),
                "model_path": ("STRING", {"default": "C:/models/Ruicheng/moge-2-vitl-normal", "tooltip": "Optional local checkpoint path. If set and exists, overrides the version mapping. No network used."}),
                "image": ("IMAGE", {"tooltip": "Input equirectangular panorama (H×W×3) as a ComfyUI IMAGE tensor."}),
                "face_resolution": ("INT", {"default": 512, "min": 128, "max": 1536, "step": 64, "tooltip": "Resolution per virtual view used to split the panorama."}),
                "resolution_level": (["Low", "Medium", "High", "Ultra"], {"default": "High", "tooltip": "Model internal token resolution. Higher = better quality, slower."}),
                "merge_method": (["z_buffer"], {"default": "z_buffer", "tooltip": "Merge strategy. z_buffer picks the nearest metric distance per panorama ray."}),
                "apply_mask": ("BOOLEAN", {"default": True, "tooltip": "Apply model’s validity mask to ignore unreliable predictions."}),
                "output_pcl": ("BOOLEAN", {"default": True, "tooltip": "Export merged world-space point cloud (.ply)."}),
                "output_glb": ("BOOLEAN", {"default": False, "tooltip": "Export textured mesh (.glb) built over the panorama grid."}),
                "filename_prefix": ("STRING", {"default": "3D/MoGe_Pano", "tooltip": "Prefix under the ComfyUI output directory for saved files."}),
                "use_fp16": ("BOOLEAN", {"default": True, "tooltip": "Use FP16 inference to reduce VRAM and improve speed."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("depth", "normal", "pcl_path", "glb_path")
    FUNCTION = "process"
    CATEGORY = "MoGe2"
    OUTPUT_NODE = True
    DESCRIPTION = "MoGe-2 panorama inference with metric-preserving z-buffer merge."

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

    def process(self,
                model: str,
                model_path: str,
                image,
                face_resolution: int,
                resolution_level: str,
                merge_method: str,
                apply_mask: bool,
                output_pcl: bool,
                output_glb: bool,
                filename_prefix: str,
                use_fp16: bool) -> Tuple[torch.Tensor, torch.Tensor, str, str]:

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
        extrinsics, intrinsics_list = get_panorama_cameras()

        # Split panorama into view images
        view_images = split_panorama_image(pano, extrinsics, intrinsics_list, resolution=face_resolution)

        # Run inference per view
        res_level_int = self._get_resolution_level_int(resolution_level)
        per_points: List[np.ndarray] = []
        per_normals: List[Optional[np.ndarray]] = []
        per_masks: List[np.ndarray] = []
        face_h, face_w = face_resolution, face_resolution
        for img in view_images:
            # prepare tensor
            img_t = torch.tensor(img, dtype=torch.float16 if use_fp16 else torch.float32, device=torch.device('cuda')).permute(2, 0, 1) / 255.0
            out = model_instance.infer(img_t, resolution_level=res_level_int, apply_mask=apply_mask, fov_x=90.0, use_fp16=use_fp16)
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

            # Apply mask to invalidate
            if apply_mask:
                invalid = ~mask
                points[invalid] = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
                if normal is not None:
                    normal[invalid] = 0.0

            per_points.append(points)
            per_normals.append(normal)
            per_masks.append(mask)

        # Prepare panorama grid and projection maps
        uv = utils3d.numpy.image_uv(width=pano_w, height=pano_h)
        spherical_dirs = spherical_uv_to_directions(uv)  # (H, W, 3) world-space unit directions

        # Accumulators for z-buffer merge
        num_views = len(per_points)
        best_dist = np.full((pano_h, pano_w), np.inf, dtype=np.float32)
        best_point_world = np.full((pano_h, pano_w, 3), np.nan, dtype=np.float32)
        best_normal_world = np.full((pano_h, pano_w, 3), 0, dtype=np.float32)
        final_mask = np.zeros((pano_h, pano_w), dtype=bool)

        for i in range(num_views):
            pts = per_points[i]
            msk = per_masks[i]
            if pts is None or msk is None:
                continue

            # Project pano directions into this view
            projected_uv, projected_depth = utils3d.numpy.project_cv(spherical_dirs, extrinsics=extrinsics[i], intrinsics=intrinsics_list[i])
            proj_valid = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)

            # Compute pixel maps for remap
            proj_pixels = utils3d.numpy.uv_to_pixel(np.clip(projected_uv, 0, 1), width=pts.shape[1], height=pts.shape[0]).astype(np.float32)

            # Remap points and mask to pano grid
            remap_points = self._remap_multi(pts, proj_pixels[..., 0], proj_pixels[..., 1], interpolation=cv2.INTER_LINEAR)
            remap_mask = self._remap_multi(msk.astype(np.uint8), proj_pixels[..., 0], proj_pixels[..., 1], interpolation=cv2.INTER_NEAREST) > 0
            valid_here = proj_valid & remap_mask & np.isfinite(remap_points).all(axis=-1)
            if not np.any(valid_here):
                continue

            # Rotate to world
            R = self._extract_rotation(extrinsics[i])
            pts_cam = remap_points.reshape(-1, 3)
            pts_world = (pts_cam @ R.T).reshape(pano_h, pano_w, 3)
            dist = np.linalg.norm(pts_world, axis=-1)

            # Z-buffer update
            better = (dist < best_dist) & valid_here
            if np.any(better):
                best_dist[better] = dist[better]
                best_point_world[better] = pts_world[better]
                final_mask[better] = True

                # normals
                normal = per_normals[i]
                if normal is not None:
                    remap_normal = self._remap_multi(normal, proj_pixels[..., 0], proj_pixels[..., 1], interpolation=cv2.INTER_LINEAR)
                    # rotate and normalize
                    n_cam = remap_normal.reshape(-1, 3)
                    n_world = (n_cam @ R.T)
                    n_world = n_world / (np.linalg.norm(n_world, axis=-1, keepdims=True) + 1e-8)
                    n_world = n_world.reshape(pano_h, pano_w, 3)
                    best_normal_world[better] = n_world[better]

        # Build visualization images
        if np.any(final_mask):
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
        if output_glb and np.any(final_mask):
            try:
                # Build a triangulated image mesh using the pano grid
                uv_grid = utils3d.numpy.image_uv(width=pano_w, height=pano_h)
                # If normals available, try to include; otherwise export without normals
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

                # Match the single-view export orientation and UV convention
                vertices = vertices * np.array([1, -1, -1], dtype=np.float32)
                vertex_uvs = vertex_uvs * np.array([1, -1], dtype=np.float32) + np.array([0, 1], dtype=np.float32)
                if vertex_normals is not None:
                    vertex_normals = vertex_normals * np.array([1, -1, -1], dtype=np.float32)

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

        # Convert to tensors for ComfyUI
        depth_tensor = self._numpy_to_tensor_image(depth_vis_rgb)
        normal_tensor = self._numpy_to_tensor_image(normal_vis)

        return (depth_tensor, normal_tensor, pcl_relative_path, glb_relative_path)

NODE_CLASS_MAPPINGS = {
    "RunMoGe2Process": RunMoGe2Process,
    "MoGe2Panorama": MoGe2Panorama,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunMoGe2Process": "MoGe2 Process",
    "MoGe2Panorama": "MoGe2 Panorama",
}
