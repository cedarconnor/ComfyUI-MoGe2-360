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
                "model": (["v1", "v2"], {"default": "v2", "tooltip": "MoGe model version. Default v2 outputs normals and metric scale; v1 has no normals."}),
                "model_path": ("STRING", {"default": "C:/models/Ruicheng/moge-2-vitl-normal", "tooltip": "Local checkpoint folder. If set and exists, overrides the version mapping. No network download."}),
                "image": ("IMAGE", {"tooltip": "Equirectangular panorama (H×W×3) as a ComfyUI IMAGE tensor. Output resolution matches input."}),
                "face_resolution": ("INT", {"default": 512, "min": 128, "max": 1536, "step": 64, "tooltip": "Per-view resolution for virtual cameras (icosahedron). Higher = more detail and smoother seams."}),
                "resolution_level": (["Low", "Medium", "High", "Ultra"], {"default": "High", "tooltip": "Model token resolution. Higher improves quality at the cost of speed/VRAM."}),
                "view_fov_x_deg": ("INT", {"default": 90, "min": 60, "max": 120, "step": 5, "tooltip": "Virtual camera horizontal FOV. Increase (e.g., 100–110) to create more overlap for seam smoothing."}),
                "merge_method": (["z_buffer", "weighted"], {"default": "weighted", "tooltip": "Merge strategy. weighted blends overlapping views by angle/depth; z_buffer picks nearest distance per ray."}),
                "angle_power": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.5, "tooltip": "Weighted merge: weight ∝ cos(theta)^p. Higher p reduces seams by favoring view centers."}),
                "depth_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Weighted merge: optional weight ∝ 1/(distance^alpha). Use small values (0.3–0.7) to limit occlusion smearing."}),
                "fill_holes": ("BOOLEAN", {"default": True, "tooltip": "Post-merge hole fill using 3×3 neighbor averages for points (and normals)."}),
                "hole_iters": ("INT", {"default": 1, "min": 0, "max": 5, "step": 1, "tooltip": "Hole-fill iterations. 1–2 usually enough; 0 disables."}),
                "apply_mask": ("BOOLEAN", {"default": True, "tooltip": "Apply model validity mask to ignore unreliable pixels before merging."}),
                "horizontal_wrap": ("BOOLEAN", {"default": False, "tooltip": "Use horizontal wrap when remapping onto the pano grid. Helps at the left/right seam."}),
                "export_per_view": ("BOOLEAN", {"default": False, "tooltip": "Export each virtual view as PLY/GLB to inspect per-view geometry and coverage."}),
                "per_view_export_format": (["ply", "glb", "both"], {"default": "ply", "tooltip": "Per-view export format for debugging and QA."}),
                "per_view_prefix": ("STRING", {"default": "3D/MoGe_Pano_Views", "tooltip": "Output prefix/path for per-view exports under ComfyUI’s output directory."}),
                "output_pcl": ("BOOLEAN", {"default": True, "tooltip": "Export merged world-space point cloud (.ply) with panorama colors."}),
                "output_glb": ("BOOLEAN", {"default": False, "tooltip": "Export textured GLB over the panorama grid. Requires utils3d image_mesh; otherwise skipped."}),
                "glb_rotate_x_deg": ("INT", {"default": 90, "min": -180, "max": 180, "step": 15, "tooltip": "Extra clockwise rotation around X (red) axis for GLB export to match viewer conventions."}),
                "filename_prefix": ("STRING", {"default": "3D/MoGe_Pano", "tooltip": "Output prefix under ComfyUI’s output directory for PLY/GLB."}),
                "use_fp16": ("BOOLEAN", {"default": True, "tooltip": "Use FP16 inference to reduce VRAM and improve speed."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("depth", "normal", "pcl_path", "glb_path")
    FUNCTION = "process"
    CATEGORY = "MoGe2"
    OUTPUT_NODE = True
    DESCRIPTION = "MoGe-2 panorama inference with metric-preserving merge (weighted default)."

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

    def process(self,
                model: str,
                model_path: str,
                image,
                face_resolution: int,
                resolution_level: str,
                view_fov_x_deg: int,
                merge_method: str,
                angle_power: float,
                depth_alpha: float,
                fill_holes: bool,
                hole_iters: int,
                apply_mask: bool,
                horizontal_wrap: bool,
                export_per_view: bool,
                per_view_export_format: str,
                per_view_prefix: str,
                output_pcl: bool,
                output_glb: bool,
                glb_rotate_x_deg: int,
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
        extrinsics, intrinsics_list = get_panorama_cameras(fov_x_deg=float(view_fov_x_deg))

        # Split panorama into view images
        view_images = split_panorama_image(pano, extrinsics, intrinsics_list, resolution=face_resolution)

        # Run inference per view
        res_level_int = self._get_resolution_level_int(resolution_level)
        per_points: List[np.ndarray] = []
        per_normals: List[Optional[np.ndarray]] = []
        per_masks: List[np.ndarray] = []
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

            # Apply mask to invalidate
            if apply_mask:
                invalid = ~mask
                points[invalid] = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
                if normal is not None:
                    normal[invalid] = 0.0

            per_points.append(points)
            per_normals.append(normal)
            per_masks.append(mask)

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
        if merge_method == 'z_buffer':
            best_dist = np.full((pano_h, pano_w), np.inf, dtype=np.float32)
            best_point_world = np.full((pano_h, pano_w, 3), np.nan, dtype=np.float32)
            best_normal_world = np.full((pano_h, pano_w, 3), 0, dtype=np.float32)
            final_mask = np.zeros((pano_h, pano_w), dtype=bool)
        else:
            accum_w = np.zeros((pano_h, pano_w), dtype=np.float32)
            accum_pts = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
            accum_norm = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
            final_mask = np.zeros((pano_h, pano_w), dtype=bool)

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
            dist = np.linalg.norm(pts_world, axis=-1)

            if merge_method == 'z_buffer':
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
                        n_cam = remap_normal.reshape(-1, 3)
                        n_world = (n_cam @ R)
                        n_world = n_world / (np.linalg.norm(n_world, axis=-1, keepdims=True) + 1e-8)
                        n_world = n_world.reshape(pano_h, pano_w, 3)
                        best_normal_world[better] = n_world[better]
            else:
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

        # Simple hole filling (optional)
        if fill_holes:
            best_point_world, best_normal_world, final_mask = self._fill_holes_average(best_point_world, best_normal_world, final_mask, iters=hole_iters)

        # Build visualization images
        if np.any(final_mask):
            if merge_method == 'weighted':
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
        if output_glb and np.any(final_mask):
            try:
                # Build a triangulated image mesh using the pano grid
                uv_grid = self._image_uv(width=pano_w, height=pano_h)
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
