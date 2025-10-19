import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import logging
import uuid
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
from typing import Any, Dict, Tuple, List, Optional, Union
import folder_paths


def _ensure_numpy_uint8_image(image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert a ComfyUI IMAGE (torch or numpy) to uint8 numpy HxWx3."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.ndim == 4:
        # Expect batch dimension first; take first element
        image = image[0]
    if image.dtype == np.float32 or image.dtype == np.float16:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).round().astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def _numpy_uint8_to_torch_image(np_img: np.ndarray) -> torch.Tensor:
    """Convert uint8 numpy image (HxWx3 or NxHxWx3) to ComfyUI IMAGE tensor."""
    arr = np_img
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr[None, ...]
    return torch.from_numpy(arr)


def _numpy_uint8_batch_to_tensor(batch: np.ndarray) -> torch.Tensor:
    """Convert uint8 numpy batch (NxHxWx3) to ComfyUI IMAGE tensor."""
    arr = batch
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _comfy_image_batch_to_numpy(image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert ComfyUI IMAGE batch to float32 numpy (NxHxWxC)."""
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = image
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def _create_view_bundle(
    pano_shape: Tuple[int, int],
    view_images_uint8: np.ndarray,
    extrinsics: List[np.ndarray],
    intrinsics: List[np.ndarray],
    face_resolution: int,
    view_fov_x_deg: float,
    horizontal_wrap: bool,
) -> Dict[str, Any]:
    """Package panorama split metadata for reuse across nodes."""
    bundle_id = uuid.uuid4().hex
    return {
        "bundle_id": bundle_id,
        "pano_shape": tuple(int(x) for x in pano_shape),
        "images_uint8": view_images_uint8,
        "extrinsics": [np.asarray(ex, dtype=np.float32) for ex in extrinsics],
        "intrinsics": [np.asarray(ik, dtype=np.float32) for ik in intrinsics],
        "face_resolution": int(face_resolution),
        "view_fov_x_deg": float(view_fov_x_deg),
        "horizontal_wrap": bool(horizontal_wrap),
    }


def _validate_view_bundle(bundle: Dict[str, Any]) -> None:
    required_keys = [
        "bundle_id",
        "pano_shape",
        "images_uint8",
        "extrinsics",
        "intrinsics",
        "face_resolution",
        "view_fov_x_deg",
        "horizontal_wrap",
    ]
    for k in required_keys:
        if k not in bundle:
            raise ValueError(f"View bundle missing required key '{k}'.")
    images = bundle["images_uint8"]
    if not isinstance(images, np.ndarray) or images.ndim != 4:
        raise ValueError("View bundle expects 'images_uint8' as numpy array (N,H,W,3).")
    if images.shape[-1] != 3:
        raise ValueError("View images must have 3 channels.")
    if images.dtype != np.uint8:
        raise ValueError("View images must be uint8.")
    num_views = images.shape[0]
    if len(bundle["extrinsics"]) != num_views or len(bundle["intrinsics"]) != num_views:
        raise ValueError("Mismatch between number of views and intrinsics/extrinsics in view bundle.")


def _create_normals_package(
    view_bundle: Dict[str, Any],
    normals_cam: np.ndarray,
    source: str,
    encoding: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bundle external per-view normals with provenance info."""
    if meta is None:
        meta = {}
    return {
        "bundle_id": view_bundle["bundle_id"],
        "view_package": view_bundle,
        "normals_cam": normals_cam.astype(np.float32),
        "source": source,
        "encoding": encoding,
        "meta": meta,
    }


def _validate_normals_package(normals_pkg: Dict[str, Any]) -> None:
    for key in ("bundle_id", "view_package", "normals_cam"):
        if key not in normals_pkg:
            raise ValueError(f"Normals package missing required key '{key}'.")
    view_bundle = normals_pkg["view_package"]
    _validate_view_bundle(view_bundle)
    normals = normals_pkg["normals_cam"]
    if not isinstance(normals, np.ndarray) or normals.ndim != 4 or normals.shape[-1] != 3:
        raise ValueError("Normals package expects 'normals_cam' as numpy array (N,H,W,3).")
    if normals.dtype not in (np.float32, np.float64):
        raise ValueError("Normals array must be float32/float64.")
    num_views = normals.shape[0]
    if normals.shape[1:3] != view_bundle["images_uint8"].shape[1:3]:
        raise ValueError("Normals resolution does not match view bundle resolution.")
    if num_views != view_bundle["images_uint8"].shape[0]:
        raise ValueError("Number of normal views does not match view bundle.")

script_directory = os.path.dirname(os.path.abspath(__file__))

log = logging.getLogger(__name__)

class RunMoGe2Process:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["v1","v2"], {"default": "v2", "tooltip": "Checkpoint choice. v2 (default) outputs normals + metric point maps; drop to v1 only when you need a lighter model and can live without normals."}),
                "image": ("IMAGE", {"tooltip": "Perspective RGB frame. Resize upstream if you need a fixed resolution; otherwise the node enforces max_size for you."}),
                "max_size": ("INT", {"default": 800, "min": 100, "max": 1000, "step": 100, "tooltip": "Largest allowed image edge before inference. Raise for more detail (more VRAM/time), lower for safety on cramped GPUs."}),
                "resolution_level": (["Low", "Medium", "High", "Ultra"], {"default": "High", "tooltip": "Internal MoGe budget. Low/Medium for previews, High for production, Ultra for the sharpest geometry if VRAM allows."}),
                "remove_edge": ("BOOLEAN", {"default": True, "tooltip": "Cull values along steep depth jumps to avoid jagged mesh edges. Turn OFF to preserve thin structures even if edges get noisy."}),
                "apply_mask": ("BOOLEAN", {"default": True, "tooltip": "Respect the validity mask predicted by the model. Disable only if you plan to handle invalid pixels yourself."}),
                "output_glb": ("BOOLEAN", {"default": True, "tooltip": "Write a textured GLB mesh to disk. Disable when you only care about depth/normal imagery or want faster iterations."}),
                "filename_prefix": ("STRING", {"default": "3D/MoGe", "tooltip": "Prefix under ComfyUI’s output directory for saved meshes and preview renders."}),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "IMAGE","STRING")
    RETURN_NAMES = ("depth", "normal", "glb_path")
    FUNCTION = "process"
    CATEGORY = "MoGe2-360"
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


class MoGe2PanoramaSplit:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Equirectangular panorama to slice into perspective views. Use the same image you will feed into MoGe2Panorama."}),
                "face_resolution": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 64, "tooltip": "Per-view render size. Match the value you plan to use in MoGe2Panorama; increase for fewer seams at the cost of VRAM and runtime."}),
                "view_fov_x_deg": ("INT", {"default": 110, "min": 60, "max": 120, "step": 5, "tooltip": "Horizontal FOV for each virtual camera. 105-115 gives healthy overlap; use lower values only when mimicking cube faces."}),
                "horizontal_wrap": ("BOOLEAN", {"default": True, "tooltip": "Record whether the panorama should wrap horizontally. Keep ON for standard equirectangular images; toggle OFF only for debugging non-wrapping inputs."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "PANORAMA_VIEWS")
    RETURN_NAMES = ("view_images", "view_package")
    FUNCTION = "process"
    CATEGORY = "MoGe2-360"

    def process(self, image, face_resolution: int, view_fov_x_deg: int, horizontal_wrap: bool):
        pano = _ensure_numpy_uint8_image(image)
        pano_h, pano_w = pano.shape[:2]
        extrinsics, intrinsics_list = get_panorama_cameras(fov_x_deg=float(view_fov_x_deg))
        view_images_list = split_panorama_image(pano, extrinsics, intrinsics_list, resolution=face_resolution)
        view_images_uint8 = np.stack(view_images_list, axis=0)
        view_tensor = _numpy_uint8_batch_to_tensor(view_images_uint8)
        view_bundle = _create_view_bundle(
            pano_shape=(pano_h, pano_w),
            view_images_uint8=view_images_uint8,
            extrinsics=extrinsics,
            intrinsics=intrinsics_list,
            face_resolution=face_resolution,
            view_fov_x_deg=float(view_fov_x_deg),
            horizontal_wrap=horizontal_wrap,
        )
        return (view_tensor, view_bundle)


class MoGe2PanoramaNormalsPack:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "view_package": ("PANORAMA_VIEWS", {"tooltip": "View bundle from MoGe2PanoramaSplit or MoGe2Panorama. Supplies the exact slice order, intrinsics, and metadata your normals must match."}),
                "normal_images": ("IMAGE", {"tooltip": "Batch of per-view normal maps aligned with the bundle (N x H x W x 3). Feed the raw output from your external estimator here."}),
                "encoding": (["unit_interval", "signed"], {"default": "unit_interval", "tooltip": "Select how the incoming normals are encoded: unit_interval for [0,1] images (common in ControlNet preprocessors), signed for [-1,1] tensors (MiDaS, BAE)."}),
                "normalize": ("BOOLEAN", {"default": True, "tooltip": "Re-normalize vectors to unit length after conversion. Leave ON unless your source already provides exact unit normals and you need to preserve magnitude tweaks."}),
                "flip_y": ("BOOLEAN", {"default": False, "tooltip": "Flip the Y component. Enable when your provider uses a Y-up convention (e.g., MiDaS/BAE) so normals line up with MoGe’s camera frame."}),
                "flip_z": ("BOOLEAN", {"default": False, "tooltip": "Flip the Z component to switch between inward/outward facing normals. Useful when your tool defines forward in the opposite direction."}),
            },
            "optional": {
                "source_name": ("STRING", {"default": "external", "tooltip": "Label describing the origin of these normals (logged in exports so you can trace their provenance)."}),
            },
        }

    RETURN_TYPES = ("PANORAMA_NORMALS",)
    RETURN_NAMES = ("external_normals",)
    FUNCTION = "process"
    CATEGORY = "MoGe2-360"

    def process(
        self,
        view_package: Dict[str, Any],
        normal_images,
        encoding: str,
        normalize: bool,
        flip_y: bool,
        flip_z: bool,
        source_name: str = "external",
    ) -> Tuple[Dict[str, Any]]:
        _validate_view_bundle(view_package)
        normals_np = _comfy_image_batch_to_numpy(normal_images)
        if normals_np.shape[-1] != 3:
            raise ValueError("Normal tensor must have 3 channels.")
        num_views = view_package["images_uint8"].shape[0]
        if normals_np.shape[0] != num_views:
            raise ValueError(f"Normals batch has {normals_np.shape[0]} views but view package expects {num_views}.")
        face_h, face_w = view_package["images_uint8"].shape[1:3]
        if normals_np.shape[1] != face_h or normals_np.shape[2] != face_w:
            raise ValueError("Normal map resolution must match the face resolution of the split views.")

        normals = normals_np.astype(np.float32)
        if encoding == "unit_interval":
            normals = (normals * 2.0) - 1.0
        elif encoding == "signed":
            normals = np.clip(normals, -1.0, 1.0)
        else:
            raise ValueError(f"Unsupported encoding '{encoding}'.")

        if flip_y:
            normals[..., 1] *= -1.0
        if flip_z:
            normals[..., 2] *= -1.0
        if normalize:
            norms = np.linalg.norm(normals, axis=-1, keepdims=True)
            normals = normals / (norms + 1e-8)

        normals_pkg = _create_normals_package(
            view_bundle=view_package,
            normals_cam=normals,
            source=source_name or "external",
            encoding=encoding,
            meta={
                "normalize": normalize,
                "flip_y": flip_y,
                "flip_z": flip_z,
            },
        )
        return (normals_pkg,)

class MoGe2Panorama:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["v1", "v2"], {"default": "v2", "tooltip": "MoGe checkpoint to load. v2 is recommended (normals + metric scale); switch to v1 only when VRAM is tight and you do not need normals."}),
                "model_path": ("STRING", {"default": "C:/models/Ruicheng/moge-2-vitl-normal", "tooltip": "Local folder that contains the chosen weights. Override when your checkpoints live elsewhere. The node never downloads models."}),
                "image": ("IMAGE", {"tooltip": "Input equirectangular panorama (HxW). The merged outputs keep this exact resolution."}),
                "face_resolution": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 64, "tooltip": "Per-view render size. Raise (1280-1536) for cleaner seams/fine detail when VRAM allows; lower for faster drafts or low-memory GPUs."}),
                "resolution_level": (["Low", "Medium", "High", "Ultra"], {"default": "Ultra", "tooltip": "MoGe inference granularity. Low/Medium for quick previews, High for production, Ultra for the sharpest geometry (heaviest VRAM)."}),
                "view_fov_x_deg": ("INT", {"default": 110, "min": 60, "max": 120, "step": 5, "tooltip": "Horizontal FOV per virtual camera. Use 105-115 for dependable overlap; drop toward 90 only when you want cube-face coverage; push to 115-118 if you still see seam gaps."}),
                "merge_method": (["z_buffer", "weighted", "affine_depth", "poisson_depth"], {"default": "z_buffer", "tooltip": "How to stitch per-view geometry. z_buffer keeps the first physical hit (best metric fidelity). Weighted/affine/poisson blend slices and are useful only after coverage is clean when you want softer seams."}),
                "normal_mode": (["auto", "internal_only", "external_only", "external_fallback"], {"default": "auto", "tooltip": "Choose where normals come from. auto prefers external normals if connected, internal_only ignores them, external_only requires them, external_fallback uses external normals per view but falls back to MoGe when a slice is missing."}),
                "zbuffer_mode": (["ray", "radial"], {"default": "ray", "tooltip": "Distance metric for z-buffer. ray compares signed distance along each panorama ray (best for seams); radial uses ||P|| and can help when your per-view origins drift."}),
                "angle_power": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.5, "tooltip": "Only used by weighted/affine/poisson. Higher values (2-4) prioritize view centers; set 0 to disable angle weighting when experimenting."}),
                "depth_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Only used by weighted/affine/poisson. >0 down-weights far geometry (start around 0.3) to reduce ghosting across overlaps."}),
                "fill_holes": ("BOOLEAN", {"default": True, "tooltip": "After merging, fill isolated invalid pixels with neighbor averages. Leave ON for watertight meshes; disable if you prefer the raw mask."}),
                "hole_iters": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1, "tooltip": "Number of hole-fill passes. 1-2 keeps detail; values >2 start to blur thin structures."}),
                "apply_mask": ("BOOLEAN", {"default": True, "tooltip": "Honor MoGe’s per-view validity mask. Disable only for debugging, otherwise invalid pixels will pollute the merge."}),
                "horizontal_wrap": ("BOOLEAN", {"default": True, "tooltip": "Wrap the pano seam when remapping. Switch OFF only when diagnosing seam issues or processing a non-wrapping projection."}),
                "skip_small_masks": ("BOOLEAN", {"default": True, "tooltip": "Reject slices whose valid region is tiny (often blank sky). Disable only when most views are genuinely sparse and you still want them merged."}),
                "min_mask_ratio": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.1, "step": 0.001, "tooltip": "Minimum valid-pixel fraction for a slice to survive filtering. Use 0.003-0.005 for indoor scenes; drop toward 0.001 for outdoor skies."}),
                "auto_relax_min_mask": ("BOOLEAN", {"default": True, "tooltip": "If filtering leaves fewer than min_valid_views, automatically re-enable skipped slices (useful when big sky regions blow out the mask). Turn OFF for deterministic filtering."}),
                "min_valid_views": ("INT", {"default": 14, "min": 1, "max": 30, "step": 1, "tooltip": "Minimum number of slices to keep after masking. Raise this when you upscale face_resolution/FOV so the auto-relax logic has stricter coverage goals."}),
                "wrap_consistency": ("BOOLEAN", {"default": True, "tooltip": "Average the first/last panorama columns when both are valid to enforce seam continuity. Disable if you need to inspect the raw seam difference."}),
                "align_in_disparity": ("BOOLEAN", {"default": True, "tooltip": "affine_depth only. Align slices in disparity (1/depth) which maintains scale better. Turn OFF to experiment with plain depth alignment."}),
                "depth_alignment": ("BOOLEAN", {"default": True, "tooltip": "Align each virtual view's depth scale to the accumulated panorama using a robust scale fit. Disable only if alignment causes visible jitter."}),
                "alignment_quantile": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.45, "step": 0.01, "tooltip": "Quantile range for trimming ratio outliers when estimating alignment scale (lower = focus on closer structures)."}),
                "alignment_max_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5, "tooltip": "Clamp on the scale factor applied during alignment to avoid extreme rescaling. Applied symmetrically as 1/scale."}),
                "compress_depth": ("BOOLEAN", {"default": True, "tooltip": "Clamp extreme merged distances before meshing/exports to stabilise geometry."}),
                "depth_compression_quantile": ("FLOAT", {"default": 0.97, "min": 0.8, "max": 0.999, "step": 0.01, "tooltip": "Upper quantile used when compressing merged distances (higher = keep more range, lower = tighter clamp)."}),
                "smooth_mesh_mask": ("BOOLEAN", {"default": True, "tooltip": "Morphologically smooth the final valid mask before meshing to suppress pixel-scale spikes/holes."}),
                "mesh_mask_kernel": ("INT", {"default": 3, "min": 1, "max": 9, "step": 1, "tooltip": "Kernel size (pixels) used for mask smoothing when smooth_mesh_mask is enabled."}),
                "polar_smooth": ("BOOLEAN", {"default": True, "tooltip": "Apply a mild blur near the zenith/nadir to calm equirectangular stretching. Disable if your poles contain important high-frequency detail."}),
                "polar_cap_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.2, "step": 0.005, "tooltip": "Fraction of image height considered a polar cap for smoothing. 0.08-0.12 works well for most HDRIs; lower values keep more detail."}),
                "polar_blur_ks": ("INT", {"default": 9, "min": 3, "max": 31, "step": 2, "tooltip": "Odd Gaussian kernel size used for polar smoothing. 7 or 9 is usually enough; larger kernels smooth more but risk softening geometry."}),
                "export_per_view": ("BOOLEAN", {"default": False, "tooltip": "Write each virtual view as a debug export. Enable when diagnosing coverage or normals; disable for production runs to save time and disk."}),
                "per_view_export_format": (["ply", "glb", "both"], {"default": "ply", "tooltip": "Format for per-view exports. PLY gives lightweight point clouds; GLB builds textured meshes; both writes both (slow)."}),
                "per_view_prefix": ("STRING", {"default": "3D/MoGe_Pano_Views", "tooltip": "Output prefix for per-view debug exports inside the ComfyUI output directory."}),
                "output_pcl": ("BOOLEAN", {"default": True, "tooltip": "Export the merged world-space point cloud (.ply). Disable if you only need the mesh or are trying to save disk space."}),
                "output_glb": ("BOOLEAN", {"default": True, "tooltip": "Export the merged textured GLB. Turn OFF when you want geometry-free depth/normal outputs or are running analysis-only jobs."}),
                "mesh_wrap_x": ("BOOLEAN", {"default": True, "tooltip": "Connect the panorama seam when generating the mesh so GLB viewers do not show a crack. Disable if you need an open seam for downstream editing."}),
                "glb_rotate_x_deg": ("INT", {"default": 90, "min": -180, "max": 180, "step": 15, "tooltip": "Extra rotation applied to the exported GLB. +90 tilts the mesh so the floor faces downward for Y-up viewers."}),
                "filename_prefix": ("STRING", {"default": "3D/MoGe_Pano", "tooltip": "Base path used for the fused PLY/GLB exports within ComfyUI outputs."}),
                "export_depth": ("BOOLEAN", {"default": False, "tooltip": "Save the metric depth map to disk. Enable when you need depth textures for other tools."}),
                "depth_format": (["png16", "exr", "both"], {"default": "png16", "tooltip": "Choose the on-disk depth format: png16 writes millimeters as 16-bit, exr writes meters as float32, both writes both versions."}),
                "depth_prefix": ("STRING", {"default": "3D/MoGe_Pano_Depth", "tooltip": "Output prefix for saved depth files."}),
                "use_fp16": ("BOOLEAN", {"default": True, "tooltip": "Run inference in mixed precision. Keep enabled for speed unless you suspect numerical issues on specific hardware."}),
                "denoise_spikes": ("BOOLEAN", {"default": True, "tooltip": "Detect and drop extreme ray-distance outliers before meshing so the GLB does not sprout spikes. Disable only if thin geometry is being mistaken for spikes."}),
                "spike_sigma": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 6.0, "step": 0.5, "tooltip": "Threshold (in robust sigma units) used by spike removal. Increase to keep more data; decrease to prune aggressive outliers."}),
            },
            "optional": {
                "mask_image": ("IMAGE", {"tooltip": "Optional label map with the same panorama resolution. Use it to drive per-label exports or to inspect coverage."}),
                "multi_glb_from_mask": ("BOOLEAN", {"default": False, "tooltip": "When true, create one GLB per mask label. Requires mask_image and is useful for separating furniture vs structure."}),
                "mask_ignore_zero": ("BOOLEAN", {"default": True, "tooltip": "Skip label 0 when generating per-label GLBs. Disable if label 0 should also export."}),
                "min_label_area_ratio": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.5, "step": 0.001, "tooltip": "Smallest area fraction a label must occupy to be exported. Raise to avoid tiny noisy parts."}),
                "multi_glb_prefix": ("STRING", {"default": "3D/MoGe_Pano_Label", "tooltip": "Prefix for per-label GLB exports."}),
                "view_package": ("PANORAMA_VIEWS", {"tooltip": "Supply a precomputed view bundle (from MoGe2PanoramaSplit or a previous run) to reuse the same slicing metadata, especially when sharing data with external normal estimators."}),
                "external_normals": ("PANORAMA_NORMALS", {"tooltip": "Optional per-view normals bundle (camera frame) from MoGe2PanoramaNormalsPack or your own node. Pair with normal_mode auto/external_* to use it."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("depth", "normal", "pcl_path", "glb_path", "depth_file")
    FUNCTION = "process"
    CATEGORY = "MoGe2-360"
    OUTPUT_NODE = True
    DESCRIPTION = "MoGe-2 panorama inference with metric-preserving merge (z-buffer default)."

    @staticmethod
    def _to_numpy_image(image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        return _ensure_numpy_uint8_image(image)

    @staticmethod
    def _numpy_to_tensor_image(img_np: np.ndarray) -> torch.Tensor:
        return _numpy_uint8_to_torch_image(img_np)

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

    def _align_view_depth(self,
                          pts_world: np.ndarray,
                          dist: np.ndarray,
                          valid_mask: np.ndarray,
                          reference_dist: np.ndarray,
                          reference_mask: np.ndarray,
                          quantile: float,
                          max_scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """Align a view's world-space distances to the accumulated reference using a robust scale fit."""
        if reference_dist is None or reference_mask is None:
            return pts_world, dist
        overlap = reference_mask & valid_mask
        if not np.any(overlap):
            return pts_world, dist
        ref_vals = reference_dist[overlap]
        cand_vals = dist[overlap]
        positive = (ref_vals > 1e-6) & (cand_vals > 1e-6) & np.isfinite(ref_vals) & np.isfinite(cand_vals)
        if not np.any(positive):
            return pts_world, dist
        ratios = ref_vals[positive] / cand_vals[positive]
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        if ratios.size == 0:
            return pts_world, dist
        try:
            lo = float(np.quantile(ratios, quantile))
            hi = float(np.quantile(ratios, 1.0 - quantile))
            trim = ratios[(ratios >= lo) & (ratios <= hi)]
            scale = float(np.median(trim)) if trim.size > 0 else float(np.median(ratios))
        except Exception:
            scale = float(np.median(ratios))
        if not np.isfinite(scale):
            return pts_world, dist
        scale = float(np.clip(scale, 1.0 / max_scale, max_scale))
        if abs(scale - 1.0) < 1e-4:
            return pts_world, dist
        return pts_world * scale, dist * scale

    def _compress_world_depth(self,
                              points_world: np.ndarray,
                              mask: np.ndarray,
                              quantile: float) -> Tuple[np.ndarray, np.ndarray]:
        """Clamp extreme distances to improve mesh stability."""
        distances = np.linalg.norm(points_world, axis=-1)
        valid = mask & np.isfinite(distances) & (distances > 0)
        if not np.any(valid):
            return points_world, distances
        try:
            cutoff = float(np.quantile(distances[valid], quantile))
        except Exception:
            cutoff = float(distances[valid].max())
        if not np.isfinite(cutoff) or cutoff <= 0:
            return points_world, distances
        scale = np.ones_like(distances, dtype=np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale[valid] = np.minimum(distances[valid], cutoff) / (distances[valid] + 1e-8)
        compressed = points_world.copy()
        compressed[valid] *= scale[valid][..., None]
        clipped_dist = distances.copy()
        clipped_dist[valid] = np.minimum(distances[valid], cutoff)
        return compressed, clipped_dist

    def _smooth_binary_mask(self, mask: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply a morphology close/open pair to reduce small holes and spikes in the mask."""
        if kernel_size <= 1:
            return mask
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_u8 = mask.astype(np.uint8)
        closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        return opened.astype(bool)

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

    def _cleanup_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Remove basic degeneracies so exports do not contain needle spikes."""
        try:
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
        except Exception as exc:
            log.debug("Mesh cleanup skipped: %s", exc)
        return mesh

    def process(self,
                model: str,
                model_path: str,
                image,
                face_resolution: int,
                resolution_level: str,
                view_fov_x_deg: int,
                merge_method: str,
                normal_mode: str,
                zbuffer_mode: str,
                angle_power: float,
                depth_alpha: float,
                fill_holes: bool,
                hole_iters: int,
                apply_mask: bool,
                horizontal_wrap: bool,
                skip_small_masks: bool,
                min_mask_ratio: float,
                auto_relax_min_mask: bool,
                min_valid_views: int,
                wrap_consistency: bool,
                align_in_disparity: bool,
                depth_alignment: bool,
                alignment_quantile: float,
                alignment_max_scale: float,
                compress_depth: bool,
                depth_compression_quantile: float,
                smooth_mesh_mask: bool,
                mesh_mask_kernel: int,
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
                denoise_spikes: bool,
                spike_sigma: float,
                view_package: Optional[Dict[str, Any]] = None,
                external_normals: Optional[Dict[str, Any]] = None,
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

        # Prepare panorama image
        pano = self._to_numpy_image(image)
        pano_h, pano_w = pano.shape[:2]

        # Resolve optional precomputed view bundle / external normals
        view_bundle = None
        normals_pkg = None
        if view_package is not None:
            _validate_view_bundle(view_package)
            view_bundle = view_package
        if external_normals is not None:
            _validate_normals_package(external_normals)
            normals_pkg = external_normals
            bundle_from_normals = normals_pkg["view_package"]
            if view_bundle is None:
                view_bundle = bundle_from_normals
            elif view_bundle["bundle_id"] != bundle_from_normals["bundle_id"]:
                raise ValueError("External normals bundle does not match supplied view package (bundle_id mismatch).")

        horizontal_wrap_effective = horizontal_wrap
        effective_view_fov_x_deg = float(view_fov_x_deg)
        alignment_quantile = float(np.clip(alignment_quantile, 0.0, 0.49))
        alignment_max_scale = max(1.0, float(alignment_max_scale))
        depth_compression_quantile = float(np.clip(depth_compression_quantile, 0.0, 0.999))

        if view_bundle is not None:
            if tuple(view_bundle["pano_shape"]) != (pano_h, pano_w):
                log.warning("View package pano shape %s differs from input panorama %s. Results may be misaligned.", view_bundle["pano_shape"], (pano_h, pano_w))
            extrinsics = view_bundle["extrinsics"]
            intrinsics_list = view_bundle["intrinsics"]
            view_images_uint8 = view_bundle["images_uint8"]
            num_views_total = view_images_uint8.shape[0]
            view_images = [view_images_uint8[idx] for idx in range(num_views_total)]
            face_h = int(view_images_uint8.shape[1])
            face_w = int(view_images_uint8.shape[2])
            effective_view_fov_x_deg = float(view_bundle.get("view_fov_x_deg", effective_view_fov_x_deg))
            horizontal_wrap_effective = bool(view_bundle.get("horizontal_wrap", horizontal_wrap_effective))
        else:
            # Build virtual cameras and split panorama
            extrinsics, intrinsics_list = get_panorama_cameras(fov_x_deg=float(view_fov_x_deg))
            view_images = split_panorama_image(pano, extrinsics, intrinsics_list, resolution=face_resolution)
            num_views_total = len(view_images)
            if num_views_total == 0:
                raise RuntimeError("Panorama split produced zero views.")
            face_h = int(view_images[0].shape[0])
            face_w = int(view_images[0].shape[1])
            view_images = [img.astype(np.uint8) if img.dtype != np.uint8 else img for img in view_images]

        face_area = face_h * face_w

        normals_override = normals_pkg["normals_cam"] if normals_pkg is not None else None
        normal_mode_lower = (normal_mode or "auto").lower()
        allowed_modes = {"auto", "internal_only", "external_only", "external_fallback"}
        if normal_mode_lower not in allowed_modes:
            log.warning("Unknown normal_mode '%s', defaulting to 'auto'.", normal_mode)
            normal_mode_lower = "auto"
        if normal_mode_lower == "external_only" and normals_override is None:
            raise ValueError("normal_mode=external_only requires an external_normals input.")

        # Run inference per view
        res_level_int = self._get_resolution_level_int(resolution_level)
        per_points: List[Optional[np.ndarray]] = [None] * num_views_total
        per_normals: List[Optional[np.ndarray]] = [None] * num_views_total
        per_masks: List[Optional[np.ndarray]] = [None] * num_views_total
        # For Poisson depth fusion (view-domain inputs)
        view_distance_maps: List[Optional[np.ndarray]] = [None] * num_views_total
        view_pred_masks: List[Optional[np.ndarray]] = [None] * num_views_total
        skipped_due_to_mask: List[Dict[str, Any]] = []
        for vi, img in enumerate(view_images):
            img_np = img.astype(np.uint8) if img.dtype != np.uint8 else img
            img_t = torch.tensor(img_np, dtype=torch.float16 if use_fp16 else torch.float32, device=torch.device('cuda')).permute(2, 0, 1) / 255.0
            out = model_instance.infer(img_t, resolution_level=res_level_int, apply_mask=apply_mask, fov_x=float(effective_view_fov_x_deg), use_fp16=use_fp16)
            out = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}

            points = out.get('points', None)
            depth = out.get('depth', None)
            mask = out.get('mask', None)
            base_normal = out.get('normal', None)
            external_normal = None
            if normals_override is not None:
                external_normal = normals_override[vi]

            normal = None
            if normal_mode_lower == "internal_only":
                normal = base_normal
            elif normal_mode_lower == "external_only":
                if external_normal is None:
                    raise ValueError(f"External normals missing for view {vi} while normal_mode=external_only.")
                normal = external_normal
            else:  # auto or external_fallback
                if external_normal is not None:
                    normal = external_normal
                else:
                    normal = base_normal

            # Ensure shapes
            if points is None or depth is None or mask is None:
                # view not usable
                per_points[vi] = np.full((face_h, face_w, 3), np.inf, dtype=np.float32)
                per_normals[vi] = None
                per_masks[vi] = np.zeros((face_h, face_w), dtype=bool)
                view_distance_maps[vi] = None
                view_pred_masks[vi] = None
                continue

            # Convert types
            points = points.astype(np.float32)
            mask = mask.astype(bool)
            if normal is not None:
                normal = np.array(normal, dtype=np.float32, copy=True)

            # Prepare distance map before mutating points
            dist_view = np.linalg.norm(points, axis=-1).astype(np.float32)
            if apply_mask:
                dist_view[~mask] = 1.0

            # Apply mask to invalidate
            if apply_mask:
                invalid = ~mask
                points[invalid] = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
                if normal is not None:
                    normal[invalid] = 0.0

            # Optionally skip views with too few valid pixels
            if apply_mask and skip_small_masks:
                valid_px = int(mask.sum())
                min_px = int(face_area * float(min_mask_ratio))
                if valid_px < max(16, min_px):
                    if auto_relax_min_mask:
                        skipped_due_to_mask.append({
                            "index": vi,
                            "points": points,
                            "normal": normal,
                            "mask": mask,
                            "dist": dist_view,
                        })
                    per_points[vi] = None
                    per_normals[vi] = None
                    per_masks[vi] = None
                    view_distance_maps[vi] = None
                    view_pred_masks[vi] = None
                    continue

            per_points[vi] = points
            per_normals[vi] = normal
            per_masks[vi] = mask
            view_distance_maps[vi] = dist_view
            view_pred_masks[vi] = mask.astype(bool)

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
                        cols = img_np[m]
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
                                img_np.astype(np.float32) / 255.0,
                                uv_grid_view,
                                n_world_view.astype(np.float32),
                                mask=m,
                                tri=True,
                            )
                        else:
                            faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                                pts_world_view,
                                img_np.astype(np.float32) / 255.0,
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
                        mesh = self._cleanup_mesh(mesh)
                        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(per_view_prefix, folder_paths.get_output_directory())
                        out_path = Path(full_output_folder) / f"{filename}_{counter:05}_view{vi:02}.glb"
                        out_path.parent.mkdir(exist_ok=True, parents=True)
                        mesh.export(out_path)
                except Exception as e:
                    log.exception("Per-view export failed for view %d: %s", vi, e)

        # If mask filtering was too aggressive, optionally reinstate skipped views
        valid_view_indices = [i for i, pts in enumerate(per_points) if pts is not None]
        target_valid = int(np.clip(min_valid_views, 1, num_views_total))
        if auto_relax_min_mask and skip_small_masks and len(valid_view_indices) < target_valid:
            if skipped_due_to_mask:
                log.info(
                    "Auto-relaxing min_mask_ratio: only %d views survived (target %d). Reinstating %d skipped views.",
                    len(valid_view_indices),
                    target_valid,
                    len(skipped_due_to_mask),
                )
                for rec in skipped_due_to_mask:
                    idx = rec["index"]
                    per_points[idx] = rec["points"]
                    per_normals[idx] = rec["normal"]
                    per_masks[idx] = rec["mask"]
                    view_distance_maps[idx] = rec["dist"]
                    view_pred_masks[idx] = rec["mask"].astype(bool)
                valid_view_indices = [i for i, pts in enumerate(per_points) if pts is not None]
            else:
                log.info(
                    "Auto-relax requested but no views were captured in the skipped cache; continuing with %d views.",
                    len(valid_view_indices),
                )

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
            borderMode = cv2.BORDER_WRAP if horizontal_wrap_effective else cv2.BORDER_REPLICATE
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

            # Compute distance metric based on zbuffer mode
            if zbuffer_mode == 'ray':
                # Ray distance: signed distance along panorama ray direction
                # For valid panoramic points from virtual cameras at origin, this should be > 0
                dist = np.sum(pts_world * spherical_dirs, axis=-1)
                # Filter out points behind the origin or too close (likely remap artifacts)
                valid_distance = (dist > 1e-4) & np.isfinite(dist)
                valid_here = valid_here & valid_distance
            else:
                # Radial distance: Euclidean norm from origin
                dist = np.linalg.norm(pts_world, axis=-1)
                valid_distance = (dist > 1e-6) & np.isfinite(dist)
                valid_here = valid_here & valid_distance

            # Additional validation: reject points with extreme coordinates (likely interpolation artifacts)
            point_magnitude = np.linalg.norm(pts_world, axis=-1)
            valid_magnitude = (point_magnitude < 1e6) & (point_magnitude > 1e-6)
            valid_here = valid_here & valid_magnitude

            if not np.any(valid_here):
                continue

            # Depth alignment for z_buffer+ray mode can cause slice-to-slice drift
            # Only apply alignment for radial mode or non-zbuffer methods
            if depth_alignment and merge_method == 'z_buffer' and zbuffer_mode == 'radial' and np.any(final_mask):
                pts_world, dist = self._align_view_depth(
                    pts_world,
                    dist,
                    valid_here,
                    best_dist,
                    final_mask,
                    alignment_quantile,
                    alignment_max_scale,
                )

            if merge_method == 'z_buffer':
                # Z-buffer update: pick closest valid point per pixel
                if zbuffer_mode == 'ray':
                    # Ray mode: closest intersection along ray (smallest positive distance)
                    # Already filtered dist > 1e-4 above, so we just need dist < best_dist
                    better = (dist < best_dist) & valid_here
                else:
                    # Radial mode: closest to origin
                    better = (dist < best_dist) & valid_here

                if np.any(better):
                    best_dist[better] = dist[better]
                    best_point_world[better] = pts_world[better]
                    final_mask[better] = True

                    # normals
                    normal = per_normals[i]
                    if normal is not None:
                        remap_normal = self._remap_multi(normal, proj_pixels[..., 0], proj_pixels[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=borderMode)
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
                        remap_normal = self._remap_multi(normal, proj_pixels[..., 0], proj_pixels[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=borderMode)
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
                    remap_normal = self._remap_multi(normal, proj_pixels[..., 0], proj_pixels[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=borderMode)
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
            elif merge_method == 'poisson_depth':
                valid_idx = [idx for idx, d in enumerate(view_distance_maps) if d is not None and view_pred_masks[idx] is not None]
                if len(valid_idx) > 0:
                    try:
                        from .moge.utils.panorama import merge_panorama_depth as _merge_panorama_depth
                        pano_depth, final_mask = _merge_panorama_depth(
                            pano_w,
                            pano_h,
                            [view_distance_maps[idx] for idx in valid_idx],
                            [view_pred_masks[idx] for idx in valid_idx],
                            [extrinsics[idx] for idx in valid_idx],
                            [intrinsics_list[idx] for idx in valid_idx],
                        )
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
                else:
                    log.warning(
                        "Poisson depth merge skipped: no valid view distance maps after filtering (auto-relax may need adjustment)."
                    )

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

        # Optional depth compression before denoising to keep distances within a stable range
        if compress_depth and best_point_world is not None and np.any(final_mask):
            best_point_world, compressed_dist = self._compress_world_depth(best_point_world, final_mask, depth_compression_quantile)
            if merge_method == 'z_buffer':
                best_dist = np.where(final_mask, compressed_dist, np.inf).astype(np.float32)

        # Optional spike rejection before hole filling / meshing
        if denoise_spikes and best_point_world is not None and np.any(final_mask):
            dist_map = np.linalg.norm(best_point_world, axis=-1)
            masked = dist_map[final_mask]
            if masked.size > 0:
                median = float(np.median(masked))
                mad = float(np.median(np.abs(masked - median)))
                robust_sigma = 1.4826 * mad + 1e-6
                threshold = median + float(spike_sigma) * robust_sigma
                spike_mask = final_mask & (dist_map > threshold)
                if np.any(spike_mask):
                    removed = int(np.count_nonzero(spike_mask))
                    log.debug(
                        "Spike filter removed %d panorama samples (median=%.3f, sigma=%.3f, threshold=%.3f)",
                        removed,
                        median,
                        robust_sigma,
                        threshold,
                    )
                    best_point_world[spike_mask] = 0.0
                    if best_normal_world is not None:
                        best_normal_world[spike_mask] = 0.0
                    if merge_method == 'z_buffer':
                        best_dist[spike_mask] = np.inf
                    final_mask[spike_mask] = False

        # Simple hole filling (optional)
        if fill_holes:
            best_point_world, best_normal_world, final_mask = self._fill_holes_average(best_point_world, best_normal_world, final_mask, iters=hole_iters)

        if smooth_mesh_mask and np.any(final_mask):
            kernel = max(1, int(mesh_mask_kernel))
            new_mask = self._smooth_binary_mask(final_mask, kernel)
            removed = final_mask & ~new_mask
            if np.any(removed):
                best_point_world[removed] = 0.0
                if merge_method == 'z_buffer':
                    best_dist[removed] = np.inf
                if best_normal_world is not None:
                    best_normal_world[removed] = 0.0
            final_mask = new_mask

        # Recompute depth exports after all adjustments
        if merge_method == 'z_buffer':
            if zbuffer_mode == 'ray':
                ray_depth = np.sum(best_point_world * spherical_dirs, axis=-1)
                ray_depth = np.where(final_mask, np.maximum(ray_depth, 0.0), np.inf).astype(np.float32)
                best_dist = ray_depth
                export_depth_map = np.where(np.isfinite(ray_depth), np.maximum(ray_depth, 0.0), 0.0).astype(np.float32)
            else:
                radial = np.linalg.norm(best_point_world, axis=-1).astype(np.float32)
                radial = np.where(final_mask, radial, 0.0)
                export_depth_map = radial
        elif merge_method == 'weighted':
            export_depth_map = np.where(final_mask, np.maximum(np.sum(best_point_world * spherical_dirs, axis=-1), 0.0), 0.0).astype(np.float32)
        elif merge_method in ('affine_depth', 'poisson_depth'):
            export_depth_map = np.where(final_mask, np.linalg.norm(best_point_world, axis=-1), 0.0).astype(np.float32)
        else:
            export_depth_map = np.zeros((pano_h, pano_w), dtype=np.float32)

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
                mesh = self._cleanup_mesh(mesh)

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
                        mesh = self._cleanup_mesh(mesh)

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
    "MoGe2PanoramaSplit": MoGe2PanoramaSplit,
    "MoGe2PanoramaNormalsPack": MoGe2PanoramaNormalsPack,
    "MoGe2Panorama": MoGe2Panorama,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunMoGe2Process": "MoGe2 Process",
    "MoGe2PanoramaSplit": "MoGe2 Panorama Split",
    "MoGe2PanoramaNormalsPack": "MoGe2 Panorama Normals Pack",
    "MoGe2Panorama": "MoGe2 Panorama",
}
