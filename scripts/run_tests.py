"""
Lightweight test harness for ComfyUI-MoGe2-360.

Runs a set of import/sanity checks that do NOT require model weights:
- Validates panorama split/merge orientation with the synthetic unit-sphere test
- Exercises basic panorama utilities (camera gen, splitting)
- Imports nodes and inspects key parameters to ensure recent options are present

Usage:
  python scripts/run_tests.py

Exits with non-zero code on failure.
"""
import os
import sys
from pathlib import Path
import traceback
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_panorama_utils():
    from moge.utils.panorama import get_panorama_cameras, split_panorama_image, spherical_uv_to_directions
    # Build a simple gradient panorama
    H, W = 128, 256
    u = np.linspace(0, 1, W, dtype=np.float32)[None, :].repeat(H, 0)
    v = np.linspace(0, 1, H, dtype=np.float32)[:, None].repeat(W, 1)
    pano = np.stack([u, v, 0.5 * np.ones_like(u)], axis=-1)
    pano = (pano * 255).astype(np.uint8)

    extrinsics, intrinsics_list = get_panorama_cameras(fov_x_deg=90.0)
    views = split_panorama_image(pano, extrinsics, intrinsics_list, resolution=128)
    assert isinstance(views, list) and len(views) > 0, "split_panorama_image returned no views"
    for img in views:
        assert img.shape == (128, 128, 3), f"unexpected view shape {img.shape}"

    # Check directions grid
    uv = np.stack(np.meshgrid(np.linspace(0, 1, W, dtype=np.float32), np.linspace(0, 1, H, dtype=np.float32), indexing='xy'), axis=-1)
    dirs = spherical_uv_to_directions(uv)
    assert dirs.shape == (H, W, 3)
    nrm = np.linalg.norm(dirs, axis=-1)
    assert np.allclose(nrm, 1.0, atol=1e-4), "direction vectors should be unit length"


def test_validate_script():
    # Import and run validate_panorama.main()
    from scripts import validate_panorama as vp
    vp.main()
    # If it returns without exception, consider pass


def test_nodes_api():
    """Import nodes as a proper package submodule so relative imports work."""
    import importlib.util
    import importlib
    import types

    # Stub minimal ComfyUI folder_paths for import-time references
    if "folder_paths" not in sys.modules:
        fp = types.ModuleType("folder_paths")
        def _get_output_directory():
            out = REPO_ROOT / "outputs"
            out.mkdir(parents=True, exist_ok=True)
            return str(out)
        def _get_save_image_path(filename_prefix, output_dir, *args, **kwargs):
            from pathlib import Path
            prefix = filename_prefix if isinstance(filename_prefix, str) and filename_prefix else "ComfyUI"
            full = Path(output_dir) / prefix
            full.mkdir(parents=True, exist_ok=True)
            filename = Path(prefix).name
            counter = 0
            subfolder = str(Path(prefix))
            return str(full), filename, counter, subfolder, filename_prefix
        fp.get_output_directory = _get_output_directory  # type: ignore[attr-defined]
        fp.get_save_image_path = _get_save_image_path    # type: ignore[attr-defined]
        sys.modules["folder_paths"] = fp
    pkg_name = "ComfyUI_MoGe2_360"
    init_file = REPO_ROOT / "__init__.py"
    assert init_file.exists(), "__init__.py missing at repo root"
    spec = importlib.util.spec_from_file_location(pkg_name, str(init_file))
    pkg = importlib.util.module_from_spec(spec)
    # Mark as package and register before execution so relative imports resolve
    pkg.__path__ = [str(REPO_ROOT)]  # type: ignore[attr-defined]
    sys.modules[pkg_name] = pkg
    assert spec and spec.loader, "invalid import spec for package"
    spec.loader.exec_module(pkg)  # type: ignore[arg-type]
    nodes = importlib.import_module(f"{pkg_name}.nodes")
    assert hasattr(nodes, 'MoGe2Panorama'), "MoGe2Panorama not found"
    Node = nodes.MoGe2Panorama
    spec = Node.INPUT_TYPES()
    req = spec.get('required', {})
    opt = spec.get('optional', {})
    # Verify new panorama fusion options exist
    for key in [
        'merge_method', 'zbuffer_mode', 'skip_small_masks', 'min_mask_ratio',
        'wrap_consistency', 'polar_smooth', 'align_in_disparity'
    ]:
        assert key in req, f"Missing required option: {key}"
    # Verify mask-based GLB export options exist
    for key in [
        'mask_image', 'multi_glb_from_mask', 'mask_ignore_zero',
        'min_label_area_ratio', 'multi_glb_prefix'
    ]:
        assert key in opt, f"Missing optional option: {key}"


def main():
    failures = []
    for name, fn in [
        ("panorama_utils", test_panorama_utils),
        ("validate_script", test_validate_script),
        ("nodes_api", test_nodes_api),
    ]:
        try:
            print(f"[TEST] {name} ...", flush=True)
            fn()
            print(f"[OK]   {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
            failures.append(name)

    if failures:
        print("\nSome tests failed:", ", ".join(failures))
        sys.exit(1)
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
