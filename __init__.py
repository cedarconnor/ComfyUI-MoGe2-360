import os
import sys

_pkg_dir = os.path.dirname(__file__)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

try:
    import utils3d  # noqa: F401
except ModuleNotFoundError:
    from . import utils3d as _local_utils3d
    sys.modules['utils3d'] = _local_utils3d

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

