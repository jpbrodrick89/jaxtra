import importlib.machinery
import importlib.util
import os
import sys


def _load_so(so_stem: str, mod_name: str, required: bool = True):
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # jaxtra/
    candidates = []
    for d in (pkg_dir, os.path.dirname(pkg_dir)):
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            path = os.path.join(d, f"{so_stem}{suffix}")
            if os.path.exists(path):
                candidates.append(path)
    if not candidates:
        if required:
            raise ImportError(
                f"{so_stem}.so not found. "
                "Build it with:  pip install -e . --no-build-isolation"
            )
        return None
    spec = importlib.util.spec_from_file_location(mod_name, candidates[0])
    ext = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = ext
    spec.loader.exec_module(ext)
    return ext
