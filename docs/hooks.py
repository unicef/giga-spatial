"""MkDocs hook: canonical URLs use the ``latest`` alias, not the docs version."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SITE_ROOT = Path(__file__).parent / "scripts" / "site_root.py"
_spec = importlib.util.spec_from_file_location("site_root", _SITE_ROOT)
_site_root = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_site_root)


def on_post_page(output, page, config):
    return _site_root.retarget_versioned_urls(output)
