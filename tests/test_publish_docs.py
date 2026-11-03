import importlib.util
from pathlib import Path

_PATH = Path(__file__).parents[1] / "docs" / "scripts" / "site_root.py"
_spec = importlib.util.spec_from_file_location("site_root", _PATH)
site_root = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(site_root)


def test_patch_releases_share_one_docs_version():
    assert site_root.docs_version("0.11.0") == "0.11"
    assert site_root.docs_version("0.11.1") == "0.11"
    assert site_root.docs_version("v0.12.0") == "0.12"


def test_latest_sitemap_rewrites_only_the_docs_version():
    sitemap = (
        "<url><loc>https://unicef.github.io/giga-spatial/0.11/user-guide/</loc></url>"
    )

    updated = site_root.to_latest_sitemap(sitemap, "0.11")

    assert updated == (
        "<url><loc>https://unicef.github.io/giga-spatial/latest/user-guide/</loc></url>"
    )


def test_canonical_moves_to_latest_and_assets_stay_versioned():
    html = (
        "<link href=https://unicef.github.io/giga-spatial/0.10/user-guide/ rel=canonical>"
        "<meta content=https://unicef.github.io/giga-spatial/0.11/ property=og:url>"
        "<meta property=og:image content=https://unicef.github.io/giga-spatial/0.11/assets/images/social/index.png>"
    )

    updated = site_root.retarget_versioned_urls(html)

    assert "giga-spatial/latest/user-guide/" in updated
    assert "property=og:url" in updated
    assert "giga-spatial/latest/" in updated.split("og:image")[0]
    assert "giga-spatial/0.11/assets/" in updated


def test_sitemap_already_on_latest_is_kept():
    sitemap = "<url><loc>https://unicef.github.io/giga-spatial/latest/</loc></url>"

    assert site_root.to_latest_sitemap(sitemap, "0.11") == sitemap


def test_latest_alias_is_read_from_versions():
    versions = [
        {"version": "0.11", "aliases": ["latest"]},
        {"version": "0.10", "aliases": []},
    ]

    assert site_root.latest_version(versions) == "0.11"
