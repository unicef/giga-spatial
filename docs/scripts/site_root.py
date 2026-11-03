"""Site-root files for the versioned docs.

Package releases are ``0.11.x``. The docs version is ``0.11``. ``latest`` is an
alias of that docs version, so the URLs Google should rank use ``/latest/``.
"""

from __future__ import annotations

import re
from pathlib import Path

SITE = "https://unicef.github.io/giga-spatial"
_VERSIONED_URL = re.compile(rf"{re.escape(SITE)}/\d+\.\d+/")
_TAG = re.compile(r"<link\b[^>]*>|<meta\b[^>]*>", re.IGNORECASE)
_VERIFICATION = re.compile(
    r'<meta name="google-site-verification" content="([^"]+)"\s*/?>'
)


def docs_version(package_version: str) -> str:
    """Return the docs version for a package release.

    ``0.11.0`` and ``0.11.1`` both publish the docs version ``0.11``.
    """
    parts = package_version.strip().lstrip("v").split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(
            f"package version must be major.minor[.patch], got {package_version!r}"
        )
    return f"{int(parts[0])}.{int(parts[1])}"


def latest_version(versions: list) -> str:
    """Return the version whose aliases include ``latest``."""
    for item in versions:
        if "latest" in item.get("aliases", []):
            return item["version"]
    raise ValueError("no version has the latest alias")


def to_latest_sitemap(sitemap_xml: str, version: str) -> str:
    """Rewrite a version's sitemap so its URLs use the ``latest`` alias.

    A sitemap already written with ``/latest/`` is returned unchanged. mike
    does that when ``canonical_version`` is ``latest``.
    """
    source = f"{SITE}/{version}/"
    target = f"{SITE}/latest/"
    if source in sitemap_xml:
        return sitemap_xml.replace(source, target)
    if target in sitemap_xml:
        return sitemap_xml
    raise ValueError(f"sitemap has no URLs under {source}")


def retarget_versioned_urls(html: str) -> str:
    """Point canonical and Open Graph URLs at ``/latest/``.

    Asset URLs stay on the version that holds the files.
    """

    def fix(match: re.Match) -> str:
        tag = match.group(0)
        lowered = tag.lower()
        canonical = "rel=canonical" in lowered or "rel=\"canonical\"" in lowered
        og_url = "property=og:url" in lowered or "property=\"og:url\"" in lowered
        if not canonical and not og_url:
            return tag
        return _VERSIONED_URL.sub(f"{SITE}/latest/", tag, count=1)

    return _TAG.sub(fix, html)


def verification_token(template: str) -> str | None:
    match = _VERIFICATION.search(template)
    return match.group(1) if match else None


def ensure_verification(html: str, token: str) -> str:
    """Keep the Search Console token on the root redirect page."""
    if "google-site-verification" in html:
        return html
    tag = f'<meta name="google-site-verification" content="{token}" />'
    needle = '<meta charset="utf-8">'
    if needle not in html:
        return html
    return html.replace(needle, needle + "\n  " + tag, 1)


def robots_txt() -> str:
    return (
        "User-agent: *\n"
        "Allow: /\n"
        "\n"
        f"Sitemap: {SITE}/sitemap.xml\n"
    )


def read_package_version(repo: Path) -> str:
    text = (repo / "gigaspatial" / "__init__.py").read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise ValueError("gigaspatial/__init__.py has no __version__")
    return match.group(1)
