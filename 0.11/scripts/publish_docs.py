"""Publish the docs for a release, then the site-root files.

A package release ``0.11.x`` updates the docs version ``0.11`` and points
``latest`` at it. After mike deploys that version, this writes ``robots.txt``
and a root sitemap whose URLs use ``/latest/``, and points every version's
canonical URL at ``/latest/``.

From a checkout of the release:

    python docs/scripts/publish_docs.py --push
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from site_root import (
    docs_version,
    ensure_verification,
    latest_version,
    read_package_version,
    retarget_versioned_urls,
    robots_txt,
    to_latest_sitemap,
    verification_token,
)

REPO = Path(__file__).resolve().parents[2]
_VERSION_DIR = re.compile(r"^\d+\.\d+$")


def _run(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def deploy(repo: Path, version: str, push: bool) -> None:
    mike = shutil.which("mike")
    if mike is None:
        raise SystemExit("mike is not on PATH")
    command = [
        mike,
        "deploy",
        version,
        "latest",
        "--update-aliases",
        "--message",
        f"Deploy docs {version} as latest.",
    ]
    if push:
        command.append("--push")
    _run(command, repo)


def update_site_root(repo: Path, push: bool) -> None:
    _run(["git", "fetch", "origin", "gh-pages"], repo)
    work = Path(tempfile.mkdtemp(prefix="gs-gh-pages-"))
    _run(["git", "worktree", "add", "--detach", str(work), "origin/gh-pages"], repo)
    try:
        _write_site_root(repo, work)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=work,
            check=True,
            capture_output=True,
            text=True,
        )
        if not status.stdout.strip():
            print("site root already up to date")
            return
        _run(["git", "add", "-A"], work)
        _run(
            [
                "git",
                "commit",
                "-m",
                "Publish robots.txt, the root sitemap, and canonical URLs for latest.",
            ],
            work,
        )
        if push:
            _run(["git", "push", "origin", "HEAD:gh-pages"], work)
    finally:
        _run(["git", "worktree", "remove", "--force", str(work)], repo)


def _write_site_root(repo: Path, work: Path) -> None:
    versions = json.loads((work / "versions.json").read_text())
    version = latest_version(versions)
    sitemap = (work / version / "sitemap.xml").read_text()
    (work / "sitemap.xml").write_text(to_latest_sitemap(sitemap, version))
    (work / "robots.txt").write_text(robots_txt())

    template = (repo / "overrides" / "main.html").read_text()
    token = verification_token(template)
    index = work / "index.html"
    if token and index.exists():
        index.write_text(ensure_verification(index.read_text(), token))

    for directory in work.iterdir():
        if not directory.is_dir() or _VERSION_DIR.match(directory.name) is None:
            continue
        for page in directory.rglob("*.html"):
            original = page.read_text()
            updated = retarget_versioned_urls(original)
            if updated != original:
                page.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--push",
        action="store_true",
        help="push the docs version and the site-root commit",
    )
    parser.add_argument(
        "--root-only",
        action="store_true",
        help="skip mike deploy and only update the site root of the published docs",
    )
    args = parser.parse_args()

    if not args.root_only:
        version = docs_version(read_package_version(REPO))
        print(f"deploying docs {version} from package {read_package_version(REPO)}")
        deploy(REPO, version, push=args.push)
    update_site_root(REPO, push=args.push)


if __name__ == "__main__":
    main()
