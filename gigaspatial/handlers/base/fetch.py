"""
Shared helpers for downloaders: streaming a URL into a data store, extracting
a zip archive into it, and selecting the tiles of a grid that a geometry hits.

The download helpers raise on any failure and leave no partial file behind,
which is the contract `BaseHandlerDownloader.download_data_unit` expects.
"""
import re
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import requests
from pyproj import CRS
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from gigaspatial.core.io.data_store import DataStore

CHUNK_SIZE = 1024 * 1024
TIMEOUT = 60


def _stream(url, write, *, session, show_progress, desc, timeout) -> None:
    getter = session.get if session is not None else requests.get
    response = getter(url, stream=True, timeout=timeout)
    try:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=desc,
            disable=not show_progress,
        ) as bar:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    write(chunk)
                    bar.update(len(chunk))
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            close()


def stream_to_store(
    url: str,
    path: Union[str, Path],
    store: DataStore,
    *,
    session: Optional[requests.Session] = None,
    show_progress: bool = False,
    desc: Optional[str] = None,
    timeout: float = TIMEOUT,
) -> Path:
    """
    Download a URL into a file of the data store.

    Args:
        url: The remote file.
        path: Where to store it.
        store: The data store to write to.
        session: Session to send the request with (e.g. one carrying
            credentials); a plain ``requests.get`` otherwise.
        show_progress: If True, shows a byte progress bar.
        desc: Progress bar label; defaults to the file name.
        timeout: Seconds to wait for the server between bytes.

    Returns:
        The stored path.

    Raises:
        requests.RequestException: If the request fails. A partially written
            file is removed.
    """
    path = Path(path)
    try:
        with store.open(str(path), "wb") as file:
            _stream(
                url,
                file.write,
                session=session,
                show_progress=show_progress,
                desc=desc or path.name,
                timeout=timeout,
            )
    except BaseException:
        if store.file_exists(str(path)):
            store.remove(str(path))
        raise
    return path


def extract_zip_to_store(
    url: str,
    dest_dir: Union[str, Path],
    store: DataStore,
    pattern: Optional[str] = None,
    *,
    session: Optional[requests.Session] = None,
    show_progress: bool = False,
    desc: Optional[str] = None,
    timeout: float = TIMEOUT,
) -> List[Path]:
    """
    Download a zip archive and store its matching members in a directory.

    The archive goes to a local temporary file, which is always deleted.
    Members are stored flat, under their base names.

    Args:
        url: The remote zip archive.
        dest_dir: Directory of the data store to extract into.
        store: The data store to write to.
        pattern: Regular expression a member name must match from its start
            (``re.match``); None extracts every file.
        session: Session to send the request with.
        show_progress: If True, shows a byte progress bar.
        desc: Progress bar label; defaults to the archive name.
        timeout: Seconds to wait for the server between bytes.

    Returns:
        The stored paths of the extracted members.

    Raises:
        requests.RequestException: If the request fails.
        zipfile.BadZipFile: If the download is not a zip archive.
        FileNotFoundError: If no member matches ``pattern``.
    """
    dest_dir = Path(dest_dir)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "archive.zip"
        with open(archive, "wb") as file:
            _stream(
                url,
                file.write,
                session=session,
                show_progress=show_progress,
                desc=desc or url.rsplit("/", 1)[-1],
                timeout=timeout,
            )

        with zipfile.ZipFile(archive) as zf:
            members = [
                name
                for name in zf.namelist()
                if not name.endswith("/")
                and (pattern is None or re.match(pattern, name))
            ]
            if not members:
                raise FileNotFoundError(
                    f"No member of {url} matches the pattern {pattern!r}"
                )
            stored = []
            for name in members:
                path = dest_dir / Path(name).name
                with zf.open(name) as member:
                    store.write_file(str(path), member)
                stored.append(path)
    return stored


def tiles_intersecting(
    tiles: gpd.GeoDataFrame, geometry: BaseGeometry, crs
) -> gpd.GeoDataFrame:
    """
    Return the rows of a tile grid whose geometry intersects a geometry.

    Args:
        tiles: The tile grid.
        geometry: The search geometry.
        crs: CRS of ``geometry``; it is reprojected to the grid's CRS when
            they differ.

    Returns:
        The intersecting tiles, in grid order.
    """
    if tiles.crs is not None and CRS.from_user_input(crs) != CRS.from_user_input(
        tiles.crs
    ):
        geometry = gpd.GeoSeries([geometry], crs=crs).to_crs(tiles.crs).iloc[0]
    hits = tiles.sindex.query(geometry, predicate="intersects")
    return tiles.iloc[sorted(hits)]
