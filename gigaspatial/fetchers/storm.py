"""
Fetch STORM tropical cyclone wind-speed return periods into memory.

Russell (2024) mosaicked the Bloemendaal et al. basin maps into one global
GeoTIFF per climate model and return period, and published that archive on
Zenodo (CC0): https://doi.org/10.5281/zenodo.10931452

Each GeoTIFF is maximum wind speed in m/s at 10 km resolution.
``constant`` is the present-day baseline. The other models are 2050 under
RCP8.5.

``fetch`` returns a ``TifProcessor``. The Zenodo zip is about 1.1 GB. It is
downloaded into a temporary directory owned by the fetcher, and later maps
are extracted from that copy. Nothing is written to a data store. ``close``
removes the temporary archive. The returned processor keeps its own copy of
the GeoTIFF.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import zipfile
from pathlib import Path, PurePosixPath
from typing import Tuple

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.base.config import source_geometry
from gigaspatial.handlers.base.fetch import stream_to_store
from gigaspatial.handlers.base.source import SourceKind, classify_source
from gigaspatial.processing.tif_processor import TifProcessor


ARCHIVE_URL = (
    "https://zenodo.org/api/records/10931452/files/"
    "STORM_FIXED_RETURN_PERIODS.zip/content"
)
ARCHIVE_NAME = "STORM_FIXED_RETURN_PERIODS.zip"
ARCHIVE_MD5 = "92450d478ff80ec0a723a3a24bd3aaf0"
DOI = "10.5281/zenodo.10931452"

MODELS: Tuple[str, ...] = (
    "constant",
    "CMCC-CM2-VHR4",
    "CNRM-CM6-1-HR",
    "EC-Earth3P-HR",
    "HadGEM3-GC31-HM",
)
RETURN_PERIODS: Tuple[int, ...] = (
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
)
DEFAULT_MODEL = "constant"
DEFAULT_RETURN_PERIOD = 100


def filename(model: str, return_period: int) -> str:
    """Return the published GeoTIFF name for one model and return period."""
    return f"STORM_FIXED_RETURN_PERIODS_{model}_{return_period}_YR_RP.tif"


@dataclass(config=ConfigDict(extra="forbid"), kw_only=True)
class STORMFetcher:
    """
    Fetch one STORM tropical cyclone wind-speed map into memory.

    The result is a ``TifProcessor`` of maximum wind speed in m/s. Pass a
    source to clip the global map to that place. With no source, the result
    is the global raster.

    The Zenodo zip (about 1.1 GB) is kept in a temporary directory for the
    life of this fetcher so later models and return periods skip the
    download. It is not written to a data store. ``close`` deletes that
    directory. Licence: CC0. DOI: https://doi.org/10.5281/zenodo.10931452

    Attributes:
        show_progress: If True, shows a byte progress bar while downloading.
        timeout: Seconds to wait for the server between bytes.

    Examples:
        >>> winds = STORMFetcher().fetch()
        >>> kenya = STORMFetcher().fetch("KEN", model="HadGEM3-GC31-HM", return_period=500)
    """

    show_progress: bool = True
    timeout: float = Field(default=60, gt=0)

    def __post_init__(self) -> None:
        self._root = Path(tempfile.mkdtemp(prefix="gigaspatial-storm-"))
        self._store = LocalDataStore(self._root)
        self._archive_lock = threading.Lock()
        self._closed = False

    def fetch(
        self,
        source=None,
        *,
        model: str = DEFAULT_MODEL,
        return_period: int = DEFAULT_RETURN_PERIOD,
    ) -> TifProcessor:
        """
        Fetch one wind-speed return-period map.

        Args:
            source: Country, geometry, points, DataFrame, or GeoDataFrame.
                Omit it for the global map. A file path is an error.
            model: Climate model. ``constant`` is the present-day baseline.
                ``CMCC-CM2-VHR4``, ``CNRM-CM6-1-HR``, ``EC-Earth3P-HR``, and
                ``HadGEM3-GC31-HM`` are 2050 under RCP8.5.
            return_period: Return period in years. One of 10, 20, 30, 40, 50,
                60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, or 10000.

        Returns:
            A ``TifProcessor``. With a source, the raster is clipped to it.
            Values are maximum wind speed in m/s.

        Raises:
            ValueError: If ``model`` or ``return_period`` is not in the
                publication, or ``source`` is a file path.
            requests.RequestException: If the download fails.
            zipfile.BadZipFile: If the archive is still unreadable after a
                second download.
            FileNotFoundError: If the zip has no member of that name.
        """
        self._check_open()
        member = self._member(model, return_period)
        if source is not None and classify_source(source) is SourceKind.PATHS:
            raise ValueError("fetch() takes a place, not a file path")

        extracted = self._extract(member)
        processor = self._processor(extracted)
        if source is None:
            return processor

        clipped = processor.clip_to_geometry(source_geometry(source))
        processor.cleanup()
        return clipped

    def close(self) -> None:
        """Delete the temporary archive. Returned processors keep their own copies."""
        if self._closed:
            return
        self._closed = True
        shutil.rmtree(self._root, ignore_errors=True)

    def __enter__(self) -> "STORMFetcher":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("This STORMFetcher is closed")

    def _member(self, model: str, return_period: int) -> str:
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {model!r}")
        if return_period not in RETURN_PERIODS:
            raise ValueError(
                "return_period must be one of "
                f"{RETURN_PERIODS}, got {return_period!r}"
            )
        return filename(model, return_period)

    def _extract(self, member_name: str) -> Path:
        try:
            return self._extract_member(self._ensure_archive(), member_name)
        except zipfile.BadZipFile:
            self._discard_archive()
            return self._extract_member(self._ensure_archive(), member_name)

    def _ensure_archive(self) -> Path:
        archive = self._root / ARCHIVE_NAME
        with self._archive_lock:
            if not archive.is_file():
                stream_to_store(
                    ARCHIVE_URL,
                    archive,
                    self._store,
                    show_progress=self.show_progress,
                    desc=ARCHIVE_NAME,
                    timeout=self.timeout,
                    md5=ARCHIVE_MD5,
                )
        return archive

    def _discard_archive(self) -> None:
        archive = self._root / ARCHIVE_NAME
        with self._archive_lock:
            if archive.is_file():
                archive.unlink()

    def _extract_member(self, archive: Path, member_name: str) -> Path:
        destination = self._root / member_name
        with self._archive_lock:
            if destination.is_file():
                return destination
            partial = destination.with_suffix(".partial")
            try:
                with zipfile.ZipFile(archive) as zipped:
                    matches = [
                        name
                        for name in zipped.namelist()
                        if not name.endswith("/")
                        and PurePosixPath(name).name == member_name
                    ]
                    if len(matches) != 1:
                        found = ", ".join(matches) or "none"
                        raise FileNotFoundError(
                            f"{ARCHIVE_NAME} should contain one member named "
                            f"{member_name}, found {found}"
                        )
                    with zipped.open(matches[0]) as member, open(partial, "wb") as out:
                        shutil.copyfileobj(member, out)
                partial.replace(destination)
            except BaseException:
                if partial.is_file():
                    partial.unlink()
                raise
        return destination

    def _processor(self, extracted: Path) -> TifProcessor:
        """A processor that owns a copy of the extracted GeoTIFF."""
        processor = TifProcessor(dataset_path=extracted)
        owned = Path(processor._working_file.derived_path(extracted.name))
        shutil.copyfile(extracted, owned)
        processor._working_file.adopt(str(owned))
        processor.dataset_path = owned
        return processor
