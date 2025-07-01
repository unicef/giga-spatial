from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Union, Literal, Dict, Any
import io
from functools import lru_cache
import logging


class Config(BaseSettings):
    """
    Unified configuration with environment variable loading.
    Manages file system paths for different data tiers (bronze, silver, gold)
    and their subpaths. All paths can be overridden through environment variables.
    """

    ADLS_CONNECTION_STRING: str = Field(default="", alias="ADLS_CONNECTION_STRING")
    ADLS_CONTAINER_NAME: str = Field(default="", alias="ADLS_CONTAINER_NAME")
    GOOGLE_SERVICE_ACCOUNT: str = Field(default="", alias="GOOGLE_SERVICE_ACCOUNT")
    API_PROFILE_FILE_PATH: Path = Field(
        default=Path("profile.share"), alias="API_PROFILE_FILE_PATH"
    )
    API_SHARE_NAME: str = Field(default="", alias="API_SHARE_NAME")
    API_SCHEMA_NAME: str = Field(default="", alias="API_SCHEMA_NAME")
    MAPBOX_ACCESS_TOKEN: str = Field(default="", alias="MAPBOX_ACCESS_TOKEN")
    MAXAR_USERNAME: str = Field(default="", alias="MAXAR_USERNAME")
    MAXAR_PASSWORD: str = Field(default="", alias="MAXAR_PASSWORD")
    MAXAR_CONNECTION_STRING: str = Field(default="", alias="MAXAR_CONNECTION_STRING")
    OPENCELLID_ACCESS_TOKEN: str = Field(default="", alias="OPENCELLID_ACCESS_TOKEN")
    GEOREPO_API_KEY: str = Field(default="", alias="GEOREPO_API_KEY")
    GEOREPO_USER_EMAIL: str = Field(default="", alias="GEOREPO_USER_EMAIL")
    GIGA_SCHOOL_LOCATION_API_KEY: str = Field(
        default="", alias="GIGA_SCHOOL_LOCATION_API_KEY"
    )
    GIGA_SCHOOL_PROFILE_API_KEY: str = Field(
        default="", alias="GIGA_SCHOOL_PROFILE_API_KEY"
    )
    GIGA_SCHOOL_MEASUREMENTS_API_KEY: str = Field(
        default="", alias="GIGA_SCHOOL_MEASUREMENTS_API_KEY"
    )

    ROOT_DATA_DIR: Path = Field(
        default=Path("."),
        description="Root directory for all data tiers",
        alias="ROOT_DATA_DIR",
    )

    BRONZE_DATA_DIR: Path = Field(
        default=Path("bronze"),
        description="Root directory for raw/bronze tier data",
        alias="BRONZE_DIR",
    )
    SILVER_DATA_DIR: Path = Field(
        default=Path("silver"),
        description="Root directory for processed/silver tier data",
        alias="SILVER_DIR",
    )
    GOLD_DATA_DIR: Path = Field(
        default=Path("gold"),
        description="Root directory for final/gold tier data",
        alias="GOLD_DIR",
    )
    VIEWS_DATA_DIR: Path = Field(
        default=Path("views"),
        description="Root directory for views data",
        alias="VIEWS_DIR",
    )
    CACHE_DIR: Path = Field(
        default=Path("cache"),
        description="Directory for temporary/cache files",
        alias="CACHE_DIR",
    )
    ADMIN_BOUNDARIES_DATA_DIR: Optional[Path] = Field(
        default=None,
        description="Root directory for administrative boundary data",
        alias="ADMIN_BOUNDARIES_DIR",
    )
    DB_CONFIG: Optional[Dict] = Field(default=None, alias="DB_CONFIG")

    DATA_TYPES: Dict[str, str] = Field(
        default={
            "google_open_buildings": "google_open_buildings",
            "mapbox_image": "mapbox_images",
            "microsoft_global_buildings": "microsoft_global_buildings",
            "ookla_speedtest": "ookla",
            "srtm": "srtm",
            "worldpop": "worldpop",
            "ghsl": "ghsl",
            "opencellid": "opencellid",
            "hdx": "hdx",
            "poi": "poi",
            "zonal": "zonal",
        },
        description="Mapping of data types to directory names",
    )

    def get_logger(self, name="GigaSpatial", console_level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)

        LOG_FORMAT = "%(levelname) -10s  %(name) -10s %(asctime) " "-30s: %(message)s"

        formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(formatter)

        if not logger.hasHandlers():
            logger.addHandler(console_handler)

        return logger

    def get_tqdm_logger_stream(self, logger: logging.Logger, level=logging.INFO):
        return TqdmToLogger(logger, level=level)

    def set_path(
        self,
        tier: Literal["bronze", "silver", "gold", "views"],
        path: Union[str, Path],
    ) -> None:
        """Dynamically set the base path for a given tier."""
        if tier not in ["bronze", "silver", "gold", "views"]:
            raise ValueError(
                f"Invalid tier: {tier}. Must be one of 'bronze', 'silver', 'gold', or 'views'."
            )

        if isinstance(path, str):
            path = Path(path)

        setattr(self, f"{tier.upper()}_DATA_DIR", path)

    def get_path(
        self,
        data_type: str,
        tier: Literal["bronze", "silver", "gold", "views"],
        version: Optional[str] = None,
    ) -> Path:
        """Dynamic path construction based on data type and tier."""
        if tier not in ["bronze", "silver", "gold", "views"]:
            raise ValueError(
                f"Invalid tier: {tier}. Must be one of 'bronze', 'silver', 'gold', or 'views'."
            )

        tier_dir = getattr(self, f"{tier.upper()}_DATA_DIR")
        type_dir = self.DATA_TYPES[data_type]
        if version:
            return self.ROOT_DATA_DIR / tier_dir / type_dir / version
        else:
            return self.ROOT_DATA_DIR / tier_dir / type_dir

    def get_admin_path(
        self,
        country_code,
        admin_level: Literal[0, 1, 2, 3, 4],
        file_suffix: str = ".geojson",
    ) -> Path:
        """Dynamic path construction for administrative boundary data based on admin level."""
        base_dir = getattr(self, "ADMIN_BOUNDARIES_DATA_DIR")
        if base_dir is None:
            raise ValueError(
                "ADMIN_BOUNDARIES_DATA_DIR is not configured. "
                "Please set the ADMIN_BOUNDARIES_DIR environment variable."
            )
        level_dir = f"admin{admin_level}"
        file = f"{country_code}_{level_dir}{file_suffix}"

        return base_dir / level_dir / file

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        validate_assignment=True,
        case_sensitive=True,
        extra="allow",
    )

    @field_validator(
        "BRONZE_DATA_DIR",
        "SILVER_DATA_DIR",
        "GOLD_DATA_DIR",
        "CACHE_DIR",
        mode="before",
    )
    def resolve_and_validate_paths(
        cls, value: Union[str, Path], resolve=False
    ) -> Union[Path, Any]:
        """Smart validator that only processes Path fields"""

        if isinstance(value, str):
            path = Path(value)
        elif isinstance(value, Path):
            path = value
        else:
            raise ValueError(f"Invalid path type for {field.name}: {type(value)}")

        resolved = path.expanduser().resolve()
        return resolved if resolve else path

    @field_validator("ADMIN_BOUNDARIES_DATA_DIR", mode="before")
    def validate_admin_boundaries_dir(
        cls, value: Union[str, Path, None]
    ) -> Optional[Path]:
        """Validator for ADMIN_BOUNDARIES_DATA_DIR that handles None and string values."""
        if value is None:
            return None
        if isinstance(value, str):
            return Path(value)
        elif isinstance(value, Path):
            return value
        else:
            raise ValueError(
                f"Invalid path type for ADMIN_BOUNDARIES_DATA_DIR: {type(value)}"
            )

    def ensure_directories_exist(self, create: bool = False) -> None:
        """Ensures all configured directories exist."""
        for field_name, field_value in self.__dict__.items():
            if (
                isinstance(field_value, Path)
                and field_value is not None
                and not field_value.exists()
            ):
                if create:
                    field_value.mkdir(parents=True, exist_ok=True)
                else:
                    raise FileNotFoundError(f"Directory does not exist: {field_value}")


class TqdmToLogger(io.StringIO):
    """
    File-like object to redirect tqdm output to a logger.
    """

    def __init__(self, logger, level=logging.INFO):
        super().__init__()
        self.logger = logger
        self.level = level
        self.buf = ""  # To store partial writes

    def write(self, buf):
        # tqdm often writes partial lines, and then a full line with \r
        # We accumulate buffer and only log when a full line (or significant update) is received
        self.buf += buf
        if "\r" in buf or "\n" in buf:  # Heuristic for a "full" update
            self.logger.log(self.level, self.buf.strip("\r\n"))
            self.buf = ""  # Reset buffer after logging

    def flush(self):
        # Ensure any remaining buffer is logged on flush
        if self.buf:
            self.logger.log(self.level, self.buf.strip("\r\n"))
            self.buf = ""


@lru_cache()
def get_default_config() -> Config:
    """Returns a singleton instance of Config."""
    return Config()


# Singleton instance
config = get_default_config()
