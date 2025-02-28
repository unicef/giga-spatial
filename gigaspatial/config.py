from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Union, Literal, Dict, Any
from functools import lru_cache


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
    CACHE_DIR: Path = Field(
        default=Path("cache"),
        description="Directory for temporary/cache files",
        alias="CACHE_DIR",
    )
    ADMIN_BOUNDARIES_DATA_DIR: Path = Field(
        default=Path("admin_boundaries"),
        description="Root directory for administrative boundary data",
        alias="ADMIN_BOUNDARIES_DIR",
    )

    DATA_TYPES: Dict[str, str] = Field(
        default={
            "google_open_buildings": "google_open_buildings",
            "mapbox_image": "mapbox_images",
            "microsoft_global_buildings": "microsoft_global_buildings",
            "ookla_speedtest": "ookla",
            "srtm": "srtm",
            "worldpop": "worldpop",
        },
        description="Mapping of data types to directory names",
    )

    def get_path(
        self,
        data_type: str,
        tier: Literal["bronze", "silver", "gold"],
        version: Optional[str] = None,
    ) -> Path:
        """Dynamic path construction based on data type and tier."""
        base_dir = getattr(self, f"{tier.upper()}_DATA_DIR")
        type_dir = self.DATA_TYPES[data_type]
        if version:
            return base_dir / type_dir / version
        else:
            return base_dir / type_dir

    def get_admin_path(
        self,
        country_code,
        admin_level: Literal[0, 1, 2, 3, 4],
        file_suffix: str = ".geojson",
    ) -> Path:
        """Dynamic path construction for administrative boundary data based on admin level."""
        base_dir = getattr(self, "ADMIN_BOUNDARIES_DATA_DIR")
        level_dir = f"admin{admin_level}"
        file = f"{country_code}_{level_dir}{file_suffix}"

        return base_dir / level_dir / file

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix = "", validate_assignment=True, case_sensitive=True, extra="allow"
    )

    @field_validator(
        "BRONZE_DATA_DIR",
        "SILVER_DATA_DIR",
        "GOLD_DATA_DIR",
        "CACHE_DIR",
        "ADMIN_BOUNDARIES_DATA_DIR",
        mode="before",
    )
    def resolve_and_validate_paths(cls, value: Union[str, Path], resolve=False) -> Union[Path, Any]:
        """Smart validator that only processes Path fields"""

        if isinstance(value, str):
            path = Path(value)
        elif isinstance(value, Path):
            path = value
        else:
            raise ValueError(f"Invalid path type for {field.name}: {type(value)}")

        resolved = path.expanduser().resolve()
        return resolved if resolve else path

    def ensure_directories_exist(self, create: bool = False) -> None:
        """Ensures all configured directories exist."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Path) and not field_value.exists():
                if create:
                    field_value.mkdir(parents=True, exist_ok=True)
                else:
                    raise FileNotFoundError(f"Directory does not exist: {field_value}")


@lru_cache()
def get_config() -> Config:
    """Returns a singleton instance of Config."""
    return Config()


# Singleton instance
config = get_config()
