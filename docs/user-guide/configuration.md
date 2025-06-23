# Configuration

The `gigaspatial` package uses a unified configuration system, managed by the `config.py` file, to handle paths, API keys, and other settings. This guide explains how to configure the package for your environment.

---

## Environment Variables Overview

Configuration is primarily managed via environment variables, which can be set in a `.env` file at the project root. Below is a table of all supported environment variables, their defaults, and descriptions:

| Variable                        | Default         | Description                                      |
|----------------------------------|-----------------|--------------------------------------------------|
| ADLS_CONNECTION_STRING           | ""              | Azure Data Lake connection string                 |
| ADLS_CONTAINER_NAME              | ""              | Azure Data Lake container name                    |
| GOOGLE_SERVICE_ACCOUNT           | ""              | Google service account credentials                |
| API_PROFILE_FILE_PATH            | profile.share   | Path to API profile file                          |
| API_SHARE_NAME                   | ""              | API share name                                    |
| API_SCHEMA_NAME                  | ""              | API schema name                                   |
| MAPBOX_ACCESS_TOKEN              | ""              | Mapbox API access token                           |
| MAXAR_USERNAME                   | ""              | Maxar API username                                |
| MAXAR_PASSWORD                   | ""              | Maxar API password                                |
| MAXAR_CONNECTION_STRING          | ""              | Maxar API connection string/key                   |
| OPENCELLID_ACCESS_TOKEN          | ""              | OpenCellID API access token                       |
| GEOREPO_API_KEY                  | ""              | UNICEF GeoRepo API key                            |
| GEOREPO_USER_EMAIL               | ""              | UNICEF GeoRepo user email                         |
| GIGA_SCHOOL_LOCATION_API_KEY     | ""              | GIGA School Location API key                      |
| GIGA_SCHOOL_PROFILE_API_KEY      | ""              | GIGA School Profile API key                       |
| GIGA_SCHOOL_MEASUREMENTS_API_KEY | ""              | GIGA School Measurements API key                  |
| ROOT_DATA_DIR                    | .               | Root directory for all data tiers                 |
| BRONZE_DIR                       | bronze          | Directory for raw/bronze tier data                |
| SILVER_DIR                       | silver          | Directory for processed/silver tier data          |
| GOLD_DIR                         | gold            | Directory for final/gold tier data                |
| VIEWS_DIR                        | views           | Directory for views data                          |
| CACHE_DIR                        | cache           | Directory for cache/temp files                    |
| ADMIN_BOUNDARIES_DIR             | admin_boundaries| Directory for admin boundary data                 |

> **Tip:** You can copy `.env_sample` to `.env` and fill in your values.

---

## Example `.env` File

```bash
# Data directories
BRONZE_DIR=/path/to/your/bronze_tier_data
SILVER_DIR=/path/to/your/silver_tier_data
GOLD_DIR=/path/to/your/gold_tier_data
VIEWS_DIR=/path/to/your/views_data
CACHE_DIR=/path/to/your/cache
ADMIN_BOUNDARIES_DIR=/path/to/your/admin_boundaries

# API keys and credentials
MAPBOX_ACCESS_TOKEN=your_mapbox_token_here
MAXAR_USERNAME=your_maxar_username_here
MAXAR_PASSWORD=your_maxar_password_here
MAXAR_CONNECTION_STRING=your_maxar_key_here
# ... other keys ...
```

---

## How Configuration is Loaded

- The `config.py` file uses [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) to load environment variables from `.env` (if present) or the system environment.
- All directory paths are resolved as `Path` objects. If a path is relative, it is resolved relative to the current working directory.
- Defaults are used if environment variables are not set.

---

## Setting Paths and Keys Programmatically

You can override directory paths in your code using the `set_path` method:

```python
from gigaspatial.config import config

# Set custom data storage paths
config.set_path("bronze", "/path/to/your/bronze_tier_data")
config.set_path("gold", "/path/to/your/gold_tier_data")
config.set_path("views", "/path/to/your/views_data")
```

> **Note:** API keys and credentials should be set via environment variables for security.

---

## Ensuring Directories Exist

To ensure all configured directories exist (and optionally create them if missing):

```python
from gigaspatial.config import config

# Raise error if any directory does not exist
config.ensure_directories_exist(create=False)

# Or, create missing directories automatically
config.ensure_directories_exist(create=True)
```

---

## Verifying the Configuration

You can print the current configuration for debugging:

```python
from gigaspatial.config import config
print(config)
```

---

## Troubleshooting

- **.env File Location:** Ensure `.env` is in your project root.
- **Absolute Paths:** Use absolute paths for directories to avoid confusion.
- **Environment Variable Precedence:** Values in `.env` override defaults, but can be overridden by system environment variables.
- **Missing Directories:** Use `config.ensure_directories_exist(create=True)` to create missing directories.
- **API Keys:** Double-check that all required API keys are set for the services you use.

---

## Next Steps

Once configuration is set up, proceed to the [Data Handling Guide](data-handling/downloading.md) *(Coming Soon)* to start using `gigaspatial`.

---

[Back to User Guide](../index.md)