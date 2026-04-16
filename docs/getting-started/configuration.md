# Configuration Guide

GigaSpatial uses a unified configuration system managed by Pydantic. Configuration can be handled through **Environment Variables** or a `.env` file located in your project root.

## Core Configuration Logic

The `gigaspatial.config` module provides a singleton `config` object that manages all system settings.

### Path Management & Data Tiers

GigaSpatial organizes data into a hierarchical "Medallion Architecture" (Bronze, Silver, Gold). All paths are resolved relative to a root data directory.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `ROOT_DATA_DIR` | The base directory for all library data. | `.` |
| `BRONZE_DIR` | Root for raw, source-aligned data. | `bronze` |
| `SILVER_DIR` | Root for processed, cleaned data. | `silver` |
| `GOLD_DIR` | Root for finalized, aggregated data. | `gold` |
| `CACHE_DIR` | Base for temporary/cached files. | `cache` |
| `ADMIN_BOUNDARIES_DIR` | Root for administrative boundary files. | *Required for some tasks* |

> [!NOTE]
> GigaSpatial automatically constructs subpaths for specific handlers. For example, if `ROOT_DATA_DIR=/data`, the path for raw WorldPop data will resolve to `/data/bronze/worldpop`.

---

## Storage Backends

GigaSpatial supports multiple storage backends via the `DataStore` abstraction.

### 1. Local Storage (Default)
If no specialized environment variables are set, GigaSpatial defaults to using your local file system as the primary storage.

### 2. Azure Data Lake Storage (ADLS)
To use ADLS, install the optional dependencies: `pip install "giga-spatial[azure]"`.

| Variable | Requirement | Description |
| :--- | :--- | :--- |
| `ADLS_CONTAINER_NAME` | **Required** | The name of your Azure container. |
| `ADLS_CONNECTION_STRING` | Optional* | Full connection string for authentication. |
| `ADLS_ACCOUNT_URL` | Optional* | The URL of your storage account. |
| `ADLS_SAS_TOKEN` | Optional* | SAS token (use with `ADLS_ACCOUNT_URL`). |

> [!IMPORTANT]
> You must provide either `ADLS_CONNECTION_STRING` **OR** both `ADLS_ACCOUNT_URL` and `ADLS_SAS_TOKEN`.

### 3. Snowflake Internal Stages
To use Snowflake, install the optional dependencies: `pip install "giga-spatial[snowflake]"`.

| Variable | Description | Example |
| :--- | :--- | :--- |
| `SNOWFLAKE_ACCOUNT` | Account identifier | `xy12345.east-us-2.azure` |
| `SNOWFLAKE_USER` | Username | `JSOW` |
| `SNOWFLAKE_PASSWORD` | Password | `********` |
| `SNOWFLAKE_WAREHOUSE` | Compute warehouse | `COMPUTE_WH` |
| `SNOWFLAKE_DATABASE` | Target database | `GIGA_DB` |
| `SNOWFLAKE_SCHEMA` | Target schema | `RAW` |
| `SNOWFLAKE_STAGE_NAME` | Internal stage for file storage | `GIGA_STAGE` |

---

## External API Keys

Many data handlers require external credentials to fetch data.

### Google Earth Engine (GEE)
Used by the GEE handler and zonal statistics modules.
- `GOOGLE_CLOUD_PROJECT`: Your GCP Project ID.
- `GOOGLE_SERVICE_ACCOUNT`: Service account email.
- `GOOGLE_SERVICE_ACCOUNT_KEY_PATH`: Local path to the `.json` service account key.

### OpenStreetMap (OSM)
By default, GigaSpatial uses a public instance of the Overpass API (`http://overpass-api.de/api/interpreter`).
- No keys are required for basic usage.
- If you use a private instance, you can override the `base_url` parameter in the `OSMLocationFetcher` constructor.

### Ookla Speedtest
GigaSpatial accesses Ookla datasets via their public AWS S3 bucket.
- No AWS credentials or API keys are required for standard access.

### Global Master API Key Table

| Variable | Service | Use Case |
| :--- | :--- | :--- |
| `MAPBOX_ACCESS_TOKEN` | Mapbox | Satellite imagery fetching |
| `MAXAR_API_KEY` | Maxar | High-res satellite imagery |
| `EARTHDATA_USERNAME` | NASA | SRTM / Elevation data |
| `EARTHDATA_PASSWORD` | NASA | (Pair with username) |
| `OPENCELLID_ACCESS_TOKEN` | OpenCellID | Cell tower location database |
| `HEALTHSITES_API_KEY` | Healthsites.io | Global health facility data |
| `GEOREPO_API_KEY` | GeoRepo | UNICEF Administrative boundaries |
| `GEOREPO_USER_EMAIL` | GeoRepo | (Pair with API Key) |
| `GIGA_SCHOOL_LOCATION_API_KEY` | Giga | School locations API |
| `GIGA_SCHOOL_PROFILE_API_KEY` | Giga | School profiles API |
| `GIGA_SCHOOL_MEASUREMENTS_API_KEY` | Giga | Connectivity measurements API |

---

## Example `.env` File

```bash
# General
ROOT_DATA_DIR=/home/user/giga_data

# Azure Data Lake
ADLS_CONTAINER_NAME=mycontainer
ADLS_CONNECTION_STRING='DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;'

# Google Earth Engine
GOOGLE_CLOUD_PROJECT=giga-spatial-project
GOOGLE_SERVICE_ACCOUNT=service-account@giga-spatial.iam.gserviceaccount.com
GOOGLE_SERVICE_ACCOUNT_KEY_PATH=/secrets/gcp-key.json

# Specialized APIs
OPENCELLID_ACCESS_TOKEN=pk.some_token_here
MAPBOX_ACCESS_TOKEN=sk.another_token_here
```
