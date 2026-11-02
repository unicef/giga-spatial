# Configuration

Paths and credentials come from the environment, or from a `.env` file in the project root.

## Data directory

| Variable | Default | What it holds |
| --- | --- | --- |
| `ROOT_DATA_DIR` | `.` | The directory the others sit under |
| `BRONZE_DIR` | `bronze` | Files a handler has downloaded |
| `SILVER_DIR` | `silver` | Files you have processed |
| `GOLD_DIR` | `gold` | Results you keep |
| `CACHE_DIR` | `cache` | Temporary files |

A handler builds its own folder under `BRONZE_DIR`. With `ROOT_DATA_DIR=/data`, WorldPop files land in `/data/bronze/worldpop`.

## Credentials

Set these only for the datasets you use. WorldPop, GHSL, OpenStreetMap, and Ookla do not need a key for ordinary access.

| Variable | Used by |
| --- | --- |
| `MAPBOX_ACCESS_TOKEN` | Mapbox images |
| `MAXAR_API_KEY` | Maxar images |
| `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD` | NASA SRTM |
| `OPENCELLID_ACCESS_TOKEN` | OpenCellID |
| `HEALTHSITES_API_KEY` | Healthsites.io |
| `GEOREPO_API_KEY`, `GEOREPO_USER_EMAIL` | UNICEF GeoRepo boundaries |
| `GIGA_SCHOOL_LOCATION_API_KEY` | Giga school locations |
| `GIGA_SCHOOL_PROFILE_API_KEY` | Giga school profiles |
| `GIGA_SCHOOL_MEASUREMENTS_API_KEY` | Giga school connectivity measurements |
| `GOOGLE_CLOUD_PROJECT`, `GOOGLE_SERVICE_ACCOUNT`, `GOOGLE_SERVICE_ACCOUNT_KEY_PATH` | Google Earth Engine |
| `ADLS_CONNECTION_STRING`, `ADLS_CONTAINER_NAME` | Azure Data Lake |

## Services

| Variable | Default | Used by |
| --- | --- | --- |
| `OVERPASS_URL` | `https://overpass-api.de/api/interpreter` | OpenStreetMap road networks. Set a mirror when the main server is down. |

## Data stores

`LocalDataStore` is the default. `ADLSDataStore`, `SnowflakeDataStore`, and `DeltaSharingDataStore` are the remote stores. Pass one as `data_store=` to a handler or a generator. They are documented in the [API reference](../api/data-stores.md).
