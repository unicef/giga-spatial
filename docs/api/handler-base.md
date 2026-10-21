# Handler base

This section is for adding a dataset to this library. The classes below are internal: this repository is their only subclasser, and the hooks are not a stable API for other packages. Documenting them does not change that. See [the decision](../adr/0002-handler-base-in-the-api-reference.md).

The guides never use these names. A caller passes a source and may set `crop_to_source`. The classes live in `gigaspatial.handlers.base`, which is not a public import path.

## Search key and search geometry

A handler turns a source into files in two different ways.

The **search key** is what the dataset's index understands. It may be a country code, a filter, a geometry, or nothing, for a dataset published as one global file. `get_search_key` is the method a dataset overrides.

The **search geometry** is always a Shapely geometry in the handler's CRS. Cropping uses only that geometry. `extract_search_geometry` builds it. A search key that is a geometry is the search geometry.

The split, and the cases where a country source is not cropped, are recorded in [Search key and search geometry](../adr/0001-search-key-and-search-geometry.md).

Anything that changes which file is fetched belongs on the data unit, not in an option the downloader has to interpret.

## The classes

`BaseHandler` is what a caller holds. `load_data` ensures the files are stored, then reads them. `BaseHandlerConfig` resolves a source into data units and stored paths. `BaseHandlerDownloader` fetches units. `BaseHandlerReader` loads stored paths and may crop.

A downloader or reader declares its config class as `config_type`, so it can also be built on its own from a config, a dict of config fields, or nothing.

Keyword arguments stop at those public methods. Each class declares an options type: `ResolveOptions`, `DownloadOptions`, or `ReadOptions`. An unknown keyword raises `TypeError`.

::: gigaspatial.handlers.base.handler.BaseHandler
    options:
      show_source: false
      members:
        - load_data
        - download_and_load
        - ensure_data_available
        - get_available_data_info
        - create_config
        - create_downloader
        - create_reader

::: gigaspatial.handlers.base.config.BaseHandlerConfig
    options:
      show_source: false
      members:
        - get_search_key
        - get_relevant_data_units_by_key
        - get_relevant_data_units
        - extract_search_geometry
        - get_search_geometry
        - get_data_unit_path
        - get_stored_paths
        - get_data_unit_paths
        - is_unit_available

::: gigaspatial.handlers.base.downloader.BaseHandlerDownloader
    options:
      show_source: false
      members:
        - download
        - download_data_units
        - download_data_unit

::: gigaspatial.handlers.base.reader.BaseHandlerReader
    options:
      show_source: false
      members:
        - load
        - load_from_paths
        - resolve_source_paths

::: gigaspatial.handlers.base.options.ResolveOptions
    options:
      show_source: false

::: gigaspatial.handlers.base.options.DownloadOptions
    options:
      show_source: false

::: gigaspatial.handlers.base.options.ReadOptions
    options:
      show_source: false
