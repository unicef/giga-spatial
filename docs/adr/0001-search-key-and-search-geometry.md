# Separate the search key from the search geometry in dataset handlers

Handlers resolve a source in two ways that used to share one method, `extract_search_geometry`: finding the data units to fetch, and finding the area to crop loaded data to. Half the handlers overrode it to return something that is not a geometry (a country code for WorldPop, OpenCellID and Google-Microsoft combined; a filter dict for HDX; the raw source for Microsoft Buildings), so cropping silently did nothing or broke for them, and the CRS of the geometry was never stated. We split the two: `get_search_key(source)` is the override point and returns whatever the dataset's unit lookup understands (by default the search geometry), while `extract_search_geometry` is owned by the base, always returns a geometry in the config's CRS, and is the only thing cropping uses.

## Considered options

- **Keep one method and let cropping type-check its result.** Rejected: a country code still cannot be cropped to, and the method's name keeps lying to the next handler author.
- **Let each handler override cropping.** Rejected: it spreads CRS handling across eleven handlers, which is how GHSL ended up with its own meaning of the `crs` option.

## Consequences

- Datasets published per country (`units_match_country = True`) skip the crop for country sources, since their files already match the country. Small leftovers are accepted: WorldPop rasters cover the country's bounding rectangle, OpenCellID files include towers near the borders of the country's mobile country codes, and Microsoft Buildings falls back to tiles overlapping the country when it is missing from the location mapping.
- Path sources are never cropped, and sources with no geometry (an HDX filter dict) raise when cropping is requested.
