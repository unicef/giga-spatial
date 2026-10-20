# API reference

These pages list the classes and functions a caller imports. Each entry is one class or function, generated from its docstring.

Each page opens with the imports to use. Classes you create in a script, such as handlers, fetchers, generators, grids, and data stores, import from `gigaspatial`. Their building blocks, such as a handler's config, downloader, and reader, import from the subpackage: `gigaspatial.handlers`, `gigaspatial.fetchers`, `gigaspatial.image_downloaders`, `gigaspatial.generators`, `gigaspatial.grid`, `gigaspatial.processing`, `gigaspatial.core.io`, `gigaspatial.core.http`, or `gigaspatial.core.schemas`. Any other module path is internal and may change between releases.

The [handler base](handler-base.md) is last. It is for adding a dataset to this library, and it is the page that explains a search key and a search geometry.

## Callers

- [Handlers](handlers.md): datasets published as files
- [Fetchers](fetchers.md): datasets served as a query
- [Generators](generators.md)
- [Grids](grid.md)
- [Data stores](data-stores.md)
- [Clients](clients.md)
- [Boundaries](boundaries.md)
- [Earth Engine](earth-engine.md)
- [Processing](processing.md)
- [Image downloaders](image-downloaders.md)

## Adding a dataset

- [Handler base](handler-base.md)
