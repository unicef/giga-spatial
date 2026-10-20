# Public import paths and lazy loading

Users import from three tiers. The top level, `from gigaspatial import X`, holds what a script creates: every handler, fetcher, image downloader, view generator, grid, data store, and client, plus `TifProcessor`, `read_dataset`, `write_dataset`, and the main `processing.geo` helpers. The public subpackages hold the building blocks: `gigaspatial.handlers` (each handler's config, downloader, and reader, plus `AdminBoundaries` and the GeoRepo and WorldPop clients), `gigaspatial.fetchers`, `gigaspatial.image_downloaders`, `gigaspatial.generators`, `gigaspatial.grid`, `gigaspatial.processing`, `gigaspatial.core.io`, `gigaspatial.core.http`, and `gigaspatial.core.schemas`. Every other module path is internal and may move between minor releases, including `gigaspatial.core`, `gigaspatial.config`, and the handler base ([ADR 0002](0002-handler-base-in-the-api-reference.md)). The packages follow the glossary: a fetcher never lives in `handlers`, whatever its class name.

The top level, `gigaspatial.handlers`, `gigaspatial.fetchers`, and `gigaspatial.processing.elevation` load their names lazily (PEP 562). Importing every dataset eagerly made `import gigaspatial` take about six seconds and load every optional integration that was installed, including Earth Engine, Snowflake, and BigQuery. Lazy loading keeps the flat top-level imports and makes a bare import take milliseconds.

## Considered options

- **Eager imports in `__init__.py`.** Rejected: the cost above, paid by every user for datasets they never touch.
- **The `lazy_loader` package (SPEC 1).** Rejected: one more dependency, and `.pyi` stubs that have to be kept in sync with the code. `gigaspatial._lazy.attach` is a few lines, and each lazy `__init__.py` repeats its names in an `if TYPE_CHECKING:` block for IDEs, type checkers, and mkdocstrings. A test checks that the two lists match.
- **Keep the `gigaspatial.io`, `gigaspatial.elevation`, and `gigaspatial.algorithms` aliases, with or without a deprecation warning.** Rejected: they were made by writing to `sys.modules`, which type checkers and IDEs cannot see, and `io` in `__all__` replaced the standard library's `io` on `from gigaspatial import *`. They were removed in 0.10.0 without a shim. The library is before 1.0.

## Consequences

- Adding a public class means adding it to the dictionary passed to `attach`, to the `TYPE_CHECKING` block, or to the subpackage's `__all__`.
- The API reference renders each class from its defining module but shows only the class name, and each page opens with the imports to use.
