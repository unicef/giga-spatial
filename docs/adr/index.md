# Design decisions

These records explain decisions that are hard to reverse or surprising without context: what was decided, why, and which alternatives were rejected. Read them before proposing to change one of these areas. A new decision gets the next number.

| Record | Decision |
| --- | --- |
| [0001](0001-search-key-and-search-geometry.md) | Handlers separate the search key, which finds the files to fetch, from the search geometry, which crops the loaded data. |
| [0002](0002-handler-base-in-the-api-reference.md) | The handler base is documented in the API reference, but it is not a stable API for other packages. |
| [0003](0003-public-import-paths.md) | Users import from the top level or a public subpackage, which load lazily. Every other module path is internal. |
