# Roadmap

This page says what the project is aiming at. It has no dates. A line here is not a promise to ship. What has shipped is in the [changelog](../changelog.md).

## More datasets

Publish more datasets through the library, including climate data. The library acquires a dataset either by persisting files and loading them for a source, or by returning the result in memory.

## Documentation

Finish the documentation site: the tutorial, the user guide, and the API reference.

## Agents

Two stages.

An MCP server whose tools run GigaSpatial. An agent calls a tool, such as population for a country or the nearest buildings to a set of points. The library downloads what is missing, runs the operation, and returns a view or a table. The model sees that result. It does not see the files.

Later, retrieval sits in front of those tools. The agent retrieves which dataset and which operation fit the question. The tool runs, and the computed result is what enters the context. The rasters are not embedded.
