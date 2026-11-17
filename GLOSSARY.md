# GigaSpatial

GigaSpatial acquires published geospatial datasets into a data store you control, loads them for a source, and processes them. A generator maps a dataset onto points of interest or zones. That mapping is one use of the data, not the purpose of the library.

## Language

**Dataset**:
A named geospatial product the library can acquire, such as a population grid or a set of building footprints.
_Avoid_: source data, layer

**Source**:
The place a dataset is requested for, or an explicit list of its stored files. A place is a country, a geometry, points, or a frame.
_Avoid_: source data, layer, area of interest

**Data store**:
The place a handler keeps a dataset's files. It may be local or remote.
_Avoid_: database, client

**Client**:
A connection used to query a database, a warehouse, or a web API. A fetcher or a handler may use one; the client itself is not a dataset.
_Avoid_: data store, fetcher

**Handler**:
A dataset published as files. Its files are kept in a data store and loaded for a source.
_Avoid_: fetcher, downloader

**Fetcher**:
A dataset served as a query, from an API or a database. The result is records, or a road network.
_Avoid_: handler

**Spatial network**:
Places as nodes with coordinates, joined by edges that carry a length or another weight. A road network is one kind; points joined to their nearest neighbours are another.
_Avoid_: graph, coordinate graph, graph engine

**Road network**:
The spatial network of a place's roads, with nodes where roads meet or end and edges along the roads between them. It is kept and processed after it is fetched; the fetcher only acquires it.
_Avoid_: road graph, street network, OSM graph

**Extract**:
An OpenStreetMap file covering one region, such as a country or one of its regions, as published by Geofabrik. A road network built from an extract covers the whole extract.
_Avoid_: dump, planet file, PBF

**Insertion**:
Adding points of interest to a spatial network as nodes, each joined to its nearest road by a connection edge, splitting the road where the connection meets it.
_Avoid_: merging, snapping, attaching, joining

**Connecting subnetwork**:
The part of a spatial network needed to join a set of points, reduced to those points and the places where the paths between them branch.
_Avoid_: cleaned graph, splitter graph

**Connection planner**:
Plans how to connect terminals, such as schools, to an existing network, such as fibre, along a spatial network at the smallest shared cost. The existing network counts as one root, so a terminal may connect directly or through other terminals.
_Avoid_: fibre planner, network optimiser, solver

**Connection plan**:
The result of a connection planner: the edges chosen, the connecting subnetwork they form, each terminal's status and the part of the existing network it reaches, and how far from the optimum the plan can be at most.
_Avoid_: solution, design, layout

**Root**:
A node of a spatial network that is already part of the existing network a connection planner connects terminals to, such as a fibre node. All roots count as one, already connected to each other. Not a source, which is the place a dataset is requested for.
_Avoid_: source, anchor, hub

**Origin**:
A place from which distances, routes, or links are measured in a spatial network, such as a school seeking a clinic or a fibre node. Not a source, which is the place a dataset is requested for.
_Avoid_: source, left, from-node

**Target**:
A candidate place that origins reach or link to in a spatial network, such as a clinic, a tower, or a fibre node.
_Avoid_: right, match, to-node

**Distance table**:
A long table of reachable origin-to-target pairs and the network distance between them, used as the cost input to accessibility and facility-location models.
_Avoid_: distance matrix, matches dataframe, cost matrix

**Image downloader**:
Saves imagery for places into a data store, from an API. The result is image files, not records, and there is no source to load.
_Avoid_: fetcher, handler

**Generator**:
An operation that maps datasets onto points of interest or zones.
_Avoid_: view

**View**:
Points of interest or zones with attributes from datasets attached. That attachment is enrichment.
_Avoid_: generator, a loaded dataset

**Enrichment**:
The attachment of dataset attributes to points of interest or zones. The result is a view.
_Avoid_: the name of the library, a dataset that has only been loaded

**Point of interest**:
A location analysed in its own right, such as a school, clinic, or cell site. A view attaches dataset attributes to points of interest.
_Avoid_: site, location, feature, object

**Zone**:
A named area that dataset attributes are summarised into, such as an administrative boundary or a grid cell.
_Avoid_: area, region, polygon, layer

**Zonal aggregation**:
Summarising a dataset's values within each zone, for example a sum of population or a count of buildings. It is one kind of enrichment.
_Avoid_: zonal stats (for the general idea), spatial join

**Grid system**:
A scheme that divides geography into discrete cells used as zones, such as H3, S2, or Mercator tiles.
_Avoid_: tiling, index (on its own)

**Administrative boundary**:
A political or statistical boundary that defines a reporting unit, such as a country, region, or district. Used as a zone.
_Avoid_: admin area, shape, region

**Building footprint**:
The polygon outline of a building, as published in a dataset such as Google or Microsoft buildings.
_Avoid_: building polygon, structure


**Reader**:
A component that loads stored files of a given format from a data store into memory.
_Avoid_: loader, parser

**Processor**:
A component that transforms loaded data or performs a spatial operation on it.
_Avoid_: engine, transformer

**Data unit**:
The smallest piece of a dataset that is fetched as a whole, such as one tile, one country file, or one global file. Anything that changes which file is fetched is part of its identity.
_Avoid_: chunk, part, tile (unless it is one)

**Search key**:
What a request's area of interest resolves to when looking up data units: a country code, a resource filter, a geometry, or nothing for a dataset published as one global file. Not necessarily a geometry.
_Avoid_: search geometry, query

**Search geometry**:
The geometry of a request's area of interest, used to crop loaded data. Always a geometry, even when data units are looked up by country code or filter.
_Avoid_: search key, bounding box (unless it is one)

**Stored file**:
A file as it exists in the data store after fetching, as opposed to the remote file it came from. One data unit may produce several stored files, for example when an archive is extracted.
_Avoid_: downloaded file, cache file
