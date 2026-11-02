# OSM extracts are stored by date, with uncropped networks beside them

`OSMRoadsHandler` stores each Geofabrik extract under the date of the extract, in Geofabrik's own folder structure: `osm/geofabrik/2026-10-27/africa/rwanda.osm.pbf`, with a country's sub-regions under the country, as in `.../south-america/brazil/nordeste.osm.pbf`. The road network built from an extract is saved beside it, one folder per network type and simplification (`rwanda.networks/drive_service-simplified/`), and always covers the whole extract with every connected part. Cropping to a source and keeping the largest connected part happen only at load. A network cropped for one source and saved next to the extract would otherwise be taken for the whole extract on the next load.

## Considered options

- **Country folders first, dates below.** Rejected: a dated snapshot is what users keep, compare, or delete together, and Geofabrik deletes its daily files after about a week, so the date is the first thing a user needs to see.
- **Store only extracts and build the network on every load.** Rejected: building a country takes a minute or more, and loading a saved network takes under a second.
- **Store the network cropped to each source.** Rejected for the reason above; the stored network must stand for the file it came from.

## Consequences

- The base's availability check covers the extract only. The network folder is a cache derived from the extract's path; a missing network is built from the extract, never downloaded again.
- Old dates are never deleted automatically, and the guide states the space extracts take (Nigeria is 680 MiB, Brazil 2 GiB).
- Reproducing an old result needs the stored extract, since Geofabrik keeps daily files for about a week.
