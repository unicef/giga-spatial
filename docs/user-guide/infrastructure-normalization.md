# Infrastructure files

Partner files, such as a KMZ of fibre nodes, are not a published dataset. This guide reads a file you already have and validates it as transmission nodes.

`TransmissionNodeTable.from_file` reads the file, cleans column names and coordinates, and validates each row. Cleaning is on by default.

```python
from gigaspatial.core.schemas.transmission_node import TransmissionNodeTable

nodes = TransmissionNodeTable.from_file("partner_nodes.gpkg")
print(nodes.to_dataframe().head())
```

`read_dataset` is the same reader, without validation. Use it when you want the table as the partner sent it.

```python
from gigaspatial import read_dataset

raw = read_dataset("partner_nodes.gpkg")
```

To clean a table you already loaded, use `TransmissionNodeProcessor` and then `from_dataframe`:

```python
from gigaspatial.core.schemas.transmission_node import (
    TransmissionNodeProcessor,
    TransmissionNodeTable,
)

cleaned = TransmissionNodeProcessor().process(raw, country="RWA")
nodes = TransmissionNodeTable.from_dataframe(cleaned, clean=False)
```

The same pattern exists for other partner tables in `gigaspatial.core.schemas`, including wireless sites, buildings, and admin boundaries. A validated table can be passed to `PoiViewGenerator` as the points in [Points and nearest buildings](points-and-buildings.md).
