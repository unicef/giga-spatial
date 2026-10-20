# Grids

A grid system partitions geography into cells. A generator maps datasets onto those cells. `H3ViewGenerator` uses `H3Hexagons`. You only need these classes when you want the cells themselves.

```python
from gigaspatial import H3Hexagons, S2Cells, MercatorTiles
from gigaspatial.grid import CountryH3Hexagons
```

::: gigaspatial.grid.h3.H3Hexagons
    options:
      show_source: false

::: gigaspatial.grid.s2.S2Cells
    options:
      show_source: false

::: gigaspatial.grid.mercator_tiles.MercatorTiles
    options:
      show_source: false
