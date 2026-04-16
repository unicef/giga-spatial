# Settlement Characterization

To design efficient connectivity solutions, it is essential to distinguish between urban centers and remote rural outposts. GigaSpatial leverages the Global Human Settlement Layer (GHSL) to characterize any POI based on its built environment.

## The "Why": Building vs. Population Scenarios

- **GHSL** provides a standardized, global view of the earth's surface, classified into different settlement types (SMOD - Settlement Model).
- **PoiViewGenerator** allows you to "drape" your POIs over this global layer to inherit their local settlement classification.

Integrating these two allows for automated stratification of your data (e.g., analyzing school connectivity costs specifically for *Rural* vs *Semi-Dense Urban* areas).

## The Workflow

### 1. Initialize the GHSL Handler

The `GHSLDataHandler` manages the downloading and reading of global settlement rasters.

```python
from gigaspatial import GHSLDataHandler

# Initialize handler for the Settlement Model (SMOD) product
ghsl_handler = GHSLDataHandler(product="GHS_SMOD", year=2020)
```

### 2. Identify Urban/Rural contexts

We use the `PoiViewGenerator` to map these global classifications to our specific points of interest.

```python
from gigaspatial import PoiViewGenerator

# Assume gdf_schools is our GeoDataFrame of school locations
view = PoiViewGenerator(gdf_schools)

# Enrich the schools with GHSL SMOD categories
# This resolves to the SMOD (Settlement Model) for each point
enriched_schools = view.map_smod(stat="median")
```

## Understanding SMOD Classifications

GHSL SMOD data provides values that correspond to standard settlement types:

| Value | Classification | Context |
| :--- | :--- | :--- |
| 30 | **Urban Centre** | High-density metropolitan areas |
| 23 | **Dense Urban Cluster** | Suburban or high-density outskirts |
| 11 | **Small Settlement** | Rural villages and dispersed clusters |
| 10 | **Rural Grid Cell** | Sparse population areas |

## Rationale for this Combination

Manual classification of thousands of sites is impossible. By combining a **Global Handler** (GHSL) with a **Local View Generator**, GigaSpatial allows you to perform "Site Stratification" at the click of a button.

This becomes especially powerful when combined with WorldPop data to calculate not just *where* a settlement is, but *how many people* live in the surrounding urban/rural cluster.
