# API Reference

Welcome to the GigaSpatial API reference documentation. This section provides detailed information about all public modules, classes, and functions in GigaSpatial.

## API Structure

GigaSpatial's API is organized into several main components:

### Core Module

The [core module](core.md) contains the fundamental classes and functions for spatial data processing:

- Spatial data structures
- Basic geometric operations
- Coordinate system management
- Data input/output operations

### Utilities

The [utilities module](utils.md) provides helper functions and tools for:

- Data conversion
- Format transformation
- Validation
- Common spatial operations

### Data Types

The [data types module](data-types.md) defines the fundamental data structures:

- Vector geometries
- Raster data structures
- Spatial reference objects
- Custom data types

## Using the API

### Installation

Since GigaSpatial is currently in development, you'll need to install it directly from the repository:

```bash
# Clone the repository
git clone https://github.com/unicef/giga-spatial
cd giga-spatial

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,test]"
```

### Import Convention

Once installed, you can import GigaSpatial components. The recommended way is:

```python
# Import the main package
from gigaspatial import GigaSpatial

# Create a GigaSpatial instance
gs = GigaSpatial()

# For specific functionality
from gigaspatial.core import spatial_operations
from gigaspatial.utils import data_helpers
from gigaspatial.io import readers, writers
```

### Basic Usage Example

Here's a simple example of using GigaSpatial:

```python
from gigaspatial import GigaSpatial
from gigaspatial.io import readers

# Initialize GigaSpatial
gs = GigaSpatial()

# Load some spatial data
data = readers.read_geojson("path/to/your/data.geojson")

# Perform spatial analysis
processed_data = gs.process(data, operation="buffer", distance=1000)

# Save results
processed_data.save("output.geojson")
```

### Error Handling

GigaSpatial uses a hierarchy of custom exceptions for clear error handling:

```python
from gigaspatial.exceptions import (
    GigaSpatialError,  # Base exception class
    GeometryError,     # For geometric operation errors
    InputError,        # For invalid input data
    TransformError     # For coordinate transformation errors
)

try:
    result = gs.process(data)
except GeometryError as e:
    print(f"Geometry processing failed: {e}")
except InputError as e:
    print(f"Invalid input data: {e}")
```

### Type Hints

All public APIs include type hints for better IDE integration and code checking:

```python
from typing import List, Optional
from gigaspatial.types import Geometry, SpatialReference

def process_geometries(
    geometries: List[Geometry],
    sref: Optional[SpatialReference] = None,
    tolerance: float = 0.001
) -> List[Geometry]:
    """Process a list of geometries with optional spatial reference."""
    ...
```

## Version Compatibility

- API stability is guaranteed for all releases following semantic versioning
- Breaking changes are only introduced in major version updates
- Deprecation warnings are issued at least one minor version before removal
- Development versions can be installed using `pip install -e git+https://github.com/unicef/giga-spatial.git#egg=gigaspatial`

## Contributing

If you'd like to contribute to the API:

1. Read our [Contributing Guide](../contributing.md)
2. Follow our [Code Style Guide](../contributing/code-style.md)
3. Submit a pull request with your changes

## Need Help?

If you need assistance with the API:

- Check the examples in each module's documentation
- Look at the [Examples](../examples/index.md) section
- Create an issue on our [GitHub repository](https://github.com/unicef/giga-spatial)