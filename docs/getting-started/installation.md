# Installation Guide

## Prerequisites

Before installing GigaSpatial, ensure you have the following prerequisites:

- Python 3.10 or higher

## Installation Methods

Since GigaSpatial is currently in development, you'll need to install it directly from the repository:

```bash
# Clone the repository
git clone https://github.com/unicef/giga-spatial
cd giga-spatial

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```


## Dependencies

GigaSpatial requires the following main dependencies:

- numpy
- pandas
- geopandas
- rasterio
- shapely
- pyproj

These dependencies will be automatically installed when you install GigaSpatial using pip.

## Verifying Installation

To verify that GigaSpatial is installed correctly, you can run:

```python
import gigaspatial
```

## Troubleshooting

If you encounter any issues during installation:

1. Ensure your Python version is compatible
2. Update pip to the latest version: `pip install --upgrade pip`
3. Check our [GitHub Issues](https://github.com/unicef/giga-spatial/issues) for known problems
4. If the problem persists, please [open a new issue](https://github.com/unicef/giga-spatial/issues/new) 