# Examples Gallery

Welcome to the GigaSpatial Examples Gallery. This section provides a collection of practical examples and tutorials to help you understand how to use GigaSpatial effectively.

## Example Categories

### Basic Examples

The [Basic Examples](basic.md) section covers fundamental operations:

- Downloading, loading and saving spatial data
- Basic geometric operations
- Common data transformations

### Advanced Examples

The Advanced Examples section demonstrates more complex use cases:

- Complex spatial analysis
- Performance optimization
- Custom processing pipelines
- Advanced visualization techniques

### Use Cases

The Use Cases section shows real-world applications:

- Infrastructure mapping
- Demographic studies

## Interactive Examples

All examples are provided as both markdown documentation and Jupyter notebooks. You can:

1. Read through the examples online
2. Download and run the notebooks locally
3. Modify the code to suit your needs

## Running the Examples

To run these examples locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/unicef/giga-spatial
cd giga-spatial

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter notebook
jupyter notebook examples/
```

### Dependencies

The examples require additional dependencies that are specified in the `[examples]` extra. These include:

- `jupyter`: For running the notebook examples
- `geopandas`: For working with geospatial data
- `rasterio`: For raster data processing

You can install specific example dependencies based on your needs:

```bash
# For basic examples only
pip install -e ".[examples-basic]"

# For all examples including advanced ones
pip install -e ".[examples-full]"
```

## Contributing Examples

We welcome contributions to our example gallery! To contribute:

1. Follow our [Contributing Guide](../contributing.md)
2. Use our example template
3. Submit a pull request

## Need Help?

If you need assistance with the examples:

- Check our [User Guide](../user-guide/index.md)
- Visit our [API Reference](../api/index.md)
- Create an issue on our [GitHub repository](https://github.com/unicef/giga-spatial) 