# Configuration

The `gigaspatial` package uses a configuration file (`config.py`) to manage paths, API keys, and other settings. This guide explains how to configure the package to suit your needs.

---

## Using Environment Variables

The package can read configuration settings from an environment file (e.g., `.env`). Here’s an example of how to set up the `.env` file based on the `env_sample`:

```bash
# Paths for different data types
BRONZE_DIR=/path/to/your/bronze_tier_data
SILVER_DIR=/path/to/your/silver_tier_data
GOLD_DIR=/path/to/your/gold_tier_data
VIEWS_DIR=/path/to/your/views_data
ADMIN_BOUNDARIES_DIR=/path/to/your/admin_boundaries_data

# API keys and tokens
MAPBOX_ACCESS_TOKEN=your_mapbox_token_here
MAXAR_USERNAME=your_maxar_username_here
MAXAR_PASSWORD=your_maxar_password_here
MAXAR_CONNECTION_STRING=your_maxar_key_here
```

The `config.py` file will automatically read these environment variables and set the paths and keys accordingly.

---

## Setting Paths Manually

You can also set paths and keys manually in your code. Here’s an example:

```python
from gigaspatial.config import config

# Example: Setting custom data storage paths
config.set_path("bronze", "/path/to/your/bronze_tier_data")
config.set_path("gold", "/path/to/your/gold_tier_data")
config.set_path("views", "/path/to/your/views_data")
```

API keys and tokens should be set through environment variables.

---

## Verifying the Configuration

After setting up the configuration, you can verify it by printing the current settings:

```python
from gigaspatial.config import config

# Print all configuration settings
print(config)
```

---

## Troubleshooting

If you encounter issues with the configuration, consider the following:

- **Ensure `.env` File Exists**: Make sure the `.env` file is in the root directory of your project.
- **Check Environment Variables**: Verify that the environment variables are correctly set in the `.env` file.
- **Use Absolute Paths**: When setting paths manually, use absolute paths to avoid issues with relative paths.

---

## Next Steps

Once the configuration is set up, you can proceed to the [Data Handling Guide](data-handling/downloading.md) to start using the `gigaspatial` package.

---

[Back to User Guide](../index.md)