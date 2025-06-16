# Installation Guide

This guide will walk you through the steps to install the `gigaspatial` package on your system. The package is compatible with Python 3.10 and above.

## Prerequisites

Before installing `gigaspatial`, ensure you have Python installed on your system. You can check your Python version by running:

```bash
python --version
```

If Python is not installed, you can download it from the [official Python website](https://www.python.org/downloads/).

## Installing from PyPI

The easiest way to install `gigaspatial` is directly from PyPI using pip:

```bash
pip install giga-spatial
```

This will install the latest stable version of the package along with all its dependencies.

## Installing from Source

If you need to install a specific version or want to contribute to the development, you can install from the source:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/unicef/giga-spatial.git
   cd giga-spatial
   ```

2. **Install the Package**:
   ```bash
   pip install .
   ```

### Installing in Development Mode

If you plan to contribute to the package or modify the source code, you can install it in development mode. This allows you to make changes to the code without reinstalling the package:

```bash
pip install -e .
```

## Installing Dependencies

The package dependencies are automatically installed when you install `gigaspatial`. However, if you need to install them manually, you can use:

```bash
pip install -r requirements.txt
```

## Verifying the Installation

After installation, you can verify that the package is installed correctly by running:

```bash
python -c "import gigaspatial; print(gigaspatial.__version__)"
```

This should print the version of the installed package.

## Troubleshooting

If you encounter any issues during installation, consider the following:

- **Ensure `pip` is up-to-date**:
  ```bash
  pip install --upgrade pip
  ```

- **Check for conflicting dependencies**: If you have other Python packages installed that might conflict with `gigaspatial`, consider using a virtual environment.

- **Use a Virtual Environment**: To avoid conflicts with other Python packages, you can create a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
  pip install giga-spatial  # or pip install . if installing from source
  ```

---

### Next Steps

Once the installation is complete, you can proceed to the [Quick Start Guide](quickstart.md) to begin using the `gigaspatial` package.