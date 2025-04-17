# Installation Guide

This guide will walk you through the steps to install the `gigaspatial` package on your system. The package is compatible with Python 3.7 and above.

## Prerequisites

Before installing `gigaspatial`, ensure you have Python installed on your system. You can check your Python version by running:

```bash
python --version
```

If Python is not installed, you can download it from the [official Python website](https://www.python.org/downloads/).

## Installing via pip

You can install the `gigaspatial` package directly from the source using `pip`. Follow these steps:

1. **Clone the Repository** (if you haven't already):
   ```bash
   git clone https://github.com/unicef/giga-spatial.git
   cd giga-spatial
   ```

2. **Install the Package**:
   Run the following command in your terminal to install the package:
   ```bash
   pip install .
   ```

   This command will install `gigaspatial` along with its dependencies.

## Installing in Development Mode

If you plan to contribute to the package or modify the source code, you can install it in development mode. This allows you to make changes to the code without reinstalling the package. To install in development mode, run:

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
  pip install .
  ```

---

### Next Steps

Once the installation is complete, you can proceed to the [Quick Start Guide](quickstart.md) to begin using the `gigaspatial` package.