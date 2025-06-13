<div style="padding-left: 20px; padding-right: 10px;">
<a href="https://giga.global/">
    <img src="https://s41713.pcdn.co/wp-content/uploads/2018/11/2020.05_GIGA-visual-identity-guidelines_v1-25.png" alt="Giga logo" title="Giga" align="right" height="60" style="padding-top: 10px;"/>
</a>

# GigaSpatial

## About Giga

Giga is a UNICEF-ITU initiative to connect every school to the Internet and every young person to information, opportunity and choice. 
Giga maps schools' Internet access in real time, creates models for innovative financing, and supports governments contracting for connectivity. 


## About Giga Spatial

**GigaSpatial** is a Python package developed as part of the Giga Applied Science Team to handle geospatial data efficiently. It provides tools for downloading, processing, and analyzing geospatial data, enabling users to work with datasets such as OpenStreetMap (OSM), Global Human Settlement Layer (GHSL), Microsoft Global Buildings, Google Open Buildings, and more. The package is designed to support Giga's mission by providing robust geospatial capabilities for mapping and analyzing school connectivity.

### Key Features
- **Data Downloading**: Download geospatial data from various sources including GHSL, Microsoft Global Buildings, Google Open Buildings, OpenCellID, and HDX datasets.
- **Data Processing**: Process and transform geospatial data, such as GeoTIFF files and vector data, with support for compression and efficient handling.
- **View Generators**: 
  - Enrich spatial context with POI (Point of Interest) data
  - Support for raster point sampling and zonal statistics
  - Area-weighted aggregation for polygon-based statistics
- **Grid System**: Create and manipulate grid-based geospatial data for analysis and modeling.
- **Data Storage**: Flexible storage options with both local and cloud (ADLS) support.
- **Configuration Management**: 
  - Centralized configuration via environment variables or `.env` file
  - Easy setup of API keys and paths

### Supported Datasets

The `gigaspatial` package supports data from the following providers:

<div align="center">
    <img src="https://raw.githubusercontent.com/unicef/giga-spatial/main/docs/assets/datasets.png" alt="Dataset Providers" style="width: 75%; height: auto;"/>
</div>

---


## Installation Guide

This guide will walk you through the steps to install the `gigaspatial` package on your system. The package is compatible with Python 3.7 and above.

---

## Prerequisites

Before installing `gigaspatial`, ensure you have Python installed on your system.  
You can check your Python version by running:

```bash
python --version
````

If Python is not installed, you can download it from the [official Python website](https://www.python.org/).

---

## Installing via pip

You can install the `gigaspatial` package directly from the source using `pip`.

### Clone the Repository

```bash
git clone https://github.com/unicef/giga-spatial.git
cd giga-spatial
```

### Install the Package

```bash
pip install .
```

This will install `gigaspatial` along with its dependencies.

---

## Installing in Development Mode

If you plan to contribute to the package or modify the source code, install it in development mode:

```bash
pip install -e .
```

---

## Installing Dependencies

Dependencies are installed automatically with the package.
To install them manually:

```bash
pip install -r requirements.txt
```

---

## Verifying the Installation

Run this command to verify installation:

```bash
python -c "import gigaspatial; print(gigaspatial.__version__)"
```

You should see the installed version printed.

---

## Troubleshooting

If you encounter issues during installation:

### 1. Ensure pip is up to date

```bash
pip install --upgrade pip
```

### 2. Check for dependency conflicts

Use a virtual environment to isolate the installation:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install .
```

```

Let me know if you want this saved as a file.
```
---

### View Generators

The **view generators** in GigaSpatial are designed to enrich the spatial context of school locations and map data into grid or POI locations. This enables users to analyze and visualize geospatial data in meaningful ways.

### Key Capabilities
1. **Spatial Context Enrichment**:
   - Automatic attribution of geospatial variables to school locations
   - Contextual layers for environmental, infrastructural, and socioeconomic factors
   - Multi-resolution data availability for different analytical needs
   - Support for both point and polygon-based enrichment

2. **Mapping to Grid or POI Locations**:
   - Map geospatial data to grid cells for scalable analysis
   - Map data to POI locations for detailed, location-specific insights
   - Support for chained enrichment using multiple datasets
   - Built-in support for administrative boundary annotations

---

## Why Use GigaSpatial?
- **Efficient Geospatial Handling**: Streamline the process of downloading, processing, and analyzing geospatial data
- **Scalable Analysis**: Map data to grid cells or POI locations for both scalable and detailed insights
- **Open Source**: Contribute to and benefit from a collaborative, transparent, and innovative geospatial toolset
- **Modern Architecture**: Built with maintainability and extensibility in mind, featuring:
  - Base handler orchestration for unified lifecycle management
  - Dedicated reader classes for major datasets
  - Modular source resolution for flexible data access
  - Comprehensive error handling and logging

## Why Open Source?  

At Giga, we believe in the power of open-source technologies to accelerate progress and innovation. By keeping our tools and systems open, we:  
- Encourage collaboration and contributions from a global community.  
- Ensure transparency and trust in our methodologies.  
- Empower others to adopt, adapt, and extend our tools to meet their needs.  

## How to Contribute  

We welcome contributions to our repositories! Whether it's fixing a bug, adding a feature, or improving documentation, your input helps us move closer to our goal of universal school connectivity.  

### Steps to Contribute  
1. Fork the repository you'd like to contribute to.  
2. Create a new branch for your changes.  
3. Submit a pull request with a clear explanation of your contribution. 

To go through the contribtution guidelines in detail you can visit the following link. 

[Click here for the detailed Contribution guidelines](https://github.com/unicef/giga-spatial/blob/main/CONTRIBUTING.md)

---

## Code of Conduct  

At Giga, we're committed to maintaining an environment that's respectful, inclusive, and harassment-free for everyone involved in our project and community. We welcome contributors and participants from diverse backgrounds and pledge to uphold the standards.

[Click here for the detailed Code of Conduct.](https://github.com/unicef/giga-spatial/blob/main/CODE_OF_CONDUCT.md)

---

## Stay Connected  

To learn more about Giga and our mission, visit our official website: [Giga.Global](https://giga.global)

## Join Us

Join us in creating an open-source future for education! 🌍  
