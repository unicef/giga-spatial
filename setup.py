from setuptools import setup, find_packages
import os
import re

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]


def read_version():
    with open(os.path.join("gigaspatial", "__init__.py")) as f:
        content = f.read()
    return re.search(r'__version__ = "(.*)"', content).group(1)


setup(
    name="giga-spatial",
    version=read_version(),
    author="Utku Can Ozturk",
    author_email="utkucanozturk@gmail.com",
    description="A package for spatial data download & processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        "gigaspatial",
        "spatial",
        "geospatial",
        "gis",
        "remote sensing",
        "data processing",
        "download",
        "openstreetmap",
        "osm",
        "ghsl",
        "grid",
        "point of interest",
        "POI",
        "raster",
        "vector",
        "school connectivity",
        "unicef",
        "giga",
        "mapping",
        "analysis",
        "python",
    ],
    url="https://github.com/unicef/giga-spatial",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    license="AGPL-3.0-or-later",
    license_files=("LICENSE",),
)
