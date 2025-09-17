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
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Telecommunications Industry",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Homepage": "https://github.com/unicef/giga-spatial",
        "Documentation": "https://unicef.github.io/giga-spatial/",
        "Source": "https://github.com/unicef/giga-spatial",
        "Issue Tracker": "https://github.com/unicef/giga-spatial/issues",
        "Discussions": "https://github.com/unicef/giga-spatial/discussions",
        "Changelog": "https://unicef.github.io/giga-spatial/changelog",
    },
    python_requires=">=3.10",
    install_requires=requirements,
    license="AGPL-3.0-or-later",
    license_files=("LICENSE",),
)
