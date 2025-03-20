from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="gigaspatial",
    version="0.3.1",
    author="Utku Can Ozturk",
    author_email="utkucanozturk@gmail.com",
    description="A package for spatial data download & processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/unicef/giga-spatial",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
