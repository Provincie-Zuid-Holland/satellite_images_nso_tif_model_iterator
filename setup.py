from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tif_model_iterator",  # Replace with your own username
    version="1.0.0",
    author="Michael de Winter",
    author_email="m.r.dewinter88@live.nl",
    description="NSO Satellite Model implementer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Provincie-Zuid-Holland/satellite_images_nso_tif_model_iterator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "requests>=2.25.0",
        "objectpath>=0.6.1",
        "earthpy>=0.9.2",
        "Fiona>=1.8.13",
        "geopandas>=0.7.0",
        "rasterio>=1.1.3",
        "Shapely>=1.7.0",
    ],
)
