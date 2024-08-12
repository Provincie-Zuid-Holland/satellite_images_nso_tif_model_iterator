# Introduction

This repository contains a .tif file  model inferencer which means that it iterates and implements a (computer vision) model on every pixel and/or extracted image kernel in a .tif file.
As long as a model has a .predict function in python any model can be used in this executer from simple models to deep learning models.
For more on what image processing kernels are: [Here](<https://en.wikipedia.org/wiki/Kernel_(image_processing)>)

!Currently only pixels are supported!

# Installation

When working with 64x Windows and Anaconda for your python environment management execute the following terminal commands in order:

```sh
conda create -n satellite_images_nso_tif_model_iterator python=3.12 -y
conda activate satellite_images_nso_tif_model_iterator
pip install -r requirements.txt
```

Navigate to the [satellite-images-nso-datascience repository](https://github.com/Provincie-Zuid-Holland/satellite-images-nso-datascience) and then run:

```sh
rebuild.bat
```


# Application and example

Copy `settings_example.py` and rename to `settings.py`. Change the variables in there as desired and then execute `example_iterator.py`.


```python

import settings
from satellite_images_nso_tif_model_iterator.filenames.file_name_generator import OutputFileNameGenerator
from satellite_images_nso_tif_model_iterator.tif_model_iterator import TifModelIteratorGenerator


if __name__ == "__main__":
    filename = settings.MODEL_PATH
    tif_file = settings.TIF_FILE_INPUT
    output_path = settings.OUTPUT_PATH

    loaded_model = pickle.load(open(filename, "rb"))
    print("Loaded model: "+filename.split("/")[-1])

    print("Implementing model on: "+tif_file)

    # Generates a output name
    output_file_name_generator = OutputFileNameGenerator(
            output_path=output_path, output_file_name= output_path + tif_file.split("/")[-1].replace(".tif", ".geojson")
        )

    # Initialize the iterator
    nso_tif_kernel_iterator_generator = (
            tif_kernel_iterator.TifModelIteratorGenerator(
                path_to_tif_file=tif_file,
                model=loaded_model["model"],
                output_file_name_generator=output_file_name_generator,
                parts=4,
                normalize_scaler=loaded_model["scaler"],
                column_names=["r", "g", "b", "n", "e", "d", "ndvi", "re_ndvi"],
                do_all_parts = True
            )
    )

    # Run the prediction.
    nso_tif_kernel_iterator_generator.predict_all_output()


```
# Performance

We have encountered issue's with writing to .shp files on high resolutions.
Better to always use .geojson format.

# Author

Michael de Winter, Pieter Kouyzer

# Contact

Contact us at vdwh@pzh.nl
