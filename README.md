# Introduction

This repository contains a .tif file (computer vision) model executer which means that it iterates and implements a (computer vision) model on every pixel and/or extracted image kernel in a .tif file.
As long as a model has a .predict function in python any model can be used in this executer from simple models to deep learning models.
For more on what image processing kernels are: [Here](<https://en.wikipedia.org/wiki/Kernel_(image_processing)>)

The iterative loop that loops over every pixel and/or extracted image kernel in a .tif, because of the long processing time, can be done in a multi processing loop, which in default this on.

Which means that it can't be run in a (jupyter) notebook interface, it has to be run from a terminal and it freezes your computer.

Also a databricks pyspark implementation is writing here which is based on this model iterator only written to take advantage of apache spark.
Found in the ./pspark folder.

# Installation

When working with 64x Windows and Anaconda for your python environment management execute the following terminal commands in order:

```sh
conda create -n satellite_images_nso_tif_model_iterator python=3.9 -y
conda activate satellite_images_nso_tif_model_iterator
pip install -r requirements.txt
```

Navigate to the [satellite-images-nso-datascience repository](https://github.com/Provincie-Zuid-Holland/satellite-images-nso-datascience) and then run:

```sh
pip install .
```

# Application

Copy `settings_example.py` and rename to `settings.py`. Change the variables in there as desired and then execute `example_iterator.py`.

# (Image Processing) Kernels.

The main functionality of this repository is to extract image kernels and the multiprocessing for loop for looping over all the pixels and/or image kernels in a given satellite .tif file to make predictions on them.

The following picture gives a illustration about how this extracting of kernels is done:
![Alt text](kernel_extract.png?raw=true "Title")

Here below we will have a code example about how this work. In this example we will use a Euclidean distance model to segment all the pixels in a .tif file into segments that are specified in a annotations file.

```python

from tif_model_iterator import tif_kernel_iterator

# This has to be run as a stand alone, so not in a python notebook.
if __name__ == '__main__':
    # Settings
    path_to_tif_file = "<PATH_TO_NSO_SAT_IMG.TIF>"
    out_path = "<PATH_TO_OUTPUT_FILE.shp"

    # The kernel size will be 32 by 32
    x_size_kernel = 32
    y_size_kernel = 32

    # Extract the x row and y column.
    x_row = 16
    y_row = 4757

    filename = "PATH/TO/MODEL/MODEL.sav"
    loaded_model = pickle.load(open(filename, 'rb')

    # Setup up a kernel generator for this .tif file.

    # The parameter are:
    # @param amodel: A prediction model with has to have a predict function and uses kernels as input.
    # @param output_location: Location where to writes the results to in .shp file.
    # @param aggregate_output: 50 cm is the default resolution but we can aggregate to 2m.
    # @param parts: break the .tif file in multiple parts, this has to be done since most extracted pixels or kernel don't fit # in memory.
    # @param begin_part: The part to begin with in order to skip certain parts.
    # @param bands: Which bands of the .tif file to use from the .tif file by default this will be all the bands.
    # @param fade: Whether to use fading kernels or not, fading is a term I coined to denouced for giving the centrale pixel # the most weight in the model while giving less weight the further the other pixels are in the model.
    # @param normalize_scaler: Whether to use a normalize/scaler on all the kernels or not, the input here so be a normalize/scaler function. You have to submit the normalizer/scaler as a argument here if you want to use a scaler, this has to be # a custom  class like nso_ds_normalize_scaler.
    # @param multiprocessing: Whether or not to use multiprocessing for loop for iterating across all the pixels.
    tif_kernel_generator = tif_kernel_iterator.tif_kernel_iterator_generator(path_to_tif_file, x_size_kernel, y_size_kernel)

    kernel = tif_kernel_generator.get_kernel_for_x_y(x_row,y_row )
    kernel.shape
    #output: (4, 32, 32)
    # This .tif file contains 4 dimensions in RGBI

    # Iterates and predicts all the pixels in a .tif file with a particular model and stores the dissolved results in the out_path file in a multiprocessing way. So this has to be run from a terminal.
    tif_kernel_generator.predict_all_output(loaded_model, out_path , parts = 3)
```

# Author

Michael de Winter

# Contact

Contact us at vdwh@pzh.nl
