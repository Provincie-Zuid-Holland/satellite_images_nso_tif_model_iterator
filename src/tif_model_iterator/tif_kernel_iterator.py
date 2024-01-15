import glob
import itertools
import os
import random
import warnings
from multiprocessing import Pool
from timeit import default_timer as timer
from xml.dom import ValidationErr

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from rasterio.plot import show
from shapely.geometry import Polygon
from sklearn import preprocessing
from tqdm import tqdm

from src.filenames.file_name_generator import OutputFileNameGenerator
from src.tif_model_iterator.__nso_ds_output import dissolve_gpd_output

warnings.filterwarnings("ignore", category=UserWarning)

"""
    This code is used to extract image processing kernels from nso satellite images .tif images and execute a model on each of those kernels with multi processing for loop.

    For more information what kernels are: https://en.wikipedia.org/wiki/Kernel_(image_processing)

    Author: Michael de Winter, Jeroen Esseveld
"""


class tif_kernel_iterator_generator:

    """
    This class set up a .tif image in order to easily extracts kernel from it.
    With various parameters to control the size of the kernel.

    Fading, which means giving less weight to pixels other than the centre kernel, is also implemented here.

    Plus a multiprocessing for loop which iterates of each of these kernels extracted from the .tif file.

    """

    def __init__(self, path_to_tif_file: str, x_size: int = 1, y_size: int = 1):
        """

        Init of the nso tif kernel.

        @param path_to_file: A path to a .tif file.
        @param x_size: the x size of the kernel. For example if x and y are 32 you get a 32 by y kernel.
        @param y_size: the y size of the kernel. For example if x and y are 32 you get a x by 32 kernel.
        """

        dataset = rasterio.open(path_to_tif_file)
        meta = dataset.meta.copy()
        data = dataset.read()
        width, height = meta["width"], meta["height"]

        self.data = data

        self.bands = data.shape[0]

        self.path_to_tif_file = path_to_tif_file

        # TODO: Because of multiprocessing we can't stored rasterio datasets, multi processing does not support this.
        # self.dataset = dataset
        self.width = width
        self.height = height

        self.x_size = x_size
        self.x_size_begin = round(x_size / 2)
        self.x_size_end = round(x_size / 2)

        self.y_size = y_size
        self.y_size_begin = round(y_size / 2)
        self.y_size_end = round(y_size / 2)

        # Skipping using a kernel if the kernel size is 1 for beter performance.
        self.pixel_values = True if x_size == 1 and y_size == 1 else False

        self.sat_name = path_to_tif_file.split("/")[-1]

    def set_fade_kernel(self, fade_power=0.045, bands=0):
        """
        Creates a fading kernel based on the shape of the other kernels and different parameters.

        A fading kernel uses weights to give other pixel than the center pixel less 989/":weight in the prediction.

        @param fade_power: the power of the fade kernel.
        @param bands: the number bands that has to be faded.
        """
        if bands == 0:
            bands = self.data.shape[0]

        self.fade_kernel = np.array(
            [
                [
                    (1 - (fade_power * max(abs(idx - 15), abs(idy - 15))))
                    for idx in range(0, self.x_size)
                ]
                for idy in range(0, self.y_size)
            ]
        )
        self.fade_kernel = np.array([self.fade_kernel for id_x in range(0, bands)])

    def fadify_kernel(self, kernel):
        """

        Multiply a kernel with the fade kernel, thus fading it.

        A fading kernel uses weights to give other pixel than the center pixel less weight in the prediction.

        @param kernel: A kernel you which to  fade.
        @return: A kernel that is faded now.
        """
        return kernel * self.fade_kernel

    def normalize_tile_kernel(self, kernel):
        """
        Normalize image kernels with sklearn's normalize.

        @param kernel: a kernel to normalize

        """

        copy_kernel = np.zeros(shape=kernel.shape)
        for x in range(0, kernel.shape[0]):
            copy_kernel[x] = preprocessing.normalize(kernel[x])

        return copy_kernel

    def normalize_min_max(self, kernel):
        """

        Normalize tif file with min max scaler.
        @param kernel: a kernel to normalize

        """

        copy_kernel = np.zeros(shape=kernel.shape)
        for x in range(0, kernel.shape[0]):
            copy_kernel[x] = (
                (kernel[x] - np.min(kernel[x]))
                / (np.max(kernel[x]) - np.min(kernel[x]))
                * 255
            )

        return copy_kernel

    def percentage_cloud(self, initial_threshold=145, initial_mean=29.441733905207673):
        """

        Create mask from tif file on first band.

        @param kernel: a kernel to detect percentage clouds on
        @param initial_threshold: an initial threshold for creating a mask
        @param initial_mean: an initial pixel mean value of the selected band

        """

        # Make sure the blue band is used in the third array element.
        kernel = self.normalize_min_max(self.data[2])
        new_threshold = round(
            (initial_threshold * kernel.mean())
            / (initial_threshold * initial_mean)
            * initial_threshold,
            0,
        )
        print(new_threshold)
        copy_kernel = kernel.copy().copy()

        for x in range(len(kernel)):
            for y in range(len(kernel[x])):
                if kernel[x][y] == 0:
                    copy_kernel[x][y] = 1
                elif kernel[x][y] <= new_threshold:
                    if kernel[x][y] > 0:
                        copy_kernel[x][y] = 2

                else:
                    copy_kernel[x][y] = 3

        percentage = round((copy_kernel == 3).sum() / (copy_kernel == 2).sum(), 4)

        return percentage

    def unfadify_tile_kernel(self, kernel):
        """
        Unfade a kernel, for example to plot it again.

        A fading kernel uses weights to give other pixel than the center pixel less weight in the prediction.

        @param kernel: A faded kernel that can be unfaded.
        @return: A unfaded kernel.
        """
        return kernel / self.fade_kernel

    def get_pixel_value(self, index_x, index_y):
        """

        Extra method which for extracting only one pixel value in a kernel, should have a faster performance than get_kernel_for_x_y for a kernel size of 1.


        @param index_x: the x coordinate.
        @param index_y: the y coordinate.

        """
        if sum([band[index_x][index_y] for band in self.data]) == 0:
            raise ValueError("Center pixel is empty")
        else:
            return [band[index_x][index_y] for band in self.data]

    def get_kernel_for_x_y(self, index_x, index_y):
        """

        Get a kernel with x,y as it's centre pixel.
        Be aware that the x,y coordinates have to be in the same coordinate system as the coordinate system in the .tif file.

        @param index_x: the x coordinate.
        @param index_y: the y coordinate.
        @return a kernel with chosen size in the init parameters
        """

        if sum([band[index_x][index_y] for band in self.data]) == 0:
            raise ValueError("Center pixel is empty")
        else:
            spot_kernel = [
                [
                    k[index_y - self.x_size_end : index_y + self.x_size_begin]
                    for k in band[
                        index_x - self.y_size_end : index_x + self.y_size_begin
                    ]
                ]
                for band in self.data
            ]
            spot_kernel = np.array(spot_kernel)
            spot_kernel = spot_kernel.astype(int)
            return spot_kernel

    def get_x_y(self, x_cor, y_cor, dataset=False):
        """

        Get the x and y, which means the x row and y column position in the matrix, based on the x, y in the geography coordinate system.
        Needed to get a kernel for a specific x and y in the coordinate system.

        Due to multi processing we have to read in the rasterio data set each time.

        @param x_cor: x coordinate in the geography coordinate system.
        @param y_cor: y coordinate inthe geography coordinate system.
        @return x,y row and column position the matrix.
        """
        # TODO: Because of multi processing we have to read in the .tif every time.
        if isinstance(dataset, bool):
            index_x, index_y = rasterio.open(self.path_to_tif_file).index(x_cor, y_cor)

        else:
            index_x, index_y = dataset.index(x_cor, y_cor)

        return pd.Series({"rd_x": int(index_x), "rd_y": int(index_y)})

    def get_x_cor_y_cor(self, index_x, index_y, dataset=False):
        """
        Returns the geometry coordinates for index_x row and index_y column.

        @param index_x: the row.
        @param index_y: the column.
        """
        if isinstance(dataset, bool):
            index_x, index_y = rasterio.open(self.path_to_tif_file).index(
                index_x, index_y
            )

        else:
            index_x, index_y = dataset.xy(index_x, index_y)

        return pd.Series({"rd_x": int(index_x), "rd_y": int(index_y)})

    def func_multi_processing_get_kernels(self, input_x_y):
        """
        This function is used to do multiprocessing predicting.

        This needs to be done in a seperate function in order to make multiprocessing work.

        @param input_x_y: a array with the row and column for the to be predicted pixel.
        @return row and column and the predicted label in numbers.
        """
        try:
            kernel = (
                self.get_kernel_for_x_y(input_x_y[0], input_x_y[1])
                if self.pixel_values == False
                else self.get_pixel_value(input_x_y[0], input_x_y[1])
            )
            return kernel

        except ValueError as e:
            if str(e) != "Center pixel is empty":
                print(e)
            return [0, 0, 0]
        except Exception as e:
            print(e)
            return [0, 0, 0]

    def func_multi_processing_predict(self, input_x_y):
        """

        This function is used to predict input with a multiprocessing function.
        The model needs to heve a predict function in order to work.

        @param input_x_y: The input row for which to do predictions on.
        @return the prediction for the input values.

        """
        try:
            # TODO: Make the bands selected able
            coords = input_x_y[-1]
            label = self.model.predict([input_x_y[:-1]])[0]
            return [coords[0], coords[1], label]

        except ValueError as e:
            if str(e) != "Center pixel is empty":
                print(e)
            return [0, 0, 0]
        except Exception as e:
            print(e)
            return [0, 0, 0]

    def predict_all_output(
        self,
        amodel,
        output_file_name_generator: OutputFileNameGenerator,
        aggregate_output=True,
        parts=10,
        begin_part=0,
        bands=[1, 2, 3, 4, 5, 6],
        band_to_column_name={
            "band1": "r",
            "band2": "g",
            "band3": "b",
            "band4": "i",
            "band5": "ndvi",
            "band6": "height",
        },
        fade=False,
        normalize_scaler=False,
        multiprocessing=True,
    ):
        """
        A multiprocessing iterator which predicts all pixel in a raster .tif, based on there kernels.
        Multiprocessing on default is true  so this has to be run from a terminal.

        For specifically large .tif files, this file has to be divided into multiple parts.
        So it can fit into memory, parameters are here which can control that.

        TODO: Make logs for runs.

        @param amodel: A prediction model with has to have a predict function and uses kernels as input.
        @param output_file_name_generator: Generates desired filenames for output files
        @param aggregate_output: 50 cm is the default resolution but we can aggregate to 2m.
        @param parts: break the .tif file in multiple parts, this has to be done since most extracted pixels or kernel don't fit in memory.
        @param begin_part: The part to begin with in order to skip certain parts.
        @param bands: Which bands of the .tif file to use from the .tif file by default this will be all the bands.
        @param band_to_column_name: Band name to column name dictionary.
        @param fade: Whether to use fading kernels or not, fading is a term I coined to denouced for giving the centrale pixel the most weight in the model while giving less weight the further the other pixels are in the model.
        @param normalize_scaler: Whether to use a normalize/scaler on all the kernels or not, the input here so be a normalize/scaler function. You have to submit the normalizer/scaler as a argument here if you want to use a scaler, this has to be a custom  class like nso_ds_normalize_scaler.
        @param multiprocessing: Whether or not to use multiprocessing for loop for iterating across all the pixels.
        """
        # Set some variables for breaking the .tif in different part parts in order to save memory.
        total_height = self.get_height() - self.x_size

        height_parts = round(total_height / parts)
        begin_height = self.x_size_begin
        end_height = self.x_size_begin + height_parts

        total_width = self.get_width() - self.y_size

        height_parts = total_height / parts

        # Set some variables for multiprocessing.
        self.set_model(amodel)
        dataset = rasterio.open(self.path_to_tif_file)

        try:
            self.fade = amodel.get_fade()
            self.normalize = amodel.get_normalize()
        except:
            self.fade = fade
            self.normalize = normalize_scaler

        self.bands = bands

        # Divide the satellite images into multiple parts and loop through the parts, using parts reduces the amount of RAM required to run this process.
        # for x_step in tqdm(range(begin_part, parts)):
        for x_step in tqdm(range(20, 25)):
            print("-------")
            print("Part: " + str(x_step + 1) + " of " + str(parts))
            # Calculate the number of permutations for this part.
            permutations = list(
                itertools.product(
                    [x for x in range(begin_height, end_height)],
                    [
                        y
                        for y in range(
                            self.y_size_begin, self.get_width() - self.y_size_end
                        )
                    ],
                )
            )
            print("Total permutations this step: " + str(len(permutations)))

            # Init the multiprocessing pool.
            start = timer()

            if multiprocessing is True:
                p = Pool()
                seg_df = p.map(self.func_multi_processing_get_kernels, permutations)
                print(
                    "Pool kernel fetching finished in: "
                    + str(timer() - start)
                    + " second(s)"
                )
            else:
                seg_df = [
                    self.func_multi_processing_get_kernels(permutation)
                    for permutation in permutations
                ]

            seg_df = pd.DataFrame(
                seg_df, columns=["band" + str(band) for band in bands]
            )

            # Check if a normalizer or a  scaler has to be used.
            if normalize_scaler is not False:
                print("Normalizing/Scaling data")
                seg_df = normalize_scaler.transform(seg_df)

            # Select correct features for model
            seg_df = seg_df.rename(band_to_column_name, axis="columns")
            seg_df = seg_df[amodel.feature_names_in_]

            seg_df["permutation"] = permutations
            seg_df = seg_df.dropna()

            seg_df = seg_df.values
            del permutations

            start = timer()

            if multiprocessing is True:
                seg_df = p.map(self.func_multi_processing_predict, seg_df)
            else:
                seg_df = [
                    self.func_multi_processing_predict(permutation)
                    for permutation in seg_df
                ]

            print("Predicting finished in: " + str(timer() - start) + " second(s)")

            seg_df = pd.DataFrame(seg_df, columns=["x_cor", "y_cor", "label"])

            # Filter empty pixels.
            seg_df = seg_df[(seg_df["x_cor"] != 0) & (seg_df["y_cor"] != 0)]
            print("Number of used pixels for this step: " + str(len(seg_df)))

            # Get the coordinates for the pixel locations.
            seg_df["rd_x"], seg_df["rd_y"] = rasterio.transform.xy(
                dataset.transform, seg_df["x_cor"], seg_df["y_cor"]
            )

            print("Got coordinates for pixels: " + str(timer() - start) + " second(s)")

            seg_df = seg_df.drop(["y_cor", "x_cor"], axis=1)

            start = timer()
            if aggregate_output == True:
                seg_df["x_group"] = np.round(seg_df["rd_x"] / 2) * 2
                seg_df["y_group"] = np.round(seg_df["rd_y"] / 2) * 2
                seg_df = seg_df.groupby(["x_group", "y_group"]).agg(
                    label=("label", lambda x: x.value_counts().index[0])
                )
                print("Group by finished in: " + str(timer() - start) + " second(s)")

                start = timer()
                seg_df["rd_x"] = list(map(lambda x: x[0], seg_df.index))
                seg_df["rd_y"] = list(map(lambda x: x[1], seg_df.index))
                print("Labels created in: " + str(timer() - start) + " second(s)")

                seg_df = seg_df[["rd_x", "rd_y", "label"]]

            start = timer()

            # Make squares from the the pixels in order to make contected polygons from them.
            if multiprocessing is True:
                seg_df["geometry"] = p.map(
                    func_cor_square, seg_df[["rd_x", "rd_y"]].to_numpy().tolist()
                )
                p.terminate()
            else:
                seg_df["geometry"] = [
                    func_cor_square(permutation)
                    for permutation in seg_df[["rd_x", "rd_y"]].to_numpy().tolist()
                ]

            seg_df = seg_df[["geometry", "label"]]

            # Store the results in a geopandas dataframe.
            seg_df = gpd.GeoDataFrame(seg_df, geometry=seg_df.geometry)
            seg_df = seg_df.set_crs(epsg=28992)
            print("Geometry made in: " + str(timer() - start) + " second(s)")
            output_file_name = output_file_name_generator.generate_part_output_path(
                x_step
            )
            dissolve_gpd_output(seg_df, output_file_name)
            print(output_file_name)

            print("Writing finished in: " + str(timer() - start) + " second(s)")
            print(seg_df.columns)
            del seg_df
            begin_height = int(round(end_height + 1))
            end_height = int(round(begin_height + height_parts))

            if end_height > self.get_height() - (self.x_size / 2):
                end_height = round(self.get_height() - (self.x_size / 2))

        all_part = 0
        first_check = 0

        for file in glob.glob(
            output_file_name_generator.glob_wild_card_for_part_extension_only()
        ):
            if first_check == 0:
                all_part = gpd.read_file(file)
                first_check = 1
            else:
                print("Append")
                all_part = pd.concat([all_part, gpd.read_file(file)])

        try:
            if (
                str(type(amodel))
                != "<class 'nso_ds_classes.nso_ds_models.deep_learning_model'>"
                or str(type(amodel))
                == "<class 'nso_ds_classes.nso_ds_models.waterleiding_ahn_ndvi_model'>"
            ):
                all_part["label"] = all_part.apply(
                    lambda x: amodel.get_class_label(x["label"]), axis=1
                )
        except Exception as e:
            print(e)
        final_output_path = output_file_name_generator.generate_final_output_path()
        all_part.dissolve(by="label").to_file(final_output_path)

        for file in glob.glob(
            output_file_name_generator.glob_wild_card_for_all_part_files()
        ):
            os.remove(os.path.join(output_file_name_generator.output_path, file))

    def get_kernel_multi_processing(self, input_x_y):
        """
        This function is used to do multiprocessing predicting.

        This will get all the kernels first to be predicted later with a keras prediction function.
        Keras performs better when you give it multiple inputs instead of one.

        @param input_x_y: a array with the row and column for the to be predicted pixel.
        @return row and column and the kernel.
        """
        try:
            # Fetches the real coordinates for the row and column needed for writing to a geoformat.
            # actual_cor = self.get_x_cor_y_cor(x,y)
            kernel = self.get_kernel_for_x_y(input_x_y[0], input_x_y[1])
            # TODO: Set normalisation if used.
            # kernel = self.normalize_tile_kernel(kernel) if self.normalize == True else kernel

            return [input_x_y[0], input_x_y[1], kernel]

        except ValueError as e:
            if str(e) != "Center pixel is empty":
                print(e)
        except Exception as e:
            print(e)

    def predict_keras_multi_processing(self, input_x_y_kernel):
        """

        This function is used to do multiprocessing predicting

        Prediction function for keras models

        @param input_x_y: a array of kernels for keras predict to use.
        @return row and column and the predicted label in numbers.
        """
        try:
            # Fetches the real coordinates for the row and column needed for writing to a geoformat.
            kernels = [arow[2] for arow in input_x_y_kernel]

            # TODO: Fix bands and labels
            predicts = self.model.predict(kernels)
            print(predicts)

            row_id = 0
            returns = []
            for input_row in input_x_y_kernel:
                returns.append([input_row[0], input_row[1], predicts[row_id]])

            return returns

        except ValueError as e:
            print("Error in multiprocessing prediction:")
            print(e)
        except Exception as e:
            print("Error in multiprocessing prediction:")
            print(e)

    def predict_all_output_keras(
        self,
        amodel,
        output_location,
        aggregate_output=True,
        parts=10,
        begin_part=0,
        keras_break_size=10000,
        multiprocessing=False,
    ):
        """

        TODO: This function is outdated and needs to update with predict_all_output
        Predict all the pixels in the .tif file with kernels per pixel. mostly the same as predict_all_output only intended for keras models.

        Uses multiprocessing to speed up the results.

        @param amodel: A prediciton model with has to have a predict function.
        @param output_location: Locatie where to writes the results too.
        @param aggregate_output: 50 cm is the default resolution but we can aggregate to 2m
        @param parts: break the .tif file in multiple parts this is needed because some .tif files can contain 3 billion pixels which won't fit in one pass in memory.
        @param begin_part: skip certain parts in the parts
        """

        # Set some variables for breaking the .tif in different part parts in order to save memory.
        total_height = self.get_height() - self.x_size

        height_parts = round(total_height / parts)
        begin_height = self.x_size_begin
        end_height = self.x_size_begin + height_parts

        total_height = self.get_height() - self.x_size
        total_width = self.get_width() - self.y_size

        height_parts = total_height / parts

        # Set some variables for multiprocessing.
        self.set_model(amodel)
        dataset = rasterio.open(self.path_to_tif_file)

        # TODO: Set normalisation if used.
        # self.normalize = amodel.get_normalize()

        self.keras_break_size = keras_break_size

        # Loop through the parts.
        for x_step in tqdm(range(begin_part, parts)):
            print("-------")
            print("Part: " + str(x_step + 1) + " of " + str(parts))
            # Calculate the number of permutations for this step.
            permutations = list(
                itertools.product(
                    [x for x in range(begin_height, end_height)],
                    [
                        y
                        for y in range(
                            self.y_size_begin, self.get_width() - self.y_size_end
                        )
                    ],
                )
            )

            permutations = np.array(permutations)

            print("Total permutations this step: " + str(len(permutations)))

            # Init the multiprocessing pool.
            # TODO: Maybe use swifter for this?
            start = timer()
            print("Getting kernels")

            if multiprocessing == True:
                p = Pool()
                permutations = np.array(
                    p.map(self.get_kernel_multi_processing, permutations)
                )
                permutations = permutations[permutations != None]
                p.terminate()

                print("kernels at first step:")
                original_shape = permutations.shape[0]
                print(permutations.shape)

                permutations = np.array_split(permutations, self.keras_break_size)
                print("after split")
                print(len(permutations))
                # print("break size: "+ str(keras_break_size ))
                p = Pool()
                permutations = p.map(self.predict_keras_multi_processing, permutations)
                p.terminate()
            else:
                permutations = np.array(
                    [
                        self.get_kernel_multi_processing(permutation)
                        for permutation in permutations
                    ],
                    dtype="object",
                )
                print(permutations)
                print("kernels at first step:")
                original_shape = permutations.shape[0]
                print(permutations.shape)
                array_split_size = round(permutations.shape[0] / self.keras_break_size)
                permutations = np.array_split(permutations, array_split_size)

                print("After split")
                print(len(permutations))
                print("With size:")
                print(len(permutations[0]))
                print("Predicting")
                permutations = [
                    self.predict_keras_multi_processing(kernels)
                    for kernels in permutations
                ]
                print("After predict")
                print(len(permutations))
                print(permutations)

            try:
                permutations = np.concatenate(permutations)
                permutations = permutations.reshape(original_shape, 3)

                print("Pool finised in: " + str(timer() - start) + " second(s)")

                start = timer()
                seg_df = pd.DataFrame(permutations, columns=["x_cor", "y_cor", "label"])
                del permutations
                seg_df = seg_df[(seg_df["x_cor"] != 0) & (seg_df["y_cor"] != 0)]
                print(seg_df)
                print("Number of used pixels for this step: " + str(len(seg_df)))

                if len(seg_df) > 0:
                    # Get the coordinates for the pixel locations.
                    seg_df["rd_x"], seg_df["rd_y"] = rasterio.transform.xy(
                        dataset.transform, seg_df["x_cor"], seg_df["y_cor"]
                    )

                    print(
                        "Got coordinates for pixels: "
                        + str(timer() - start)
                        + " second(s)"
                    )

                    seg_df = seg_df.drop(["y_cor", "x_cor"], axis=1)

                    start = timer()
                    if aggregate_output == True:
                        seg_df["x_group"] = np.round(seg_df["rd_x"] / 2) * 2
                        seg_df["y_group"] = np.round(seg_df["rd_y"] / 2) * 2
                        seg_df = seg_df.groupby(["x_group", "y_group"]).agg(
                            label=("label", lambda x: x.value_counts().index[0])
                        )
                        print(
                            "Group by finised in: "
                            + str(timer() - start)
                            + " second(s)"
                        )

                        start = timer()
                        seg_df["rd_x"] = list(map(lambda x: x[0], seg_df.index))
                        seg_df["rd_y"] = list(map(lambda x: x[1], seg_df.index))
                        print(
                            "Labels created in: " + str(timer() - start) + " second(s)"
                        )

                        seg_df = seg_df[["rd_x", "rd_y", "label"]]

                    start = timer()

                    # Make squares from the the pixels in order to make contected polygons from them.
                    p = Pool()
                    seg_df["geometry"] = p.map(
                        func_cor_square, seg_df[["rd_x", "rd_y"]].to_numpy().tolist()
                    )

                    p.terminate()
                    seg_df = seg_df[["geometry", "label"]]

                    # Store the results in a geopandas dataframe.
                    seg_df = gpd.GeoDataFrame(seg_df, geometry=seg_df.geometry)
                    seg_df = seg_df.set_crs(epsg=28992)
                    print("Geometry made in: " + str(timer() - start) + " second(s)")
                    try:
                        dissolve_gpd_output(
                            seg_df,
                            output_location.replace(".", "_part_" + str(x_step) + "."),
                        )
                        print(
                            output_location.replace(".", "_part_" + str(x_step) + ".")
                        )
                    except:
                        print("Warning nothing has been written")

                    print("Writing finised in: " + str(timer() - start) + " second(s)")
                    print(seg_df.columns)
                    del seg_df
                    begin_height = int(round(end_height + 1))
                    end_height = int(round(begin_height + height_parts))

                    if end_height > self.get_height() - (self.x_size / 2):
                        end_height = round(self.get_height() - (self.x_size / 2))
                else:
                    print("WARNING! Empty DataFrame!")
            except Exception as e:
                print(e)

        all_part = 0
        first_check = 0

        for file in glob.glob(output_location.replace(".", "_part_*.")):
            print(file)
            if first_check == 0:
                all_part = gpd.read_file(file)
                first_check = 1
            else:
                print("Append")
                all_part = all_part.append(gpd.read_file(file))

        all_part.dissolve(by="label").to_file(output_location)

        for file in glob.glob(output_location.replace(".", "_part_*.").split(".")[0]):
            os.remove(file)

    def sample_pixels(self, amount=100):
        """
        Sample pixels from the tif file.

        @param amount: the size of the sample.
        @return sample array.
        """

        if (self.get_height() * self.get_width()) <= amount:
            raise "Sample amount higher than total number of pixels, so can't sample"

        height_amount = amount if self.get_height() < amount else self.get_height()
        width_amount = amount if self.get_width() < amount else self.get_width()

        height_sample = random.sample(range(0, self.get_height()), height_amount)
        width_sample = random.sample(range(0, self.get_width()), width_amount)

        return_samples = []

        permutations = list(
            itertools.product(
                [height_sample[x] for x in range(1, len(height_sample))],
                [width_sample[y] for y in range(1, len(width_sample))],
            )
        )
        x_samp = 0

        while len(return_samples) < amount:
            try:
                return_samples.append(
                    self.get_pixel_value(
                        permutations[x_samp][0], permutations[x_samp][1]
                    )
                )

            except Exception as e:
                if str(e) != "Center pixel is empty":
                    print(e)

            x_samp = x_samp + 1

        return return_samples

    def sample_kernels(self, amount=100):
        """
        Sample kernels from the tif file.

        @param amount: the size of the sample.
        @return sample array of kernels.
        """

        if (self.get_height() * self.get_width()) <= amount:
            raise "Sample amount higher than total number of pixels, so can't sample"

        height_amount = amount if self.get_height() < amount else self.get_height()
        width_amount = amount if self.get_width() < amount else self.get_width()

        height_sample = random.sample(range(0, self.get_height()), height_amount)
        width_sample = random.sample(range(0, self.get_width()), width_amount)

        return_samples = []

        permutations = list(
            itertools.product(
                [height_sample[x] for x in range(1, len(height_sample))],
                [width_sample[y] for y in range(1, len(width_sample))],
            )
        )
        x_samp = 0

        while len(return_samples) < amount:
            try:
                return_samples.append(
                    self.get_kernel_for_x_y(
                        permutations[x_samp][0], permutations[x_samp][1]
                    )
                )

            except Exception as e:
                if str(e) != "Center pixel is empty":
                    print(e)

            x_samp = x_samp + 1

        return return_samples

    def set_model(self, amodel):
        """
        Set a model coupled to this .tif generator.
        Mostly used for multiprocessing purposes

        @param amodel: The specific model to set.

        """
        self.model = amodel

    def get_height(self):
        """
        Get the height of the .tif file.

        @return the height of the .tif file.
        """
        return self.height

    def get_width(self):
        """
        Get the width of the .tif file.

        @return the width of the .tif file.
        """
        return self.width

    def get_data(self):
        """

        Return the numpy array with all the spectral data in it.

        @return the numpy data with the spectral data  in it.
        """
        return self.data

    def get_sat_name(self):
        """

        Return the satellite name based on the file extension.

        @return string with the satellite name.
        """

        return self.sat_name


def normalizedata(data):
    """
    Normalize between 0 en 1.


    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def plot_kernel(kernel, y=0):
    """
    Plot a kernel or .tif image.

    Multiple inputs are correct either a numpy array or x,y coordinates.

    @param kernel: A kernel that you want to plot or x coordinate.
    @param y: the y coordinate you want to plot.
    """

    if isinstance(kernel, int):
        rasterio.plot.show(
            np.clip(self.get_kernel_for_x_y(kernel, y)[2::-1], 0, 2200) / 2200
        )
    else:
        rasterio.plot.show(np.clip(kernel[2::-1], 0, 2200) / 2200)


def func_cor_square(input_x_y):
    """
    This function is used to make squares out of pixels for a inter connected output.

    @param input_x_y a pixel input variable to be made into a square.
    @return the the squared pixel.
    """
    rect = [round(input_x_y[0] / 2) * 2, round(input_x_y[1] / 2) * 2, 0, 0]
    rect[2], rect[3] = rect[0] + 2, rect[1] + 2
    coords = Polygon(
        [
            (rect[0], rect[1]),
            (rect[2], rect[1]),
            (rect[2], rect[3]),
            (rect[0], rect[3]),
            (rect[0], rect[1]),
        ]
    )
    return coords
