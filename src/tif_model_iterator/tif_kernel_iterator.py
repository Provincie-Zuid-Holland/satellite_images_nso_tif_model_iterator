import glob
import math
import os
import warnings
from copy import deepcopy
from timeit import default_timer as timer

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Polygon
from sklearn.base import ClassifierMixin
from tqdm import tqdm

from src.filenames.file_name_generator import OutputFileNameGenerator
from src.tif_model_iterator.__nso_ds_output import dissolve_gpd_output

warnings.filterwarnings("ignore", category=UserWarning)

"""
    This code is used to inference a model on every pixel in a .tif file.

    TODO: Kernels are not supported anymore currently only models with pixels are used.

    Author: Michael de Winter, Pieter-Kouyzer
"""


class TifKernelIteratorGenerator:
    """

    This class set up a .tif image in order to easily inference a model on every pixel.
    We call it here a iterator.

    """

    def __init__(
        self,
        path_to_tif_file: str,
        model: ClassifierMixin,
        output_file_name_generator: OutputFileNameGenerator,
        parts: int,
        normalize_scaler: str = False,
        column_names: list = [
            "r",
            "g",
            "b",
            "n",
            "e",
            "d",
            "ndvi",
            "re_ndvi",
            "height",
        ],
    ):
        """

        Init of the tif Kernel

        @param path_to_file: A path to a .tif file.
        @param model: A prediction model with has to have a predict function and uses pixels as input.
        @param output_file_name_generator: Generates desired filenames for output files
        @param parts: Into how many parts to break the .tif file, this has to be done since most extracted pixels or kernel don't fit in memory thus we divide the pixels into smaller chunks.
        @param normalize_scaler: Whether to use a normalize/scaler on all the kernels or not, the input here so be a normalize/scaler function. You have to submit the normalizer/scaler as a argument here if you want to use a scaler, this has to be a custom  class like nso_ds_normalize_scaler.
        @param column_names: names of the bands in the tif file.
        """
        self.model = model
        self.output_file_name_generator = output_file_name_generator
        self.parts = parts
        self.normalize_scaler = normalize_scaler
        self.column_names = column_names
        self.dataset = rasterio.open(path_to_tif_file)
        meta = self.dataset.meta.copy()
        self.descriptions = deepcopy(self.dataset.descriptions)
        self.data = self.dataset.read()
        self.width, self.height = meta["width"], meta["height"]
        self.bands = [band + 1 for band in range(0, self.data.shape[0])]

    def predict_all_output(
        self,
        begin_part=0,
    ):
        """
        Predicts labels for all self.data and writes it to file, by cutting it into parts and
        using these smaller parts to avoid running out of memory for big files.

        @param begin_part: Allows you to begin at a later part, if your computer froze halfway
        """
        x_step_size = math.ceil(self.height / self.parts)
        bottom = 0
        top = self.width

        # Divide the satellite images into multiple parts and loop through the parts, using parts reduces the amount of RAM required to run this process.
        for x_step in tqdm(range(begin_part, self.parts)):
            self._create_and_write_part_output(
                x_step=x_step, x_step_size=x_step_size, bottom=bottom, top=top
            )

        self._write_full_gdf_to_file()
        self._clean_up_part_files()

    def _create_and_write_part_output(
        self,
        x_step: int,
        x_step_size: int,
        bottom: int,
        top: int,
    ):
        """
        Creates the label predictions for a part of self.data and writes it to file

        @param: x_step: Which step is currently being processed
        @param: x_step_size: The step size for each step in the x direction
        @param bottom: Lowest y coordinate
        @param top: Highest y coordinate
        """
        print("-------")
        print("Part: " + str(x_step + 1) + " of " + str(self.parts))
        left_boundary = x_step * x_step_size
        right_boundary = (x_step + 1) * x_step_size

        # lower and the top pixels will always be the same.
        subset_data = self.data[:, left_boundary:right_boundary, bottom:top]

        subset_df = self._create_pixel_coordinate_dataframe(
            data=subset_data, left_boundary=left_boundary
        )

        subset_df = self._filter_out_empty_pixels(subset_df)

        if len(subset_df) == 0:
            print("This part is empty, so we skip the next steps.")
            return

        # Check if a normalizer or a  scaler has to be used.
        if self.normalize_scaler is not False:
            print("Normalizing/Scaling data")
            start = timer()
            subset_df[self.column_names] = self.normalize_scaler.transform(
                subset_df[self.column_names]
            )
            print(f"Normalizing/scaling finished in: {str(timer() - start)} second(s)")

        subset_df = self._predict_labels(df=subset_df)

        subset_df = self.transform_to_polygons(subset_df)
        self._write_part_to_file(
            gdf=subset_df,
            step=x_step,
        )

    def _create_pixel_coordinate_dataframe(
        self, data: np.array, left_boundary: int
    ) -> pd.DataFrame:
        """
        Transforms data, such that it becomes a pandas dataframe with the self.bands columns + rd_x and rd_y coordinates (Rijksdriehoeks coordinates)

        @param data: numpy array such with 3 shapes, first are the bands, then x, then y coordinates
        @param left_boundary: the x_coordinate coresponding to the left boundary of data

        @return Pandas DataFrame with columns: bands + [rd_x, rd_y]
        """
        print("Creating Pixel Coordinates")
        start = timer()
        z_shape = data.shape[0]
        x_shape = data.shape[1]
        y_shape = data.shape[2]
        x_coordinates = [
            [left_boundary + x for y in range(0, data.shape[2])]
            for x in range(0, data.shape[1])
        ]
        y_coordinates = [
            [y for y in range(0, data.shape[2])] for x in range(0, data.shape[1])
        ]

        # We have to have a square so we are getting the upper left, upper right, lower left and lower right coordinates of the corners of a pixel
        rd_x_ul, rd_y_ul = rasterio.transform.xy(
            self.dataset.transform, x_coordinates, y_coordinates, offset="ul"
        )

        rd_x_ur, rd_y_ur = rasterio.transform.xy(
            self.dataset.transform, x_coordinates, y_coordinates, offset="ur"
        )

        rd_x_ll, rd_y_ll = rasterio.transform.xy(
            self.dataset.transform, x_coordinates, y_coordinates, offset="ll"
        )

        rd_x_lr, rd_y_lr = rasterio.transform.xy(
            self.dataset.transform, x_coordinates, y_coordinates, offset="lr"
        )

        data = np.append(data, rd_x_ul).reshape([z_shape + 1, x_shape, y_shape])
        data = np.append(data, rd_y_ul).reshape([z_shape + 2, x_shape, y_shape])
        data = np.append(data, rd_x_ur).reshape([z_shape + 3, x_shape, y_shape])
        data = np.append(data, rd_y_ur).reshape([z_shape + 4, x_shape, y_shape])
        data = np.append(data, rd_x_ll).reshape([z_shape + 5, x_shape, y_shape])
        data = np.append(data, rd_y_ll).reshape([z_shape + 6, x_shape, y_shape])
        data = np.append(data, rd_x_lr).reshape([z_shape + 7, x_shape, y_shape])
        data = np.append(data, rd_y_lr).reshape([z_shape + 8, x_shape, y_shape])

        data = data.reshape(-1, x_shape * y_shape).transpose()

        df = pd.DataFrame(
            data,
            columns=self.column_names
            + [
                "rd_x_ul",
                "rd_y_ul",
                "rd_x_ur",
                "rd_y_ur",
                "rd_x_ll",
                "rd_y_ll",
                "rd_x_lr",
                "rd_y_lr",
            ],

        )

        print(
            f"Pixel coordinates creation finished in: {str(timer() - start)} second(s)"
        )
        return df

    @staticmethod
    def _filter_out_empty_pixels(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove pixels which have all RGB values zero, as these correspond to the tif_file not being filled there.
        Note: RGB corresponds to bands 1, 2, 3.

        @param df: DataFrame with at least bands 1,2,3 for columns

        @return df: Filtered version of df
        """
        # We want to have RGB values != 0 for any point we want to predict on RGB is in bands 1,2,3
        non_empty_pixel_mask = (df[["r", "g", "b"]] != 0).any(axis="columns")
        return df[non_empty_pixel_mask]

    def _predict_labels(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predicts labels for df

        @param df: DataFrame with bands for column names. Should correspond with self.model features

        @return: As df, but with predict 'label' column added
        """
        print("Predicting labels")
        start = timer()
        feature_names = getattr(self.model, "feature_names_in_", None)
        if feature_names is not None:
            X = df[self.model.feature_names_in_]
        else:
            X = df.iloc[:, : len(self.bands)].values
        df["label"] = self.model.predict(X)
        print(f"Predicting finished in: {str(timer() - start)} second(s)")
        return df

    @staticmethod
    def transform_to_polygons(df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Changes the rd_x, rd_y coordinates of df into square polygons, so df can be a GeoDataFrame with Polygons as geometry

        @param df: DataFrame with rd_x, rd_y and label columns

        @return gdf: GeoDataFrame like df, but with rd_x and rd_y transposed into a square polygon geometry
        """
        print("Creating geometry")
        start = timer()
        # Make squares from the the pixels in order to make connected polygons from them.
        df["geometry"] = df.apply(
            lambda x: Polygon(
                [
                    (x["rd_x_ul"], x["rd_y_ul"]),
                    (x["rd_x_ur"], x["rd_y_ur"]),
                    (x["rd_x_lr"], x["rd_y_lr"]),
                    (x["rd_x_ll"], x["rd_y_ll"]),
                    (x["rd_x_ul"], x["rd_y_ul"]),
                ]
            ),
            axis=1,
        )

        df = df[["geometry", "label"]]

        gdf = gpd.GeoDataFrame(df, geometry=df.geometry)
        gdf = gdf.set_crs(epsg=28992)
        print("Geometry made in: " + str(timer() - start) + " second(s)")
        return gdf

    def _write_part_to_file(
        self,
        gdf: gpd.GeoDataFrame,
        step: int,
    ):
        """
        Writes a part of the tif file to an outputfile.

        @param gdf: GeoDataFrame to write to part file
        @param step: the step currently being written to file
        """
        print("Dissolving and writing to file")

        output_file_name = self.output_file_name_generator.generate_part_output_path(
            step
        )

        dissolve_gpd_output(gdf, output_file_name)

    def _write_full_gdf_to_file(self):
        """
        Reads all files of the parts, combines them into 1 and writes the full gdf to file.
        """
        all_part_files = glob.glob(
            self.output_file_name_generator.glob_wild_card_for_part_extension_only()
        )
        full_gdf = pd.concat([gpd.read_file(file) for file in all_part_files])

        try:
            if (
                str(type(self.model))
                != "<class 'nso_ds_classes.nso_ds_models.deep_learning_model'>"
                or str(type(self.model))
                == "<class 'nso_ds_classes.nso_ds_models.waterleiding_ahn_ndvi_model'>"
            ):
                full_gdf["label"] = full_gdf.apply(
                    lambda x: self.model.get_class_label(x["label"]), axis=1
                )
        except Exception as e:
            print(e)
        final_output_path = self.output_file_name_generator.generate_final_output_path()

        if ".geojson" in final_output_path:
            print("Writing to geojson")
            full_gdf.dissolve(by="label").to_file(final_output_path, driver="GeoJSON")
        else:
            full_gdf.dissolve(by="label").to_file(final_output_path)

    def _clean_up_part_files(self):
        """
        Deletes all part files to clean up.
        """
        for file in glob.glob(
            self.output_file_name_generator.glob_wild_card_for_all_part_files()
        ):
            os.remove(os.path.join(self.output_file_name_generator.output_path, file))
