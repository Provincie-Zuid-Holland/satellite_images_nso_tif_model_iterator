import contextlib
import glob
import io
import os
import pickle

import geopandas as gpd
import geopandas.testing
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError

import tests.test_settings as test_settings
from satellite_images_nso_tif_model_iterator.filenames.file_name_generator import (
    OutputFileNameGenerator,
)
from satellite_images_nso_tif_model_iterator.h3_hexagons.nso_tif_model_iterater_hexagon_output import (
    output_h3_hexagons_from_pixels,
)
from satellite_images_nso_tif_model_iterator.tif_model_iterator import (
    tif_model_iterator,
)


# Add functions
def is_valid_tif_raster(file_path):
    """
    Check if a file is a valid TIFF raster file using rasterio.

    Args:
        file_path (str): Path to the file to check

    Returns:
        bool: True if valid TIFF raster, False otherwise
        str: Message with details about validation
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"

    # Check file extension (optional check)
    if not file_path.lower().endswith((".tif", ".tiff")):
        return False, f"File doesn't have a .tif or .tiff extension: {file_path}"

    try:
        # Try to open the file with rasterio
        with rasterio.open(file_path) as dataset:
            # Get basic raster information
            driver = dataset.driver
            count = dataset.count
            dtype = dataset.dtypes[0]
            shape = dataset.shape

            # Check if it's a TIFF format (rasterio uses uppercase "GTiff")
            if driver != "GTiff":
                return (
                    False,
                    f"File is a raster but not in TIFF format. Format: {driver}",
                )

            # Check if it has at least one raster band
            if count < 1:
                return False, f"File is a TIFF but has no raster bands"

            # Return detailed information about the raster
            return (
                True,
                f"Valid TIFF raster: {count} band(s), {dtype} data type, {shape[0]}x{shape[1]} pixels",
            )

    except RasterioIOError as e:
        return False, f"Not a valid raster file: {str(e)}"
    except ValueError as e:
        return False, f"Invalid raster format: {str(e)}"
    except Exception as e:
        return False, f"Error validating file: {str(e)}"


# Begin Tests
def test_predict_all_function_superview_files():
    """
    Test if the iterator works on small .tif files from the superview constellation.

    """

    final_artefact = pickle.load(
        open(os.path.abspath(test_settings.model_path_sv).replace("\\", "/"), "rb")
    )
    selected_features = ["r", "g", "b", "i", "ndvi", "ndwi"]

    falses = -1

    # Predict small .tif files.
    for a_tif_file in glob.glob(test_settings.tif_file_input + "*SV*.tif"):
        a_tif_file = a_tif_file.replace("\\", "/")

        with contextlib.redirect_stdout(io.StringIO()):
            output_file_name_generator = OutputFileNameGenerator(
                output_path=os.path.abspath(test_settings.output_path_test) + "/",
                output_file_name=os.path.abspath(test_settings.output_path_test)
                + "/"
                + a_tif_file.split("/")[-1].replace(".tif", ".parquet"),
            )

        nso_tif_model_iterator_generator = tif_model_iterator.TifModelIteratorGenerator(
            path_to_tif_file=a_tif_file,
            model=final_artefact["model"],
            output_file_name_generator=output_file_name_generator,
            parts=1,
            normalize_scaler=final_artefact["scaler"],
            column_names=selected_features,
            dissolve_parts=False,
            square_output=False,
            do_all_parts=True,
        )

        nso_tif_model_iterator_generator.predict_all_output()

        if falses == -1:
            falses = 0

        # The iterator should produce small .parquet files, let's if they are filled with data.
        for afile in glob.glob(
            os.path.abspath(test_settings.output_path_test) + "*SV*.parquet"
        ):
            afile = afile.replace("\\", "/")

            df_check = pd.read_parquet(afile)

            if (
                len(df_check) <= 0
                or (["rd_x", "rd_y", "label"] != df_check.columns).any()
            ):
                falses = falses + 1

            os.remove(afile)
    assert falses == 0


def test_predict_all_function_pneo_files():
    """

    Test if the iterator works on small .tif files from the pneo constellation

    """

    final_artefact = pickle.load(
        open(os.path.abspath(test_settings.model_path_pneo), "rb")
    )
    selected_features = ["r", "g", "b", "n", "e", "d", "ndvi", "re_ndvi"]

    falses = -1
    # Predict small .tif files.
    for a_tif_file in glob.glob(test_settings.tif_file_input + "*pneo*.tif"):
        a_tif_file = a_tif_file.replace("\\", "/")

        with contextlib.redirect_stdout(io.StringIO()):
            output_file_name_generator = OutputFileNameGenerator(
                output_path=os.path.abspath(test_settings.output_path_test) + "/",
                output_file_name=os.path.abspath(test_settings.output_path_test)
                + "/"
                + a_tif_file.split("/")[-1].replace(".tif", ".parquet"),
            )

            nso_tif_kernel_iterator_generator = (
                tif_model_iterator.TifModelIteratorGenerator(
                    path_to_tif_file=a_tif_file,
                    model=final_artefact["model"],
                    output_file_name_generator=output_file_name_generator,
                    parts=1,
                    normalize_scaler=final_artefact["scaler"],
                    column_names=selected_features,
                    dissolve_parts=False,
                    square_output=False,
                    do_all_parts=True,
                )
            )

            nso_tif_kernel_iterator_generator.predict_all_output()

            if falses == -1:
                falses = 0

            for afile in glob.glob(test_settings.output_path_test + "*PNEO*.parquet"):
                afile = afile.replace("\\", "/")
                df_check = pd.read_parquet(afile)

                if (
                    len(df_check) <= 0
                    or (["rd_x", "rd_y", "label"] != df_check.columns).any()
                ):
                    falses = falses + 1

                os.remove(afile)

    assert falses == 0


def test_raster_output():
    final_artefact = pickle.load(
        open(os.path.abspath(test_settings.model_path_pneo), "rb")
    )
    selected_features = ["r", "g", "b", "n", "e", "d", "ndvi", "re_ndvi"]

    falses = -1
    # Predict small .tif files.
    for a_tif_file in glob.glob(test_settings.tif_file_input + "*pneo*.tif"):
        a_tif_file = a_tif_file.replace("\\", "/")

        with contextlib.redirect_stdout(io.StringIO()):
            output_file_name_generator = OutputFileNameGenerator(
                output_path=os.path.abspath(test_settings.output_path_test) + "/",
                output_file_name=os.path.abspath(test_settings.output_path_test)
                + "/"
                + a_tif_file.split("/")[-1].replace(".tif", ".tif"),
            )

            nso_tif_kernel_iterator_generator = (
                tif_model_iterator.TifModelIteratorGenerator(
                    path_to_tif_file=a_tif_file,
                    model=final_artefact["model"],
                    output_file_name_generator=output_file_name_generator,
                    parts=1,
                    normalize_scaler=final_artefact["scaler"],
                    column_names=selected_features,
                    dissolve_parts=False,
                    square_output=True,
                    do_all_parts=True,
                )
            )

            nso_tif_kernel_iterator_generator.predict_all_output()

            if falses == -1:
                falses = 0

            for afile in glob.glob(test_settings.output_path_test + "*PNEO*.tif"):
                afile = afile.replace("\\", "/")

                if not is_valid_tif_raster(afile):
                    falses = falses + 1

                os.remove(afile)

    assert falses == 0


# TODO: Fix hexagon unit tests.
# def test_hexagon():
#    """
#
#    Test if the hexagon output is working.
#
#    """
#
#    path_hexagons = output_h3_hexagons_from_pixels(
#        os.path.abspath(test_settings.hexagon_test_file),
#        output_file_path=test_settings.a_hexagon_test_output_file,
#        resolution=13,
#    )
#
#    check_gpd_assert = geopandas.testing.assert_geodataframe_equal(
#        gpd.read_file(path_hexagons),
#        gpd.read_file(test_settings.a_hexagon_compare_test_file),
#    )
#    os.remove(path_hexagons)
#
#   assert check_gpd_assert == None
#    os.remove(path_hexagons)
#
#    assert check_gpd_assert == None
