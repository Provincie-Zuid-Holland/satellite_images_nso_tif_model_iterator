import glob
import pickle
import os
import contextlib
import io
from satellite_images_nso_tif_model_iterator.tif_model_iterator import (
    tif_model_iterator,
)
from satellite_images_nso_tif_model_iterator.filenames.file_name_generator import (
    OutputFileNameGenerator,
)
from satellite_images_nso_tif_model_iterator.h3_hexagons.nso_tif_model_iterater_hexagon_output import (
    output_h3_hexagons_from_pixels,
)
import pandas as pd
import geopandas as gpd
import geopandas.testing
import tests.test_settings as test_settings


# Functions
def is_valid_geodataframe_with_polygons(df):
    """
    Check if the given DataFrame is a GeoDataFrame containing only Polygon or MultiPolygon geometries.

    Parameters:
    df (pd.DataFrame): The DataFrame to check.

    Returns:
    bool: True if the DataFrame is a GeoDataFrame with only Polygon or MultiPolygon geometries, False otherwise.
    """
    if isinstance(df, gpd.GeoDataFrame):
        if "geometry" in df.columns:
            # Check if the 'geometry' column contains only Polygon or MultiPolygon
            geom_types = df["geometry"].geom_type.unique()
            return all(
                geom_type in {"Polygon", "MultiPolygon"} for geom_type in geom_types
            )
    return False


# Begin Tests
def test_predict_all_function_superview_files():

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


def test_hexagon():

    path_hexagons = output_h3_hexagons_from_pixels(
        os.path.abspath(test_settings.hexagon_test_file), 
        output_file_path=test_settings.a_hexagon_test_output_file,
        resolution=13,
    )

    check_gpd_assert = geopandas.testing.assert_geodataframe_equal(gpd.read_file(path_hexagons), gpd.read_file(test_settings.a_hexagon_compare_test_file))
    #os.remove(path_hexagons)

    
    assert  check_gpd_assert== None
