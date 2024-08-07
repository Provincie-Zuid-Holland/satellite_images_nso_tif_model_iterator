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
from pathlib import Path
import pandas as pd
import geopandas as gpd


# import settings_test


# All these variables are local!
# TODO: pytest does not seem to find the settings_test file after several attempts, for now hardcoded settings in the file.
model_path_sv = "C:/repos/satellite-images-nso-datascience/saved_models/Superview_Nieuwkoopse_plassen_20190302_113613_to_20221012_104900_random_forest_classifier.sav"
test_tif_files_dir_sv = "E:/output/test/Nieuwkoopse_plassen/input_data/*SV*.tif"


model_path_pneo = "C:/repos/satellite-images-nso-datascience/saved_models/PNEO_Nieuwkoopse_plassen_20230603_to_20230905_random_forest_classifier.sav"
test_tif_files_dir_pneo = "E:/output/test/Nieuwkoopse_plassen/input_data/*PNEO*.tif"


a_tif_file_hexagon_test = "E:/output/test/Nieuwkoopse_plassen/input_data/20230905_105231_PNEO-03_1_1_30cm_RD_12bit_RGBNED_Uithoorn_Nieuwkoopse_Plassen_De_Haeck_cropped_ndwi_re_ndvi_Ground_test.tif"

output_path_test = "E:/output/test/Nieuwkoopse_plassen/output/"

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


# Tests


def test_predict_all_function_superview_files():

    final_artefact = pickle.load(open(model_path_sv, "rb"))
    selected_features = ["r", "g", "b", "i", "ndvi", "ndwi"]
    # Predict small .tif files.

    falses = -1

    for a_tif_file in glob.glob(test_tif_files_dir_sv):
        a_tif_file = a_tif_file.replace("\\", "/")
        print(a_tif_file)

        with contextlib.redirect_stdout(io.StringIO()):
            output_file_name_generator = OutputFileNameGenerator(
                output_path=output_path_test,
                output_file_name=output_path_test
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
            skip_done_part=False,
        )

        nso_tif_model_iterator_generator.predict_all_output()

        if falses == -1:
            falses = 0

        for afile in glob.glob(output_path_test + "*SV*.parquet"):
            afile = afile.replace("\\", "/")
            print(afile)

            print(pd.read_parquet(afile)["label"].value_counts())

            print(afile.split("_test")[0].split("_")[-1])
            if (
                pd.read_parquet(afile)["label"].value_counts().index[0]
                != afile.split("_test")[0].split("_")[-1]
            ):
                print("Wrong!!!!!!!")
                falses = falses + 1
            os.remove(afile)

        print("False rating off: " + str(falses / len(glob.glob(output_path_test))))

    assert falses == 0


def test_predict_all_function_pneo_files():

    final_artefact = pickle.load(open(model_path_pneo, "rb"))
    selected_features = ["r", "g", "b", "n", "e", "d", "ndvi", "re_ndvi"]

    falses = -1
    # Predict small .tif files.
    for a_tif_file in glob.glob(test_tif_files_dir_pneo):
        a_tif_file = a_tif_file.replace("\\", "/")

        with contextlib.redirect_stdout(io.StringIO()):
            output_file_name_generator = OutputFileNameGenerator(
                output_path=output_path_test,
                output_file_name=output_path_test
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
                    skip_done_part=False,
                )
            )

            nso_tif_kernel_iterator_generator.predict_all_output()

            if falses == -1:
                falses = 0

            for afile in glob.glob(output_path_test + "*PNEO*.parquet"):
                afile = afile.replace("\\", "/")

                if (
                    pd.read_parquet(afile)["label"].value_counts().index[0]
                    != afile.split("_test")[0].split("_")[-1]
                ):

                    falses = falses + 1

                print(
                    "False rating off: "
                    + str(falses / len(glob.glob(output_path_test + "*PNEO*.parquet")))
                )

                os.remove(afile)

    assert falses == 0


def test_hexagon():

    with contextlib.redirect_stdout(io.StringIO()):
        output_file_name_generator = OutputFileNameGenerator(
            output_path=output_path_test,
            output_file_name=output_path_test
            + a_tif_file_hexagon_test.split("/")[-1].replace(".tif", ".parquet"),
        )

    final_artefact = pickle.load(open(model_path_pneo, "rb"))
    selected_features = ["r", "g", "b", "n", "e", "d", "ndvi", "re_ndvi"]

    nso_tif_kernel_iterator_generator = tif_model_iterator.TifModelIteratorGenerator(
        path_to_tif_file=a_tif_file_hexagon_test,
        model=final_artefact["model"],
        output_file_name_generator=output_file_name_generator,
        parts=1,
        normalize_scaler=final_artefact["scaler"],
        column_names=selected_features,
        dissolve_parts=False,
        square_output=False,
        skip_done_part=False,
    )

    nso_tif_kernel_iterator_generator.predict_all_output()
    print(
        "File stored here: "
        + str(nso_tif_kernel_iterator_generator.get_final_output_path())
    )

    path_hexagons = output_h3_hexagons_from_pixels(
        nso_tif_kernel_iterator_generator.get_final_output_path(),
        13,
    )

    file_exists = is_valid_geodataframe_with_polygons(gpd.read_file(path_hexagons))
    os.remove(path_hexagons)
    os.remove(nso_tif_kernel_iterator_generator.get_final_output_path())
    assert file_exists
