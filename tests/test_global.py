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
import pandas as pd

# import settings_test

# All these variables are local!
# TODO: pytest does not seem to find the settings_test file after several attempts, for now hardcoded settings in the file.
model_path = "C:/repos/satellite-images-nso-datascience/saved_models/Superview_Nieuwkoopse_plassen_20190302_113613_to_20221012_104900_random_forest_classifier.sav"
test_tif_files_dir = "E:/output/test/Nieuwkoopse_plassen/*SV*.tif"
output_path_test = "E:/output/test/Nieuwkoopse_plassen/"
output_path = "E:/output/test/Nieuwkoopse_plassen/"

final_artefact = pickle.load(open(model_path, "rb"))


selected_features = ["r", "g", "b", "i", "ndvi", "ndwi"]


def test_predict_all_function():

    # Predict small .tif files.
    for a_tif_file in glob.glob(test_tif_files_dir):
        a_tif_file = a_tif_file.replace("\\", "/")

        with contextlib.redirect_stdout(io.StringIO()):
            output_file_name_generator = OutputFileNameGenerator(
                output_path=output_path_test,
                output_file_name=output_path
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

            falses = 0
            for afile in glob.glob(output_path_test + "*SV*.parquet"):
                afile = afile.replace("\\", "/")

                if (
                    pd.read_parquet(afile)["label"].value_counts().index[0]
                    != afile.split("_test")[0].split("_")[-1]
                ):

                    falses = falses + 1

            print(
                "False rating off: "
                + str(falses / len(glob.glob(output_path_test + "*SV*.parquet")))
            )

            os.remove(
                output_path + a_tif_file.split("/")[-1].replace(".tif", ".parquet")
            )
            assert falses == 0
