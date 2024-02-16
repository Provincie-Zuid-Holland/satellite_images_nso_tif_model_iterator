import pickle

import settings
from src.filenames.file_name_generator import OutputFileNameGenerator
from src.tif_model_iterator import tif_kernel_iterator
import glob

if __name__ == "__main__":
    filename = settings.MODEL_PATH
    tif_file_regex = settings.TIF_FILE_INPUT_REGEX
    output_path = settings.OUTPUT_PATH

    print(filename)
    loaded_model = pickle.load(open(filename, "rb"))

    tif_files = [file for file in glob.glob(tif_file_regex)]

    for tif_file in tif_files:
        print("----------")
        print(tif_file)

        tif_file = tif_file.replace("\\", "/")
        output_path = output_path
        date = output_path.split("/")[-1].split("_")[0]

        output_file_name = output_path + tif_file.split("/")[-1].replace(
            ".tif", ".geojson"
        ).replace("\\", "/")

        output_file_name_generator = OutputFileNameGenerator(
            output_path=output_path, output_file_name=output_file_name
        )

        nso_tif_kernel_iterator_generator = (
            tif_kernel_iterator.TifKernelIteratorGenerator(
                path_to_tif_file=tif_file,
                model=loaded_model["model"],
                output_file_name_generator=output_file_name_generator,
                parts=4,
                normalize_scaler=loaded_model["scaler"],
                aggregate=False,
                resolution_aggregate=0.2,
                column_names=["r", "g", "b", "n", "e", "d", "ndvi", "re_ndvi"],
            )
        )

        nso_tif_kernel_iterator_generator.predict_all_output()
