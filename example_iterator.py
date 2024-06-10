import pickle

import settings
from src.filenames.file_name_generator import OutputFileNameGenerator
from src.tif_model_iterator import tif_kernel_iterator
import glob

if __name__ == "__main__":
    filename = settings.MODEL_PATH
    tif_file_regex = settings.TIF_FILE_INPUT_REGEX
    output_path = settings.OUTPUT_PATH

    number_of_parts = 20

    loaded_model = pickle.load(open(filename, "rb"))
    print("Loaded model: " + filename.split("/")[-1])

    tif_files = [file for file in glob.glob(tif_file_regex)]

    for tif_file in tif_files:
        tif_file = tif_file.replace("\\", "/")
        print("----------")
        print("Implementing model on: " + tif_file)

        try:
            output_file_name_generator = OutputFileNameGenerator(
                output_path=output_path,
                output_file_name=output_path
                + tif_file.split("/")[-1].replace(".tif", settings.OUTPUT_EXTENTSION),
            )

            nso_tif_kernel_iterator_generator = (
                tif_kernel_iterator.TifKernelIteratorGenerator(
                    path_to_tif_file=tif_file,
                    model=loaded_model["model"],
                    output_file_name_generator=output_file_name_generator,
                    parts=number_of_parts,
                    normalize_scaler=loaded_model["scaler"],
                    column_names=settings.COLUMN_NAMES,
                    dissolve_parts=False,
                    square_output=False,
                )
            )

            nso_tif_kernel_iterator_generator.predict_all_output()
            print(
                "File stored here: "
                + str(nso_tif_kernel_iterator_generator.return_final_output_path())
            )

        except Exception as e:
            print(e)
