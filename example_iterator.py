import pickle
import settings
from filenames.file_name_generator import OutputFileNameGenerator
from tif_model_iterator import tif_kernel_iterator
from h3_hexagons.nso_tif_model_iterater_hexagon_output import (
    output_h3_hexagons_from_pixels,
)
import glob


if __name__ == "__main__":
    filename = settings.MODEL_PATH
    tif_file_regex = settings.TIF_FILE_INPUT_REGEX
    output_path = settings.OUTPUT_PATH

    number_of_parts = 4

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
                    dissolve_parts=settings.DISSOLVE_PARTS,
                    square_output=settings.SQUARE_OUTPUT,
                    skip_done_part=False,
                )
            )

            nso_tif_kernel_iterator_generator.predict_all_output()
            print(
                "File stored here: "
                + str(nso_tif_kernel_iterator_generator.return_final_output_path())
            )

            if settings.HEXAGON_OUTPUT:
                print(
                    "Making hexagons with resolution "
                    + str(settings.HEXAGON_RESOLUTION)
                )
                output_h3_hexagons_from_pixels(
                    nso_tif_kernel_iterator_generator.return_final_output_path(),
                    settings.HEXAGON_RESOLUTION,
                )

        except Exception as e:
            print(e)
