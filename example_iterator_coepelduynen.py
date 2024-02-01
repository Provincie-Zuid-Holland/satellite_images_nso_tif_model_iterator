import pickle

import settings
from src.filenames.file_name_generator import OutputFileNameGenerator
from src.loaders.normalize_scaler_loader import NormalizeScalerLoader
from src.tif_model_iterator import tif_kernel_iterator
import glob

if __name__ == "__main__":
    filename = settings.MODEL_PATH
    print(filename)
    loaded_model = pickle.load(open(filename, "rb"))

    tif_files = [
        file
        for file in glob.glob("E:/data/coepelduynen/2023*PNEO*re_ndvi*asphalt*.tif")
    ]

    for tif_file in tif_files:
        print("----------")
        print(tif_file)

        tif_file = tif_file.replace("\\", "/")
        output_path = "E:/output/Coepelduynen_segmentations_production/"
        date = output_path.split("/")[-1].split("_")[0]

        output_file_name = (
            "E:/output/Coepelduynen_segmentations_production/"
            + tif_file.split("/")[-1].replace(".tif", ".shp").replace("\\", "/")
        )

        scaler = pickle.load(
            open(
                [
                    file
                    for file in glob.glob(
                        "C:/repos/satellite-images-nso-datascience/scalers/"
                        + date
                        + "*.pkl"
                    )
                ][0],
                "rb",
            )
        )

        output_file_name_generator = OutputFileNameGenerator(
            output_path=output_path, output_file_name=output_file_name
        )

        nso_tif_kernel_iterator_generator = (
            tif_kernel_iterator.TifKernelIteratorGenerator(
                path_to_tif_file=tif_file,
                model=loaded_model,
                output_file_name_generator=output_file_name_generator,
                parts=4,
                normalize_scaler=scaler,
                aggregate_output=2,
            )
        )

        nso_tif_kernel_iterator_generator.predict_all_output()
