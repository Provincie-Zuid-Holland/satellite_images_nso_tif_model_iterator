import pickle

from tif_model_iterator import tif_kernel_iterator
from loaders.normalize_scaler_loader import NormalizeScalerLoader
from src.filenames.file_name_generator import OutputFileNameGenerator
import settings

if __name__ == '__main__':
    
    filename = settings.MODEL_PATH
    loaded_model = pickle.load(open(filename, 'rb'))

    tif_file = settings.TIF_FILE
    output_path = settings.OUTPUT_PATH
    output_file_name = settings.OUTPUT_FILENAME
    path_to_scaler = settings.PATH_TO_SCALER

    scaler_loader = NormalizeScalerLoader(path_to_scaler_files=path_to_scaler, tif_filepath=tif_file)
    scaler = scaler_loader.load()

    output_file_name_generator = OutputFileNameGenerator(output_path=output_path, output_file_name=output_file_name)

    nso_tif_kernel_iterator_generator = tif_kernel_iterator.tif_kernel_iterator_generator(tif_file)

    nso_tif_kernel_iterator_generator.predict_all_output(
        loaded_model,
        output_file_name_generator=output_file_name_generator,
        parts=20,
        multiprocessing=True,
        normalize_scaler=scaler
    )
