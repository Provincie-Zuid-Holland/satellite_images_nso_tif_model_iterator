from tif_model_iterator import tif_kernel_iterator
import pickle
import settings


if __name__ == '__main__':
    
    filename = settings.MODEL_PATH
    loaded_model = pickle.load(open(filename, 'rb'))

    tif_file = settings.TIF_FILE
    output_location = settings.OUTPUT_PATH


    nso_tif_kernel_iterator_generator = tif_kernel_iterator.tif_kernel_iterator_generator(tif_file)


    nso_tif_kernel_iterator_generator.predict_all_output(loaded_model, output_location, parts=3, multiprocessing = True )