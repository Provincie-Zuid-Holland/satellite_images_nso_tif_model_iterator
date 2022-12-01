from tif_model_iterator import tif_kernel_iterator
import pickle


if __name__ == '__main__':
    filename = "PATH/TO/MODEL/MODEL.sav"
    loaded_model = pickle.load(open(filename, 'rb'))

    tif_file = "PATH/TO/TIF/FILE/FILE.tif"
    output_location = "PATH/TO/OUTPUT/FILE.shp"


    nso_tif_kernel_iterator_generator = tif_kernel_iterator.nso_tif_kernel_iterator_generator(tif_file)


    nso_tif_kernel_iterator_generator.predict_all_output(loaded_model, output_location, parts=3, multiprocessing = True )