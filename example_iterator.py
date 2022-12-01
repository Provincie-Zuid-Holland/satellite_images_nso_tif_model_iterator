from tif_model_iterator import tif_kernel_iterator
import pickle


if __name__ == '__main__':
    filename = "C:/repos/satellite-images-nso-datascience/models/randomforest_classifier_coepelduynen_contrast_annotations_grid_search_all_data_2019_2022_small.sav"
    loaded_model = pickle.load(open(filename, 'rb'))

    tif_file = "E:/data/coepelduynen/20210709_103835_SV1-01_SV_RD_11bit_RGBI_50cm_KatwijkAanZee_natura2000_coepelduynen_cropped_ndvi_height.tif"
    output_location = "E:/output/test.shp"


    nso_tif_kernel_iterator_generator = tif_kernel_iterator.nso_tif_kernel_iterator_generator(tif_file)


    nso_tif_kernel_iterator_generator.predict_all_output(loaded_model, output_location, parts=3, multiprocessing = True )