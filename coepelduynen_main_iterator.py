from tif_model_iterator import tif_kernel_iterator
import pickle
import settings
import glob
import pickle


if __name__ == '__main__':
    
    filename = settings.MODEL_PATH
    loaded_model = pickle.load(open(filename, 'rb'))


    tif_files = [file for file in glob.glob("E:/data/coepelduynen/2023*PNEO*re_ndvi*asphalt*.tif")]


    for tif_file in tif_files[0:1]:
        print("-----")
        print(tif_file)

        tif_file = tif_file.replace("\\","/")
        output_location = "E:/output/Coepelduynen_segmentations/"+tif_file.split("/")[-1].replace(".tif", ".shp").replace("\\","/")
        date = output_location.split("/")[-1].split("_")[0]

        nso_tif_kernel_iterator_generator = tif_kernel_iterator.tif_kernel_iterator_generator(tif_file)

        #print(filename)
        #print(filename.split("/")[-1].split["_"])
        #for file in glob.glob("C:/repos/satellite-images-nso-datascience/scalers/ "):

        ascaler = pickle.load(open([file for file in glob.glob("C:/repos/satellite-images-nso-datascience/scalers/"+date+"*.pkl")][0] ,'rb'))
     

        #nso_tif_kernel_iterator_generator.predict_all_output(loaded_model, output_location, parts=10, multiprocessing = True, normalize_scaler="PATH/TO/SCALER" )
        nso_tif_kernel_iterator_generator.predict_all_output(loaded_model, output_location, parts=7, multiprocessing = True, normalize_scaler= ascaler , bands = ascaler.columns_names, aggregate_output = False )