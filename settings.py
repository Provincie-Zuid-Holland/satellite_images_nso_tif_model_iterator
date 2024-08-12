# TIF_FILE_INPUT_REGEX = "E:/data/nieuwkoopse_plassen_schippersgat/2023*ndwi*ndvi*.tif"
# TIF_FILE_INPUT_REGEX = "E:/data/nieuwkoopse_plassen_schippersgat/*50cm*.tif"

TIF_FILE_INPUT_REGEX = "E:/data/nieuwkoopse_plassen/*SV*.tif"

# TIF_FILE_INPUT_REGEX = "E:/data/coepelduynen/2023*PNEO*re_ndvi*asphalt*.tif"
# MODEL_PATH = "C:/repos/satellite-images-nso-datascience/saved_models/randomforest_classifier_coepelduynen_contrast_annotations_grid_search_all_data_2019_2022_small_balanced_v1.3.sav"

# OUTPUT_PATH = "E:/output/Coepelduynen_segmentations_test/"
OUTPUT_PATH = "E:/output/nieuwkoopse_plassen/"
# MODEL_PATH = "C:/repos/satellite-images-nso-datascience/saved_models/PNEO_Coepelduynen_20230402_105321_to_20230910_105008_random_forest_classifier.sav"
# MODEL_PATH = "C:/repos/satellite-images-nso-datascience/saved_models/PNEO_Schippersgat_20230603_to_20230905_random_forest_classifier.sav"
NUMBER_OF_PARTS = 4
MODEL_PATH = "C:/repos/satellite-images-nso-datascience/saved_models/Sentinel2_Nieuw_Koopse_Plassen_2023-06-04-00_0_to_2023-06-04-00_0_random_forest_classifier.sav"
# COLUMN_NAMES = ["r", "g", "b", "n", "e", "d", "ndvi", "re_ndvi"]
COLUMN_NAMES = ["r", "g", "b", "i"]
OUTPUT_EXTENTSION = ".parquet"
HEXAGON_OUTPUT = True
HEXAGON_RESOLUTION = 12
DISSOLVE_PARTS = False
SQUARE_OUTPUT = False
# COLUMN_NAMES = ["r", "g", "b", "i", "ndvi", "height"]


# TIF_FILE = "E:/data/coepelduynen/20230402_105321_PNEO-03_1_49_30cm_RD_12bit_RGBNED_Zoeterwoude_natura2000_coepelduynen_cropped_ndvi_re_ndvi.tif"
# OUTPUT_PATH = "E:/output/Coepelduynen_segmentations/20230402_105321_PNEO-03_1_49_30cm_RD_12bit_RGBNED_Zoeterwoude_natura2000_coepelduynen_cropped_ndvi_re_ndvi.shp"
