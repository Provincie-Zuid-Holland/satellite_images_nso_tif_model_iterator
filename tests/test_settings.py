import os

model_path_sv = "./tests/test_input/model_tests/Superview_Nieuwkoopse_plassen_20190302_to_20221012_random_forest_classifier_test.sav"
hexagon_test_file = "./tests/test_input/hexagon_input/20230905_105231_PNEO-03_1_1_30cm_RD_12bit_RGBNED_Uithoorn_Nieuwkoopse_Plassen_De_Haeck_cropped_ndwi_re_ndvi_Waterplants_test.parquet"
model_path_pneo = "./tests/test_input/model_tests/PNEO_Schippersgat_20230905_105231_to_20230905_105231_random_forest_classifier.sav"
tif_file_input = "./tests/test_input/Nieuwkoopse_plassen_raster_images/"
output_path_test = "./tests/test_output/"

a_tif_file_hexagon_test = (
    os.path.abspath(output_path_test)
    + "/input_data/20230905_105231_PNEO-03_1_1_30cm_RD_12bit_RGBNED_Uithoorn_Nieuwkoopse_Plassen_De_Haeck_cropped_ndwi_re_ndvi_Ground_test.tif"
)
