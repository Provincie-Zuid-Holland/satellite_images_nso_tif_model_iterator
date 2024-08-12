# A regex filepath for glob to find downloaded NSO .tif files with
TIF_FILE_INPUT_REGEX = "path/data/tif_files/*.tif"

# Where to store annotations
OUTPUT_PATH = "E:/output/nieuwkoopse_plassen/"

# Path to a model to be inferred.
MODEL_PATH = "path/to/model/PNEO_Nieuwkoopse_plassen_20230603_to_20230905_random_forest_classifier.sav"
# Bands which should be in the .tf files
COLUMN_NAMES = ["r", "g", "b", "n", "e", "d", "ndvi", "re_ndvi"]
# Which file extentsion the parts and the final results should be stored, not a GEOJSON has to be squared and dissolved output
OUTPUT_EXTENTSION = ".parquet"
DISSOLVE_PARTS = False
SQUARE_OUTPUT = False
HEXAGON_OUTPUT = True
HEXAGON_RESOLUTION = 13
