MODEL_PATH = ""  # path to .sav file containing sklearn model
TIF_FILE_INPUT_REGEX = ""  # path to a directory regex with satellite image as .tif files
OUTPUT_PATH = ""  # folder where output should go
OUTPUT_FILENAME = (
    ""  # Filename fo resulsting output file. Ends in either '.shp' or '.geojson'
)
PATH_TO_SCALER = ""  # folder containing scalers, note that naming of scalers in this path should be consistent with name of TIF_FILE
COLUMN_NAMES = []  # names of the columns that are used by the model
