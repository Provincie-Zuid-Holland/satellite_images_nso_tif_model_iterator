import glob
import os

from nso_ds_classes.nso_ds_normalize_scaler import scaler_class_all


class NormalizeScalerLoader:
    """
    Loads scaler from file
    Note: This expects tiff_filepath and scaler files to have the same naming convention
    """

    def __init__(
        self, path_to_scaler_files: str, tif_filepath: str, column_names: list
    ):
        self.path_to_scaler_files = path_to_scaler_files
        self.tif_filepath = tif_filepath
        self.column_names = column_names

    def load(self) -> scaler_class_all:
        tif_file_name = os.path.basename(self.tif_filepath)
        band_filepaths = glob.glob(f"{self.path_to_scaler_files}/{tif_file_name}*")

        return scaler_class_all(
            scaler_file_array=band_filepaths, column_names=self.column_names
        )
