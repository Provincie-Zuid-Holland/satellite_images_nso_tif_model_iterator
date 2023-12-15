import os

from nso_ds_classes.nso_ds_normalize_scaler import scaler_class_all


class NormalizeScalerLoader:
    def __init__(self, path_to_scaler_files: str, tif_filepath: str):
        self.path_to_scaler_files = path_to_scaler_files
        self.tif_filepath = tif_filepath

    def load(self) -> scaler_class_all:
        tif_file_name = os.path.basename(self.tif_filepath)
        file_bands = [
            os.path.join(self.path_to_scaler_files, tif_file_name + f"_band{band_id}.save")
            for band_id in range(1, 7)
        ]

        return scaler_class_all(
            scaler_file_band1=file_bands[0],
            scaler_file_band2=file_bands[1],
            scaler_file_band3=file_bands[2],
            scaler_file_band4=file_bands[3],
            scaler_file_band5=file_bands[4],
            scaler_file_band6=file_bands[5],
        )
