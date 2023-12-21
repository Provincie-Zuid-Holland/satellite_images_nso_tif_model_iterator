import os


class OutputFileNameGenerator:
    """
    Generates filenames for output files
    """

    def __init__(self, output_path: str, output_file_name: str):
        """
        Note that supported extensions are `.shp` and `.geojson`
        """
        self.output_path = output_path
        self.extension = output_file_name.split(".")[-1]
        self.base_name = output_file_name[: -len(self.extension) - 1]

    def generate_final_output_path(self) -> str:
        file_name = f"{self.base_name}.{self.extension}"
        return os.path.join(self.output_path, file_name)

    def generate_part_output_path(self, part_number: int) -> str:
        file_name = f"{self.base_name}_part_{part_number}.{self.extension}"
        return os.path.join(self.output_path, file_name)

    def glob_wild_card_for_all_part_files(self) -> str:
        wild_card = f"{self.base_name}_part_*"
        return os.path.join(self.output_path, wild_card)

    def glob_wild_card_for_part_extension_only(self) -> str:
        wild_card = f"{self.base_name}_part_*.{self.extension}"
        return os.path.join(self.output_path, wild_card)
