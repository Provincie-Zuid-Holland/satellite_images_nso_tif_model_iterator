import h3
import pyproj
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon


def output_h3_hexagons(final_output_path, resolution=12):
    """
    Makes single geo points output into hexagon output.


    @param final_output_path: path to a row pixelbased point data.
    @param resolution: the resolution of the hexagon, look at https://h3geo.org/docs/core-library/restable/ for details
    """

    # TODO: Make clear exception that the data is not in order!
    # if self.square_output:
    #    raise "Hexagons can only be made with none square for now! Rerun with False Square output"

    # if self.dissolve_parts:
    #    raise "Hexagons can only be made with none dissolved polygons for now! Rerun with false dissolve parts!"

    # Define the transformation using pyproj.Transformer

    # TODO: Not make a assumption about the base crs!
    transformer = pyproj.Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

    # final_output_path = self.output_file_name_generator.generate_final_output_path()
    df = []

    if ".csv" in final_output_path:
        df = pd.read_csv(final_output_path)

    if ".parquet" in final_output_path:
        df = pd.read_parquet(final_output_path)

    if len(df) == 0:
        raise "Result output dataframe not loaded correctly"

    if "Unnamed: 0" in df.columns:
        df = df.drop(["Unnamed: 0"], axis=1)

    if "geometry" in df.columns:
        df = df.drop(["geometry"], axis=1)

    df["lon"], df["lat"] = transformer.transform(df["rd_x"].values, df["rd_y"].values)

    # match the point with a hexagon id
    df["hexagon_id"] = df.apply(
        lambda x: h3.geo_to_h3(x["lon"], x["lat"], resolution), axis=1
    )

    # group by the hexagon id and use the most frequent label.
    df = (
        df.groupby("hexagon_id")["label"]
        .apply(lambda x: x.mode().iloc[0])
        .reset_index()
    )

    # Get the polygon for the hexagon.
    df["geometry"] = df.apply(
        lambda x: Polygon(h3.h3_to_geo_boundary(x["hexagon_id"])), axis=1
    )

    # Export results to a .geojson
    gpd.GeoDataFrame(df, geometry=df["geometry"]).to_file(
        final_output_path.split(".")[0]
        + "_hexagons_res"
        + str(resolution)
        + ".geojson",
        driver="GeoJSON",
    )
