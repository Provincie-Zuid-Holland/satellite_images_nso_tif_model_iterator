import h3
import pyproj
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from timeit import default_timer as timer


def read_data(path_to_file, crs):
    """

    Function for reading in the geopoint .csv or .parquet file, although .parquet is recommended.

    TODO: For now it assumes the crs has to be in 4326, assuming that the output date is alwaus in 28892, this has to be more dynamic.

    @param path_to_file: path to where the file is located.
    @param crs: the coordinate reference system for the geopoint, for now only 28892 whill be converted to 4326
    """

    # TODO: Make clear exceptions to check if the data is valid.
    if crs is "28892":
        # Make a transformer to convert from 28992 to 4326
        transformer = pyproj.Transformer.from_crs(
            "EPSG:28992", "EPSG:4326", always_xy=True
        )

    df = []

    if ".csv" in path_to_file:
        df = pd.read_csv(path_to_file)

    if ".parquet" in path_to_file:
        df = pd.read_parquet(path_to_file)

    if len(df) == 0:
        raise "Result output dataframe not loaded correctly"

    if "Unnamed: 0" in df.columns:
        df = df.drop(["Unnamed: 0"], axis=1)

    if "geometry" in df.columns:
        df = df.drop(["geometry"], axis=1)

    if crs is "28892":
        df["lon"], df["lat"] = transformer.transform(
            df["rd_x"].values, df["rd_y"].values
        )
    else:
        df["lon"], df["lat"] = df["rd_x"], df["rd_y"]

    return df


def transform_data(df, resolution):
    """
    Maps every geopoint to a hexagon id.
    And then groups by each geopoint on it's hexagon id and takes the value which is most frequent.


    @param df: pandas dataframe with geopoints in it in WGS84
    @param resolution: The resolution of the hexagon
    return a dataframe of groupby geopoints based on their hexagon id's.
    """

    start = timer()
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

    print("Hexagon indexing finished in: " + str(timer() - start) + " second(s)")
    return df


def output_h3_hexagons_from_pixels(path_to_file, resolution=12, crs="28892"):
    """
    Makes single geo points output with label output into a hexagon output.

    @param path_to_file: path to a file which contains row pixel based geopoint data.
    @param resolution: the resolution of the hexagon, look at https://h3geo.org/docs/core-library/restable/ for details.
    @return path to where the file is stored.
    """

    df = read_data(path_to_file, crs)
    df = transform_data(df, resolution)

    output_name = (
        path_to_file.split(".")[0] + "_hexagons_res" + str(resolution) + ".geojson"
    )

    # Export results to a .geojson
    gpd.GeoDataFrame(df, geometry=df["geometry"]).to_file(output_name, driver="GeoJSON")

    return output_name
