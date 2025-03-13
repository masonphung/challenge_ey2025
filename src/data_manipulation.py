import pandas as pd
import rioxarray as rxr
from pyproj import Proj, Transformer, CRS
import geopandas as gpd
from tqdm import tqdm
import numpy as np

class DataManipulator:
    # Extract values within buffer
    def extract_avg_band(band_index, raster_data):
        """
        Extract average band values within a buffer around each location.
        
        Parameters:
        band_index (int): Band index to extract
        raster_data (xarray.DataArray): Raster data
        
        Returns:
        List: Average band values within buffer
        """
        band_values = []
        # Loop through each geometry and extract average band value
        for geom in tqdm(gdf.geometry, desc=f"Extracting Band {band_index}"):
            # Clip the raster around the geometry
            clipped = raster_data.sel(band=band_index).rio.clip([geom])  # Clip raster using buffer
            # Computes the mean value of the clipped raster
            avg_value = clipped.mean().values.item()  # Get mean value
            # Appends it to band_values
            band_values.append(avg_value)
        return band_values
    
    def map_satellite_data_with_buffer(sen2_tiff_path, land_tiff_path, ground_df_path, buffer_radius=50):
        """
        Extract average band values within a buffer around each location.

        Parameters:
        sen2_tiff_path (str): Path to Sentinel-2 GeoTIFF dataset
        land_tiff_path (str): Path to Land satellite GeoTIFF dataset
        ground_df_path (str): Path to ground truth dataset (CSV)
        buffer_radius (int): Buffer radius (meters) for averaging pixel values

        Returns:
        DataFrame: Extracted band values with buffering
        """

        # Load the GeoTIFF data
        sen_data = rxr.open_rasterio(sen2_tiff_path)
        land_data = rxr.open_rasterio(land_tiff_path)
        tiff_crs = sen_data.rio.crs

        # Read the ground_df, take lat & lon values
        df = pd.read_csv(ground_df_path)

        # Convert lat/lon to raster CRS
        transformer = Transformer.from_crs("EPSG:4326", tiff_crs, always_xy=True)
        df["x"], df["y"] = transformer.transform(df["Longitude"].values, df["Latitude"].values)

        # Convert to GeoDataFrame for spatial operations
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs=tiff_crs)

        # Create buffer zone (circular area around each point)
        gdf["geometry"] = gdf.geometry.buffer(buffer_radius)
        
        # Extract average band values
        df["B01"] = extract_avg_band(1, sen_data)
        df["B02"] = extract_avg_band(2, sen_data)
        df["B03"] = extract_avg_band(3, sen_data)
        df["B04"] = extract_avg_band(4, sen_data)
        df["B06"] = extract_avg_band(5, sen_data)
        df["B08"] = extract_avg_band(6, sen_data)
        df["B11"] = extract_avg_band(7, sen_data)
        df["B12"] = extract_avg_band(8, sen_data)
        df["L11"] = extract_avg_band(1, land_data)

        return df
    
    def calc_band_indices(data):
        """
        Calculate the following indices from the satellite data:
        Parameters:
        data (DataFrame): The satellite data
        Returns:
        data (DataFrame): The satellite data with the indices columns
        """
        # Define the indices calculations
        indices = {
            "NDVI": (data["B08"] - data["B04"]) / (data["B08"] + data["B04"]),
            "NDBI": (data["B11"].astype("float64") - data["B08"].astype("float64")) / (data["B11"].astype("float64") + data["B08"].astype("float64")),
            "NDWI": (data["B03"] - data["B08"]) / (data["B03"] + data["B08"]),
            "EVI": 2.5 * (data["B08"] - data["B04"]) / (data["B08"] + 6 * data["B04"] - 7.5 * data["B02"] + 1),
            "SAVI": ((data["B08"] - data["B04"]) / (data["B08"] + data["B04"] + 0.5)) * 1.5,
            "NBAI": ((data["B11"] + data["B12"]) - data["B08"]) / ((data["B11"] + data["B12"]) + data["B08"])
        }

        # Compute and clean each index dynamically
        for index_name, index_value in indices.items():
            data[index_name] = index_value.replace([np.inf, -np.inf], np.nan).astype("float64")
        return data