import pystac_client
import planetary_computer
import xarray as xr
from odc.stac import stac_load
from pyproj import Proj, Transformer, CRS
import geopandas as gpd
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import rioxarray as rxr
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed


class SatelliteDataExtractor:
    def __init__(self, lower_left, upper_right, time_window):
        """
        Initialize the data processor with bounding coordinates and time window.
        """
        self.lower_left = lower_left
        self.upper_right = upper_right
        self.bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])
        self.time_window = time_window

    def look_for_data(self, collections, query):
        """
        Searches for satellite data matching the criteria.

        Parameters:
        collections (list): List of data collections (e.g., ["sentinel-2-l2a"]).
        query (dict): Query parameters such as cloud cover limits.

        Returns:
        tuple: (items, signed_items)
        """
        stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = stac.search(
            bbox=self.bounds, 
            datetime=self.time_window,
            collections=collections,
            query=query,
        )
        
        items = list(search.get_items())
        print('Number of scenes that touch our region:', len(items))

        signed_items = [planetary_computer.sign(item).to_dict() for item in items]
        return items, signed_items

    def get_data(self, collections, query, bands, resolution):
        """
        Retrieves satellite data based on bounding box and query.

        Parameters:
        collections (list): List of satellite collections.
        query (dict): Query filters (e.g., cloud cover).
        bands (list): List of bands to fetch.
        resolution (float): Resolution in degrees.

        Returns:
        xarray.Dataset: Retrieved satellite data.
        """
        items, _ = self.look_for_data(collections, query)
        data = stac_load(
            items,
            bands=bands,
            crs="EPSG:4326", # Latitude-Longitude
            resolution=resolution,
            chunks={"x": 2048, "y": 2048},
            dtype="float64",
            patch_url=planetary_computer.sign,
            bbox=self.bounds,
            resampling='bilinear',
        )
        print("Data loaded!")
        return data

    def process_data(self, dataset_type, filename, bands, scene):
        """
        Processes and saves satellite data based on dataset type.

        Parameters:
        dataset_type (str): "sentinel2" or "landsat".
        filename (str): Output file path.
        bands (list): List of bands to save.
        scene (int): Scene index for saving data.
        """
        if dataset_type.lower() == "sentinel2":
            print("Processing Sentinel-2 data...")
            print(f"Collecting data of {len(bands)} bands including {bands}")
            raw_data = self.get_data(
                collections=["sentinel-2-l2a"],
                query={"eo:cloud_cover": {"lt": 30}},
                bands=bands,
                resolution=10/111320.0,
            )
            print(f"Using Bounds: {self.bounds}, Time Window: {self.time_window}")
            print("Calculating the bands median...")
            processed_data = raw_data.median(dim="time").compute()
        
        elif dataset_type.lower() == "landsat":
            print("Processing Landsat data...")
            print(f"Collecting data of {len(bands)} bands including {bands}")
            raw_data = self.get_data(
                collections=["landsat-c2-l2"],
                query={"eo:cloud_cover": {"lt": 50}, "platform": {"in": ["landsat-8"]}},
                bands=["lwir11"],
                resolution=30/111320.0,
            )
            print(f"Using Bounds: {self.bounds}, Time Window: {self.time_window}")
            print("Scaling landsat data...")
            # scale and convert temperature to celcius
            scale = 0.00341802 
            offset = 149.0 
            kelvin_celsius = 273.15
            processed_data = raw_data.astype(float) * scale + offset - kelvin_celsius
        
        else:
            raise ValueError("Invalid dataset type. Choose either 'sentinel2' or 'landsat'.")
        
        print("Preparing data for saving...")
        self.save_data(processed_data, filename, bands, scene)

    def save_data(self, data, filename, bands, scene=None):
        """
        Saves the processed data to a GeoTIFF file.

        Parameters:
        data (xarray.Dataset): Dataset to be saved.
        filename (str): Output file path.
        bands (list): List of bands to save.
        scene (int, optional): Scene index (if applicable).
        """
        # Check if "time" exists before selecting a scene
        if "time" in data.dims and scene is not None:
            data_slice = data.isel(time=scene)
        else:
            data_slice = data  # No time dimension, use full dataset

        height = data_slice.dims["latitude"]
        width = data_slice.dims["longitude"]

        gt = rasterio.transform.from_bounds(
            self.lower_left[1], self.lower_left[0], 
            self.upper_right[1], self.upper_right[0], 
            width, height
        )
        data_slice.rio.write_crs("epsg:4326", inplace=True)
        data_slice.rio.write_transform(transform=gt, inplace=True)

        with rasterio.open(
            filename, 'w', driver='GTiff', width=width, height=height,
            crs='epsg:4326', transform=gt, count=len(bands),
            compress='lzw', dtype='float64'
        ) as dst:
            for i, band in enumerate(bands, start=1):
                dst.write(getattr(data_slice, band), i)
        
        print(f"Data saved: {filename}\n")

class SatelliteDataManipulator:
    def __init__(self, sen2_tiff_path, land_tiff_path, ground_df_path, band_mappings, buffer_distances=[0]):
        self.sen2_tiff_path = sen2_tiff_path
        self.land_tiff_path = land_tiff_path
        self.ground_df_path = ground_df_path
        self.band_mappings = band_mappings
        self.buffer_distances = buffer_distances

        # Load the GeoTIFF data
        print("Loading GeoTIFF data...")
        self.tiff_data = {
            "sentinel": rxr.open_rasterio(self.sen2_tiff_path),
            "landsat": rxr.open_rasterio(self.land_tiff_path)
        }
        self.tiff_crs = self.tiff_data["sentinel"].rio.crs  # Assume both datasets share the same CRS
        print("Data loaded!")

        # Load ground truth data
        print("Loading ground truth data...")
        self.ground_df = pd.read_csv(self.ground_df_path)
        self.latitudes = self.ground_df['Latitude'].values
        self.longitudes = self.ground_df['Longitude'].values
        print("Data loaded!")

        # Convert lat/lon to the TIFF's CRS
        self.transformer = Transformer.from_crs("EPSG:4326", self.tiff_crs, always_xy=True)
        self.transformed_coords = [self.transformer.transform(lon, lat) for lat, lon in zip(self.latitudes, self.longitudes)]

    def map_satellite_data(self):
        """
        """
        band_values = {f"{band_name}": [] for band_name in self.band_mappings.keys()}
        
        # Extract pixel values
        for (x, y) in tqdm(self.transformed_coords, total=len(self.transformed_coords), desc="Mapping values"):
            for band_name, (dataset_key, band_num) in self.band_mappings.items():
                dataset = self.tiff_data[dataset_key]  # Select correct dataset
                value = dataset.sel(x=x, y=y, band=band_num, method="nearest").values
                band_values[f"{band_name}"].append(value)
        
        return pd.DataFrame(band_values)

    def map_satellite_data1(self):
        """
        Extracts band values at the given coordinates with optional focal buffer processing.
        """
        def _compute_focal_mean(self, dataset, x, y, band_num, buffer_distance):
            """
            Computes the focal mean of a band within a buffer distance with boundary checks.

            Parameters:
            - dataset: xarray dataset for Sentinel/Landsat
            - x, y: coordinates in raster CRS
            - band_num: band number to extract
            - buffer_distance: radius (in meters) for the buffer

            Returns:
            - Mean value of the band within the buffer
            """
            resolution = abs(dataset.rio.resolution()[0])  # Pixel resolution in meters
            buffer_pixels = int(buffer_distance / resolution)  # Convert buffer to pixel size

            # Calculate the boundaries of the buffer window
            x_min = x - buffer_pixels
            x_max = x + buffer_pixels
            y_min = y - buffer_pixels
            y_max = y + buffer_pixels

            # Ensure the window stays within the bounds of the raster
            x_min = max(x_min, dataset.x.min())
            x_max = min(x_max, dataset.x.max())
            y_min = max(y_min, dataset.y.min())
            y_max = min(y_max, dataset.y.max())

            # Debugging prints to check if the window selection is correct
            print(f"Buffer={buffer_distance}m | Transformed coords (x, y): ({x}, {y})")
            print(f"Window coordinates: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

            # Extract a square window around (x, y) with the buffer size
            window = dataset.sel(x=slice(x_min, x_max),
                                y=slice(y_min, y_max),
                                band=band_num,
                                method="nearest")

            # Handle case where the window may not contain any valid data
            if window.isnull().all():
                print(f"Warning: No valid data found in the buffer window for coords ({x}, {y}) with buffer {buffer_distance}m.")
                return float('nan')  # or another placeholder for missing values

            return float(window.mean().values)  # Compute mean and convert to float
        
        # Initialize dictionary for extracted values
        band_values = {f"{band_name}_buffer{buffer}m": [] for band_name in self.band_mappings.keys() for buffer in self.buffer_distances}

        # Extract pixel values
        for (x, y) in tqdm(self.transformed_coords, total=len(self.transformed_coords), desc="Mapping values"):
            for band_name, (dataset_key, band_num) in self.band_mappings.items():
                dataset = self.tiff_data[dataset_key]  # Select correct dataset

                for buffer in self.buffer_distances:
                    if buffer == 0:  # No buffer, extract single pixel value
                        try:
                            value = dataset.sel(x=x, y=y, band=band_num, method="nearest").values
                        except KeyError:
                            print(f"Band {band_num} not found in {dataset}")
                        band_values[f"{band_name}_buffer{buffer}m"].append(value)
                    else:  # Apply focal mean within the buffer
                        value = _compute_focal_mean(dataset, x, y, band_num, buffer)
                        band_values[f"{band_name}_buffer{buffer}m"].append(value)

        # Convert extracted values to DataFrame
        extracted_df = pd.DataFrame(band_values)

        return extracted_df

    def calc_band_indices(self, data):
        """
        Compute NDVI, NDBI, NDWI, EVI, SAVI, and NBAI for all buffer sizes.
        """
        # Initialize dictionary for indices
        indices = {
            "NDVI": (data["B08"] - data["B04"]) / (data["B08"] + data["B04"]),
            "NDBI": (data["B11"] - data["B08"]) / (data["B11"] + data["B08"]),
            "NDWI": (data["B03"] - data["B08"]) / (data["B03"] + data["B08"]),
            "EVI": 2.5 * (data["B08"] - data["B04"]) / (data["B08"] + 6 * data["B04"] - 7.5 * data["B02"] + 1),
            "SAVI": ((data["B08"] - data["B04"]) / (data["B08"] + data["B04"] + 0.5)).clip(lower=1e-8) * 1.5,
            "NBAI": ((data["B11"] + data["B12"]) - data["B08"]) / ((data["B11"] + data["B12"]) + data["B08"])
        }

        # Loop through all buffer sizes and apply the indices calculation
        for buffer in self.buffer_distances:
            for index_name, index_value in indices.items():
                # Add buffer size information to column names
                column_name = f"{index_name}_buffer{buffer}m"
                data[column_name] = index_value.replace([np.inf, -np.inf], np.nan)

        return data
    
class WeatherDataProcessor:
    def __init__(self, weather_df_path, satground_df):
        self.weather_df_path = weather_df_path
        self.satground_df = satground_df
        
        # Load the weather data
        print("Loading weather data...")
        self.bronx_weather = pd.read_excel(f"{self.weather_df_path}", sheet_name="Bronx", engine="openpyxl")
        self.manhattan_weather = pd.read_excel(f"{self.weather_df_path}", sheet_name="Manhattan", engine="openpyxl")
        
    def classify_weather_data(self):
        """
        Classify weather data into different locations."""
        def classify_location(lat, lon):
            if 40.70 <= lat <= 40.88 and -74.01 <= lon <= -73.90:
                return "Manhattan"
            elif 40.78 <= lat <= 40.91 and -73.93 <= lon <= -73.78:
                return "Bronx"
            else:
                return "Unknown"
        # Apply function to dataset
        self.satground_df['location'] = self.satground_df.apply(lambda row: classify_location(row['Latitude'], row['Longitude']), axis=1)
        print ("Mapped location name with the corresponding lat & lon")
        return self.satground_df
        
    def process_weather_data(self):
        """
        """
        print("Processing weather data...")
        # Rename the column to 'datetime'
        self.bronx_weather = self.bronx_weather.rename(columns={"Date / Time": "datetime"})
        self.manhattan_weather = self.manhattan_weather.rename(columns={"Date / Time": "datetime"})
        
        # Convert satground_df to datetime dtype
        self.satground_df['datetime'] = pd.to_datetime(self.satground_df['datetime'], format='%d-%m-%Y %H:%M')
        # Convert weather data to datetime dtype, remove timezone sign
        self.bronx_weather['datetime'] = pd.to_datetime(self.bronx_weather['datetime']).dt.tz_localize(None)
        self.manhattan_weather['datetime'] = pd.to_datetime(self.manhattan_weather['datetime']).dt.tz_localize(None)

        # Add 'location' column to weather DataFrames
        self.bronx_weather['location'] = 'Bronx'
        self.manhattan_weather['location'] = 'Manhattan'

        # Concatenate both weather datasets
        weather_df = pd.concat([self.bronx_weather, self.manhattan_weather])

        # Ensure all DataFrames are sorted by datetime
        self.satground_df = self.satground_df.sort_values('datetime')
        weather_df = weather_df.sort_values('datetime')
        print("Processing completed!")

        # Merge using 'Datetime' and 'location'
        uhi_data = pd.merge_asof(self.satground_df, weather_df, on='datetime', by='location')
        print("Successfully merged satground and weather dataset! Let's take a look at the final dataset:")
        uhi_data.head()    
        # Return the final df
        return uhi_data
    
    
def drop_duplicate(df):
    """
    Drop duplicate rows in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    # Remove duplicate rows from the DataFrame based on specified columns and keep the first occurrence
    columns_to_check = df.drop(columns=['Longitude', 'Latitude', 'datetime', 'UHI Index']).columns
    for col in columns_to_check:
        # Check if the value is a numpy array and has more than one dimension
        df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) and x.ndim > 0 else x)

    # Now remove duplicates
    df = df.drop_duplicates(subset=columns_to_check, keep='first')
    df.describe()
    return df
