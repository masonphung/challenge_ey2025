import pystac_client
import planetary_computer
from odc.stac import stac_load
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point
import pandas as pd
import numpy as np


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

    def process_data(self, dataset_type, filename, bands, scene=None):
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
            # Scale and convert temperature to celcius
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
    def __init__(self, sentinel_tiff, landsat_tiff, ground_df_path, sentinel_bands, landsat_bands, buffer_sizes=[0]):
        self.sentinel_tiff = sentinel_tiff
        self.landsat_tiff = landsat_tiff
        self.ground_df_path = ground_df_path
        self.sentinel_bands = sentinel_bands
        self.landsat_bands = landsat_bands
        self.buffer_sizes = buffer_sizes
        
    def extract_band_values(self):
        """
        Extract Sentinel-2 & Landsat band values at given locations, with optional focal buffers.
        
        Parameters:
        - sentinel_tiff (str): Path to Sentinel-2 GeoTIFF.
        - landsat_tiff (str): Path to Landsat GeoTIFF.
        - ground_df_path (str): Path to CSV containing Latitude, Longitude.
        - sentinel_bands (list): List of Sentinel-2 bands to extract (e.g., ["B02", "B03"]).
        - landsat_bands (list): List of Landsat bands to extract (e.g., ["lwir11"]).
        - buffer_sizes (list): List of buffer sizes in meters (e.g., [0, 50, 100]).

        Returns:
        - pd.DataFrame: Dataframe with extracted band values (with buffer names if applicable).
        """
        print(f"Chosen sentinel bands: {self.sentinel_bands}")
        print(f"Chosen landsat bands: {self.landsat_bands}")
        print(f"Buffer sizes: {self.buffer_sizes}")
        # Band-to-index mapping based on the input sentinel bands
        sentinel_band_map = {band: idx+1 for idx, band in enumerate(self.sentinel_bands)}
        
        # For Landsat, assuming one band is always 'lwir11'
        landsat_band_map = {band: 1 for band in self.landsat_bands}

        # Read ground data
        ground_df = pd.read_csv(self.ground_df_path)
        geometry = [Point(xy) for xy in zip(ground_df["Longitude"], ground_df["Latitude"])]
        ground_gdf = gpd.GeoDataFrame(ground_df, geometry=geometry, crs="EPSG:4326")

        # Function to extract value at a point with optional buffer
        def extract_value(src, band_index, x, y, buffer):
            row, col = src.index(x, y)
            if buffer == 0:
                return src.read(band_index)[row, col]
            else:
                res = src.res[0]  # Assume square pixels, get resolution in degrees
                buffer_pixels = int(buffer / (res * 111320))  # Convert meters to pixel units
                window = Window(col - buffer_pixels, row - buffer_pixels, 2 * buffer_pixels + 1, 2 * buffer_pixels + 1)
                data = src.read(band_index, window=window)
                return np.nanmean(data)  # Return mean value within buffer

        # Extract values
        extracted_data = []
        
        
        print("Extracting band values...")
        for _, row in ground_gdf.iterrows():
            x, y = row["Longitude"], row["Latitude"]
            extracted_row = {}

            with rasterio.open(self.sentinel_tiff) as sen_src, rasterio.open(self.landsat_tiff) as land_src:
                for band in self.sentinel_bands:
                    band_index = sentinel_band_map[band]
                    for buffer in self.buffer_sizes:
                        feature_name = f"{band}" if buffer == 0 else f"{band}_buffer{buffer}"
                        extracted_row[feature_name] = extract_value(sen_src, band_index, x, y, buffer)

                for band in self.landsat_bands:
                    band_index = landsat_band_map[band]
                    for buffer in self.buffer_sizes:
                        feature_name = f"{band}" if buffer == 0 else f"{band}_buffer{buffer}"
                        extracted_row[feature_name] = extract_value(land_src, band_index, x, y, buffer)

            extracted_data.append(extracted_row)

        return pd.DataFrame(extracted_data)

    def calc_band_indices(self, data):
        """
        Calculate the following indices: NDVI, NDBI, NDWI, EVI, SAVI, NBAI from the extracted satellite data:
        Returns:
        data (DataFrame): The satellite data with newly calculated band combination indices
        """
        # Define the indices calculations
        indices = {}
        for each in self.buffer_sizes:
            if each == 0:
                print("Calculating combination indices without buffer...")
                indices.update(
                    {
                        "NDVI": (data["B08"] - data["B04"]) / (data["B08"] + data["B04"]),
                        "NDBI": (data["B11"].astype("float64") - data["B08"].astype("float64")) / (data["B11"].astype("float64") + data["B08"].astype("float64")),
                        "NDWI": (data["B03"] - data["B08"]) / (data["B03"] + data["B08"]),
                        "MNDWI": (data["B03"] - data["B11"]) / (data["B03"] + data["B11"]),
                        "EVI": 2.5 * (data["B08"] - data["B04"]) / (data["B08"] + 6 * data["B04"] - 7.5 * data["B02"] + 1),
                        "SAVI": ((data["B08"] - data["B04"]) / (data["B08"] + data["B04"] + 0.5)) * 1.5,
                        "NBAI": ((data["B11"] + data["B12"]) - data["B08"]) / ((data["B11"] + data["B12"]) + data["B08"])
                    }
                )
            else:
                print(f"Calculating combination indices with buffer size of {each}m...")
                indices.update(
                    {
                        f"NDVI_buffer{each}": (data[f"B08_buffer{each}"] - data[f"B04_buffer{each}"]) / (data[f"B08_buffer{each}"] + data[f"B04_buffer{each}"]),
                        f"NDBI_buffer{each}": (data[f"B11_buffer{each}"] - data[f"B08_buffer{each}"]) / (data[f"B11_buffer{each}"] + data[f"B08_buffer{each}"]),
                        f"NDWI_buffer{each}": (data[f"B03_buffer{each}"] - data[f"B08_buffer{each}"]) / (data[f"B03_buffer{each}"] + data[f"B08_buffer{each}"]),
                        f"MNDWI_buffer{each}": (data[f"B03_buffer{each}"] - data[f"B11_buffer{each}"]) / (data[f"B03_buffer{each}"] + data[f"B11_buffer{each}"]),
                        f"EVI_buffer{each}": 2.5 * (data[f"B08_buffer{each}"] - data[f"B04_buffer{each}"]) / (data[f"B08_buffer{each}"] + 6 * data[f"B04_buffer{each}"] - 7.5 * data[f"B02_buffer{each}"] + 1),
                        f"SAVI_buffer{each}": ((data[f"B08_buffer{each}"] - data[f"B04_buffer{each}"]) / (data[f"B08_buffer{each}"] + data[f"B04_buffer{each}"] + 0.5)) * 1.5,
                        f"NBAI_buffer{each}": ((data[f"B11_buffer{each}"] + data[f"B12_buffer{each}"]) - data[f"B08_buffer{each}"]) / ((data[f"B11_buffer{each}"] + data[f"B12_buffer{each}"]) + data[f"B08_buffer{each}"])
                    }
                )

        # Compute and clean each index dynamically
        for index_name, index_value in indices.items():
            data[index_name] = index_value.replace([np.inf, -np.inf], np.nan).astype("float64")
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

