from lxml import etree
import numpy as np
import pandas as pd
from xml.dom import minidom
from shapely.geometry import Polygon, Point, box


class process_KML:
    def __init__(self, kml_path, cell_sizes, feature_names):
        self.kml_path = kml_path
        self.df = None
        self.feature_names = feature_names
        self.polygons = self.parse_kml(path=self.kml_path)
        self.min_lat, self.max_lat, self.min_lon, self.max_lon = self.extract_bounding_box(self.kml_path)      
        # Initialize empty dataframes
        self.train_dfs = {}
        self.test_dfs = {}
        # Step 1: Load the original dataset
        self.train_df = pd.read_csv("data/raw/Training_data_uhi_index_2025-02-18.csv")
        self.test_df = pd.read_csv("data/test/Submission_template.csv")

        for cell_size in cell_sizes:
            print(f"Processing for cell size: {cell_size}")


            # Step 3: Create the grid cells within the bounding box
            grid_cells = self.create_grid(self.min_lat, self.max_lat, self.min_lon, self.max_lon, cell_size)

            # Calculate the building coverage and count for each grid cell
            coverage = self.calculate_coverage(building_footprints=self.polygons, grid_cells=grid_cells)
            building_counts = self.building_count(building_footprints=self.polygons, grid_cells=grid_cells)

            # Assign coverage and count to training and test datasets
            train_df_copy = self.assign_coverage_and_count_to_points(
                self.train_df, grid_cells, 
                coverage, building_counts
            )
            test_df_copy = self.assign_coverage_and_count_to_points(
                self.test_df, grid_cells, 
                coverage, building_counts
            )

            # Rename columns to indicate cell size
            for feature in self.feature_names:
                train_df_copy.rename(columns={feature: f"{feature}_{cell_size}"}, inplace=True)
                test_df_copy.rename(columns={feature: f"{feature}_{cell_size}"}, inplace=True)

            # Store the processed dataframes
            self.train_dfs[cell_size] = train_df_copy
            self.test_dfs[cell_size] = test_df_copy

            # Print results inside the loop
            print(f"Statistics:")
            print("Training set:")
            for feature in self.feature_names:
                print(f"{feature} mean: {round(train_df_copy[f'{feature}_{cell_size}'].mean(),2)}")
            print("Test set:")
            for feature in self.feature_names:
                print(f"{feature} mean: {round(test_df_copy[f'{feature}_{cell_size}'].mean(),2)}")
            print("\n")

        # Merge datasets on their common index (e.g., lat/lon or existing keys in train_df/test_df)
        combined_train_df = self.train_df.copy()
        combined_test_df = self.test_df.copy()

        for cell_size in cell_sizes:
            combined_train_df = combined_train_df.merge(
                self.train_dfs[cell_size], 
                on=['Latitude', 'Longitude', 'datetime', 'UHI Index'], 
                how='left'
            )
            combined_test_df = combined_test_df.merge(
                self.test_dfs[cell_size], 
                on=['Latitude', 'Longitude', 'UHI Index'], 
                how='left'
            )

        print("Data merged with the final ground training dataset")
        print("Data merged with the final ground test dataset\n")

        # Save the combined datasets
        path = "data/interim"
        combined_train_df.to_csv(f"{path}/ground_df.csv", index=False)
        combined_test_df.to_csv(f"{path}/test_ground_df.csv", index=False)

        # Print a preview of the combined dataset
        print(f"Combined training data saved at:{path}/ground_df.csv")
        print(f"Combined test data saved at: {path}/test_ground_df.csv")

        
    def parse_kml(self,path):
        """
        Parse a KML file and extract building footprints as polygons.
        Handles both two and three component coordinates.
        """
        kml = minidom.parse(path)
        polygons = []
        
        # Loop through each Placemark in the KML file
        for placemark in kml.getElementsByTagName("Placemark"):
            coords_text = placemark.getElementsByTagName("coordinates")[0].firstChild.nodeValue.strip()
            coords = []
            for coord in coords_text.split():
                # Split each coordinate into components
                coord_parts = coord.split(',')
                if len(coord_parts) == 2:
                    lon, lat = map(float, coord_parts)  # Only longitude and latitude
                elif len(coord_parts) == 3:
                    lon, lat, _ = map(float, coord_parts)  # Ignore altitude
                else:
                    continue  # Skip invalid coordinates
                coords.append((lon, lat))
            polygons.append(Polygon(coords))  # Create a Polygon for each footprint
        return polygons
    
    def extract_bounding_box(self, path):
        """
        Extracts the bounding box (minimum and maximum latitude and longitude) 
        from a KML file.

        This function:
        1. Parses the KML file and retrieves all coordinate elements.
        2. Iterates through the coordinates to determine the min/max latitude and longitude.
        3. Ignores altitude values if present.
        4. Returns the bounding box as (min_lat, max_lat, min_lon, max_lon).

        Parameters:
        kml_file (str): The file path to the KML file.

        Returns:
        tuple: (min_lat, max_lat, min_lon, max_lon) representing the bounding box.

    """
        tree = etree.parse(path)
        root = tree.getroot()

        # Define namespaces
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}

        # Find all coordinates within the KML file
        coords = root.xpath('.//kml:coordinates', namespaces=ns)

        min_lat, max_lat = float('inf'), float('-inf')
        min_lon, max_lon = float('inf'), float('-inf')

        # Iterate through all coordinates and find the bounding box
        for coord in coords:
            coords_text = coord.text.strip()
            for coord_pair in coords_text.split():
                coord_values = coord_pair.split(',')
                lon = float(coord_values[0])
                lat = float(coord_values[1])

                # Ignore the altitude (if present)
                if len(coord_values) == 3:
                    _ = coord_values[2]  # We don't need altitude

                # Update the bounding box
                min_lat = min(min_lat, lat)
                max_lat = max(max_lat, lat)
                min_lon = min(min_lon, lon)
                max_lon = max(max_lon, lon)
        
        print(f"Bounding Box: \nMin Lat: {min_lat}\nMax Lat: {max_lat}\nMin Lon: {min_lon}\nMax Lon: {max_lon}")
        return min_lat, max_lat, min_lon, max_lon
    
    def create_grid(self, min_lat, max_lat, min_lon, max_lon, cell_size):
        """
        Description: Creates a grid of rectangular cells within a given bounding box.

        This function:
        1. Converts latitude and longitude degrees into approximate meters.
        2. Determines the number of grid cells required in both latitude and longitude directions.
        3. Iterates through the bounding box to generate individual grid cells.
        4. Uses the `shapely.geometry.box` function to create rectangular polygons representing each grid cell.

        Parameters:
        min_lat (float): Minimum latitude of the bounding box.
        max_lat (float): Maximum latitude of the bounding box.
        min_lon (float): Minimum longitude of the bounding box.
        max_lon (float): Maximum longitude of the bounding box.
        cell_size (int, optional): Size of each grid cell in meters (default is 500m).

        Returns:
        list: A list of `shapely.geometry.Polygon` objects representing the grid cells.
        """
        # Convert degrees to meters (approximation at mid-latitude)
        lat_to_meters = 111320  # meters per degree of latitude
        lon_to_meters = 111320 * np.cos(np.radians((max_lat + min_lat) / 2))  # meters per degree of longitude at mid-latitude
        
        # Number of grid cells in each direction
        n_lat_cells = int((max_lat - min_lat) * lat_to_meters / cell_size)
        n_lon_cells = int((max_lon - min_lon) * lon_to_meters / cell_size)
        
        # Create the grid
        grid_cells = []
        for i in range(n_lat_cells):
            for j in range(n_lon_cells):
                # Calculate the bounds of each grid cell
                cell_min_lat = min_lat + i * cell_size / lat_to_meters
                cell_max_lat = min_lat + (i + 1) * cell_size / lat_to_meters
                cell_min_lon = min_lon + j * cell_size / lon_to_meters
                cell_max_lon = min_lon + (j + 1) * cell_size / lon_to_meters
                
                # Create a polygon for the grid cell
                grid_cell = box(cell_min_lon, cell_min_lat, cell_max_lon, cell_max_lat)
                grid_cells.append(grid_cell)
        
        return grid_cells
    
    def calculate_coverage(self, building_footprints, grid_cells):
        coverage = []
        """
        Description: Calculates the total building coverage area within each grid cell. This function
        iterates over each grid cell in the provided list, checks for intersections between the grid cell and building footprints,
        computes the intersection area where buildings overlap with the grid cell, and then aggregates the total building coverage per grid cell.

        Parameters:
        building_footprints: A list of polygons representing building footprints.
        grid_cells: A list of polygons representing the grid cells.

        Returns:
        list: A list of coverage values where each entry corresponds to the total building area within a grid cell.
        """
        # Loop over each grid cell and calculate the intersection with building footprints
        for cell in grid_cells:
            cell_coverage = 0
            for building in building_footprints:
                if building.intersects(cell):  # Direct intersection without buffer
                    # Calculate the intersection area between the building and the grid cell
                    intersection = building.intersection(cell)
                    cell_coverage += intersection.area
            
            # Normalize coverage by dividing by the area of the grid cell
            normalized_coverage = cell_coverage / cell.area if cell.area > 0 else 0
            coverage.append(normalized_coverage)
        
        return coverage
    
    def building_count(self,building_footprints, grid_cells):
        building_counts = []
        """ 
        escription: Computes the number of buildings that intersect each grid cell. This function iterates over a list of grid cells and counts how many building footprints
        intersect each grid cell without applying any buffering.

        Parameters:
        building_footprints (list of shapely.geometry.Polygon): A list of polygons representing building footprints.
        grid_cells (list of shapely.geometry.Polygon): A list of polygons representing the grid cells.

        Returns:
        list: A list where each entry corresponds to the number of buildings intersecting a grid cell.
        """
        
        # Loop through grid cells to count how many buildings intersect each grid cell
        for cell in grid_cells:
            count = sum(1 for building in building_footprints if building.intersects(cell))  # Direct intersection without buffer
            building_counts.append(count)
        
        return building_counts
    
    def assign_coverage_and_count_to_points(self, df, grid_cells, coverage, building_counts):
        """
        Assigns building coverage and building count to each point based on the grid cell it falls into.

        Parameters:
        df (pd.DataFrame): DataFrame containing 'Longitude' and 'Latitude' columns.
        grid_cells (list): List of shapely Polygon objects representing grid cells.
        coverage (list): List of coverage values corresponding to each grid cell.
        building_counts (list): List of building count values corresponding to each grid cell.

        Returns:
        pd.DataFrame: A new DataFrame with 'building_coverage' and 'building_count' columns.
        """
        assigned_coverage = []
        assigned_building_count = []

        for _, row in df.iterrows():
            point = Point(row['Longitude'], row['Latitude'])  # Create a point from longitude and latitude
            
            # Default values if point is not in any grid cell
            point_coverage = 0  
            point_building_count = 0  

            # Check which grid cell the point belongs to
            for i, grid_cell in enumerate(grid_cells):
                if grid_cell.contains(point):
                    point_coverage = coverage[i]  
                    point_building_count = building_counts[i]  
                    break  # Stop searching once the point is found in a grid cell
            
            assigned_coverage.append(point_coverage)
            assigned_building_count.append(point_building_count)
        
        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()
        
        # Add the new columns to the copy of the dataframe
        df_copy['building_coverage'] = assigned_coverage
        df_copy['building_count'] = assigned_building_count
        
        return df_copy