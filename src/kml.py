import os
from fastkml import kml
from lxml import etree
import numpy as np
import pandas as pd
from xml.dom import minidom
from shapely.geometry import Polygon

# Get the absolute path of the project's root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
# Change working directory
os.chdir(project_root)

class process_KML:
    def __init__(self, kml_path, buffer_distance):
        self.kml_path = kml_path
        self.df = None
        self.polygons = self.parse_kml()
        self.min_lat, self.max_lat, self.min_lon, self.max_lon = self.extract_bounding_box(self.kml_path)
        self.buffer_distance = buffer_distance
        
    def parse_kml(self):
        """
        Parse a KML file and extract building footprints as polygons.
        Handles both two and three component coordinates.
        """
        kml = minidom.parse(self.file_path)
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
            print(f"Extracted {len(polygons)} building footprints.")
        return polygons
    
    def extract_bounding_box(self):
        tree = etree.parse(self.kml_path)
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

        return min_lat, max_lat, min_lon, max_lon
    
    def create_grid(self, cell_size=500):
        # Convert degrees to meters (approximation at mid-latitude)
        lat_to_meters = 111320  # meters per degree of latitude
        lon_to_meters = 111320 * np.cos(np.radians((self.max_lat + self.min_lat) / 2))  # meters per degree of longitude at mid-latitude
        
        # Number of grid cells in each direction
        n_lat_cells = int((self.max_lat - self.min_lat) * lat_to_meters / cell_size)
        n_lon_cells = int((self.max_lon - self.min_lon) * lon_to_meters / cell_size)
        
        # Create the grid
        grid_cells = []
        for i in range(n_lat_cells):
            for j in range(n_lon_cells):
                # Calculate the bounds of each grid cell
                cell_min_lat = self.min_lat + i * cell_size / lat_to_meters
                cell_max_lat = self.min_lat + (i + 1) * cell_size / lat_to_meters
                cell_min_lon = self.min_lon + j * cell_size / lon_to_meters
                cell_max_lon = self.min_lon + (j + 1) * cell_size / lon_to_meters
                
                # Create a polygon for the grid cell
                grid_cell = self.box(cell_min_lon, cell_min_lat, cell_max_lon, cell_max_lat)
                grid_cells.append(grid_cell)
        
        return grid_cells
    
    def calculate_coverage_with_buffer(self, building_footprints, grid_cells):
        coverage = []
        # Loop over each grid cell and calculate the intersection with building footprints (with buffer)
        for cell in grid_cells:
            cell_coverage = 0
            for building in building_footprints:
                # Buffer the building footprint by the given distance (in meters)
                buffered_building = building.buffer(self.buffer_distance)

                if buffered_building.intersects(cell):
                    # Calculate the intersection area between the buffered building and the grid cell
                    intersection = buffered_building.intersection(cell)
                    cell_coverage += intersection.area
            
            # Normalize coverage by dividing by the area of the grid cell
            normalized_coverage = cell_coverage / cell.area if cell.area > 0 else 0
            coverage.append(normalized_coverage)
        
        return coverage