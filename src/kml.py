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
    def __init__(self, kml_path):
        self.kml_file = kml_path
        self.df = None
        
    def load_kml(self):
        
        
    def parse_kml(file_path):
        """
        Parse a KML file and extract building footprints as polygons.
        Handles both two and three component coordinates.
        """
        kml = minidom.parse(file_path)
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