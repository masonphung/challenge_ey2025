# The 2025 EY Open Science AI and Data Challenge: Cooling Urban Heat Islands 
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project Objective
The objective is to create a machine learning model that predicts the Urban Heat Island (UHI) effect, which represents a range of temperatures varying across different locations in urban areas. The model will be developed using data extracted from European Sentinel-2 optical satellite imagery, NASA Landsat optical satellite imagery, and the Building Footprints dataset.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README file for project overview and instructions
├── data
│   ├── interim        <- Intermediate data after initial transformations.
│   ├── processed      <- Final datasets prepared for modelling.
│   ├── raw            <- Original, unprocessed data.
│   └── test           <- Final submission results or test outputs.
│
├── docs               <- Project documentation, references, or important papers
│
├── models             <- Trained models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks documenting the model development process
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── report            <- a document to describe the model development approach
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── data_manipulation.py     <- Important class and functions for extracting satellite band values and calculating vegetation indices. 
    │
    ├── data_processing.py       <- Important class and functions for cleaning, transforming, and preparing data for analysis.
    │
    ├── kml.py             <- Building footprints file
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
├── setup.cfg          <- Configuration file for flake8
├── submission.csv          <- Submission file for assessment
