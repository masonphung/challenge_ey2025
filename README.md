# The 2025 EY Open Science AI & Data Challenge
<a target="_blank" href="https://challenge.ey.com/2025">
    <img src="docs/marketing-2025-img.jpg" />
</a>

> This project was developed as part of the EY Open Science AI & Data Challenge 2025, where global teams worked to design machine learning models that predict the Urban Heat Island (UHI) effect across New York City.

--------
## **Cooling Urban Heat Islands** -  Team Vietnamese

**Mason Phung**, Macquarie University, Master of Data Science

**Claire Dinh**, Macquarie University, Master of Business Analytics

**Wendy Nguyen**, Macquarie University, Master of Business Analytics

### Challenge Achivements
📍 Ranked 67th globally out of 10,000+ participants

🇦🇺 Top 7 in Australia

--------
## Project Overview

Urban Heat Islands (UHIs) represent a critical climate issue in urban centers. Our solution aims to predict UHI intensity across NYC using multi-source data including:

- Ground-traverse temperature data

- Sentinel-2 and Landsat-8 satellite imagery

- Building footprint data

- Local weather station measurements

Through careful data preprocessing, feature engineering, and model tuning, we developed an end-to-end regression pipeline capable of predicting UHI index at meter-scale precision.

--------
## Workflow Summary

I. Data Acquisition & Processing
- Extract and merge Sentinel-2, Landsat LST, KML-based building data, and weather data
- Compute grid-based building coverage and building count
- Apply focal buffer technique to reflect real-world surrounding effects

II. Feature Engineering
- Create vegetation indices (e.g., NDVI), normalize timestamps, derive satellite-derived features
- Integrate spatial features from KML overlays and building polygons

III. Model Development
- Tune and train XGBoost regressors using SHAP-driven feature selection
- Evaluate using R², MAE, and RMSE on held-out data

IV. Final Prediction
- Predict UHI index on test set with top 50 most important features
- Generate final CSV for leaderboard submission

--------
## Directory

```
├── README.md          <- Project README
├── data
│   ├── interim        <- Intermediate data after initial transformations.
│   ├── processed      <- Final datasets prepared for modelling.
│   ├── raw            <- Original, unprocessed data.
│   └── test           <- Final submission results or test outputs.
│
├── docs               <- Project documents, references and challenge questions.
│
├── models             <- Trained models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks contain project process and KML processing guide.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── data_manipulation.py     <- Important class and functions for extracting satellite band values and calculating vegetation indices. 
    │
    ├── data_processing.py      <- Script to collect, process and transform Sentinel-2 and Landsat-8 data.
    │
    ├── kml_processor.py        <- Code to process KML files and extract relevant features for the model.
    │
    ├── modeling.py             <- Code to train, tune and evaluate the model.
    │
    ├── utils.py                <- Utility functions to support the project.
```

--------

## Results
| Dataset              | R² Score | MAE     | RMSE    |
|----------------------|----------|---------|---------|
| Local Train Set      | 0.9990   | 0.0004  | 0.0000  |
| Local Test Set       | 0.9610   | 0.0023  | 0.0000  |
| Competition Test Set | 0.9680   | —       | —       |

## Key Takeaways

> *“The success of this project came not from exotic modeling techniques, but from rigorous data preprocessing and feature engineering. Proper handling of spatial features and grid-cell overlays using building data proved crucial to improving model performance.”*
— Team Vietnamese

## Discussion
Feel free to reach out to Mason through [LinkedIn](https://www.linkedin.com/in/masonphung/) or [GitHub](https://www.github.com/masonphung).