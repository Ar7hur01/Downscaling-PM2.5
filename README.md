# High-Resolution Intra-Urban PM2.5 Downscaling

## Project Overview
Coarse satellite data often fails to capture intra-urban fine particulate matter (PM2.5) gradients, obscuring local health risks and environmental disparities in complex urban topographies. This repository provides a novel, multi-source machine learning workflow designed to downscale PM2.5 concentrations to a highly localized **100 m spatial resolution**. 

The workflow fuses meteorological and atmospheric satellite data, Inverse Distance Weighted (IDW) hourly traffic emissions, and Topographic Position Index (TPI) metrics. These proxies are calibrated against a network of low-cost sensors (Sensor.Community), firmly anchored by official reference station data (Kopfklinik / Würzburg, Germany) during the summer period of May–August 2024.

## Key Features
- **Multi-Source Data Fusion:** Integrates Sentinel-5P, MODIS MAIAC, ERA5, ICON-D2 weather models, localized traffic data, and micro/macro topography (TPI 300m/1500m).
- **Advanced Machine Learning:** Utilizes a baseline-calibrated Random Forest Regressor optimized for geospatial prediction.
- **Robust Spatial Validation:** Employs a 4-fold Anchored Leave-Location-Out (LLO) Spatial Cross-Validation to ensure spatial transferability and prevent overfitting.
- **High-Resolution Mapping:** Generates automated 100x100m PM2.5 prediction maps (Golden Day scenarios, summer means, and extreme value maps).

## Repository Structure
The project follows a strict, chronologically numbered pipeline to ensure reproducibility:

```text
pm25-downscaling-wuerzburg/
├── data/                    # Ignored in version control
│   ├── raw/                 # Raw downloads (Sensors, ICON, Sentinel, Traffic, DEM)
│   ├── processed/           # Cleaned, masked, and merged datasets
│   └── output/              # Final tabular data and generated PM2.5 TIFF maps
├── scripts/                 # Core execution pipeline
│   ├── 01_download/         # Scripts for Earth Engine, API, and manual data retrieval
│   ├── 02_preprocessing/    # Traffic IDW calculations, CRS projections, and cleaning
│   ├── 03_fusion/           # Final feature dataset compilation
│   ├── 04_modeling/         # Random Forest training, LLO CV, and Golden Day mapping
│   └── 05_validation/       # Loss analysis, precipitation validation, and statistics
├── .gitignore               # Excludes large data files and virtual environments
├── pyproject.toml / uv.lock # Dependency management configuration
└── README.md                # Project documentation
