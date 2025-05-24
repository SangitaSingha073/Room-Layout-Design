# Room Layout Design Model

This project generates and predicts room layouts for a given plot size and number of rooms using a machine learning model (Random Forest Regressor). It creates synthetic room layout data, trains a model, and generates visualizations and JSON files for predicted layouts. The project includes scripts for data generation, preprocessing, model training, layout prediction, and visualization.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Main Script](#running-the-main-script)
  - [Running the Predictor Script](#running-the-predictor-script)
- [Output Files](#output-files)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The project consists of several Python scripts:
- `data_generator.py`: Generates synthetic room layout data and saves it as JSON files.
- `preprocessor.py`: Preprocesses data by scaling numerical features and encoding room types.
- `model_trainer.py`: Trains Random Forest Regressor models for room coordinates, width, and height.
- `layout_generator.py`: Generates and visualizes room layouts with overlap correction.
- `predictor.py`: Allows users to input custom plot dimensions and room types to predict layouts.
- `main.py`: Orchestrates the entire process (data generation, preprocessing, training, and sample layout generation).

The model predicts room positions (`x`, `y`), sizes (`width`, `height`), and ensures no overlaps in the layout. Outputs include JSON files and PNG visualizations.

## Requirements
- Python 3.8+
- Required Python packages:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `joblib`
- Install dependencies using:
  ```bash
  pip install pandas numpy scikit-learn matplotlib joblib
