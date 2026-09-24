# MSTNet Project
This repository contains the implementation of MSTNet, a neural network for moisture content prediction based on six-channel near-infrared sensor signals of gas-liquid two-phase flow.
The model accepts input signals with shape (B, L, C) and outputs moisture content with shape (B, 1).

## Dataset Information
The original raw dataset cannot be publicly released.
The CSV files under `synthetic_dataset/` are synthetic test data, which do not represent the real gas-liquid two-phase flow measurements. This synthetic dataset is provided to reproduce the main experimental results of the paper.

## Environment & Dependencies
All required packages are listed in `requirements.txt`.
Create the environment and install dependencies:
```bash
pip install -r requirements.txt

## Model Training
Run the training script:python train.py
The trained model checkpoint will be saved to saved_models/best_MSTNet_model.pth.

## Model Testing and Visualization
Run the test script to evaluate model performance and generate prediction visualization figures:python test.py

## Configuration
Users can modify dataset paths and model hyperparameters in config.py.

## Repository Structure
├── model/├──MSTNet.py                         # Model definition files
          └──__Init__.py
├── saved_models/├──best_MSTNet_model.pth      # Saved model checkpoints
├── synthetic_dataset/                         # Synthetic CSV dataset for result reproduction
├── utils/├──Prediction_analysis.py            # Helper functions and plotting scripts
          ├──dataloder.py
          ├──plots.py
          └──__Init__.py
├── LICENSE                                    # Open-source license
├── README.md                                  # Project readme file
├── config.py                                  # Configuration file
├── requirements.txt                           # Python package dependencies
├── test.py                                    # Test & visualization script
└── train.py                                   # Training script

## License
This project is released under the MIT License. See the LICENSE file for details.
