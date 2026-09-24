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
