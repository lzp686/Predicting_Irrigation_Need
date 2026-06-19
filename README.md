# Predicting Irrigation Need

This repository contains our solution for the Kaggle competition [Playground Series S6E4](https://www.kaggle.com/competitions/playground-series-s6e4/overview).

## Overview

The task is a tabular binary classification problem: given agricultural and weather-related features, predict whether irrigation is needed. The competition was evaluated using **balanced accuracy**.

Our approach is based on exploratory data analysis, tree-based models, pseudo-label experiments, and ensemble learning.

## Highlights

- Strong focus on feature analysis and model comparison
- LightGBM and XGBoost as the main base learners
- Pseudo-label related experiments for robustness
- Final prediction through stacking and soft voting

The most informative feature in our analysis is `SoilMoisture`, followed by crop-related and weather-related features such as `CropDays`, `CropType`, `GrowthStage`, `SoilTemp`, `Temp`, `Rainfall_mm`, `Humidity`, and `WindSpeed`.

## Repository Structure

- `train_pseudo.py`: training script for model fitting and pseudo-label experiments
- `irrigation_ensemble.py`: script for combining model outputs into the final prediction
- `softvote.py`: helper script for soft voting experiments
- `formula_check.py`: utility script for checking formulas or implementation details

## Method

### Exploratory Data Analysis

We first inspected the feature distributions and target relationships to understand which variables were most useful for the task. This step showed that:

- `SoilMoisture` is the most informative feature
- `CropType` and `GrowthStage` provide strong categorical separation
- `Rainfall_mm`, `Temp`, `Humidity`, and `WindSpeed` are useful auxiliary signals

### Base Models

We trained several tree-based models, with the main focus on:

- LightGBM
- XGBoost

These models were selected because they work well on structured tabular data and handle nonlinear feature interactions effectively.

### Pseudo-Label Experiments

The script `train_pseudo.py` was used for training and pseudo-label related experiments. The goal was to improve generalization by leveraging additional confident predictions during the training workflow.

### Ensemble

The final prediction pipeline is implemented in `irrigation_ensemble.py`. We combined model outputs using:

- stacking with logistic regression as the meta-learner
- soft voting over multiple base predictions

In practice, stacking was used to fuse the base learner predictions, while soft voting served as an additional ensemble strategy for comparison and robustness.

## Result

Our final submission achieved a public leaderboard score of **0.98044**.

## Usage

### 1. Prepare the data

Download the competition data from Kaggle and place it in the expected input directory used by the scripts.

### 2. Train the base models

```bash
python train_pseudo.py
```

### 3. Run the ensemble step

```bash
python irrigation_ensemble.py
```

### 4. Optional ensemble experiments

```bash
python softvote.py
python formula_check.py
```

## Requirements

Recommended environment:

- Python 3.10+
- numpy
- pandas
- scikit-learn
- lightgbm
- xgboost

Install dependencies with:

```bash
pip install numpy pandas scikit-learn lightgbm xgboost
```

## License

This repository is provided for educational and research reference. If you plan to reuse the code, please check the original competition rules and follow the applicable licensing and attribution requirements.

## Acknowledgements

- Kaggle competition: [Playground Series S6E4](https://www.kaggle.com/competitions/playground-series-s6e4/overview)
- Our public repository: [Predicting_Irrigation_Need](https://github.com/lzp686/Predicting_Irrigation_Need)
- Kaggle community and competition organizers for providing the benchmark and dataset

## Notes

- The exact score may vary slightly depending on random seed and validation setup.
- The code is open-sourced for reference and reproducibility.
