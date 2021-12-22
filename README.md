# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Identify credit card customers that are most likely to churn.

Project files structure:
- data: csv-data of customers
- images: plots of feature importances, roc curves, prediction results
- logs: logs of churn_script_logging_and_tests.py
- models: load and save trained models

## Installation Dependencies
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn

## Running Files
How do you run your files? What should happen when you run your files?

churn_library.py: all necessary functions to predict credit card customers that are most likely to churn

python churn_library.py
run all functions to predict credit card customers depending of 'Churn' column

run to:
- import csv data
- perform eda and save images
- train and save models
- plot and save feature importances

churn_script_logging_and_tests.py: test all functions of churn_library.py and save logging

python churn_script_logging_and_tests.py
test all functions of churn_library and write logging file ./logs/churn_library.log
attention overrides ./models



