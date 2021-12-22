"""
Test functions of churn_library.py and log results

author: bjoern
date: 22.12.21
"""

import os
import logging
import pandas as pd
import numpy as np

from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        df = import_data("./data/bank_data.csv")
        # delete existing eda images
        if os.path.exists('./images/eda/normalized_count_marital_status.svg'):
            os.remove('./images/eda/normalized_count_marital_status.svg')
        if os.path.exists('./images/eda/dist_plot_total_trans_ct.svg'):
            os.remove('./images/eda/dist_plot_total_trans_ct.svg')

        perform_eda(df)
        # check if Churn column created
        assert 'Churn' in df.columns
        logging.info("Testing creation of Churn category: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing creation of Churn category: The dataframe doesn't appear to have Churn column")
        raise err

    try:
        assert os.path.exists(
            './images/eda/normalized_count_marital_status.svg')
        assert os.path.exists('./images/eda/dist_plot_total_trans_ct.svg')
        logging.info("Saving of eda images: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing creation of eda images: The eda images are note available")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        test_data = {'Churn': [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2], 'Gender': [
            'm', 'm', 'm', 'm', 'm', 'm', 'w', 'w', 'w', 'w', 'w', 'w']}
        test_df = pd.DataFrame(test_data)
        category_lst = ['Gender']
        return_df = encoder_helper(test_df, category_lst)
        # check if Churn column created with correct mean values
        assert return_df['Gender_Churn'][0] == 0.5
        assert return_df[return_df['Gender'] ==
                         'w']['Gender_Churn'].iloc[0] == 2.0
        logging.info(
            "Testing of encoding of new columns with proportion of churn: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing of encoding with new columns and proportion of churn failed as new columns are not available or wrong mean values")
        raise err

    # test if response value is correct
    try:
        return_df = encoder_helper(test_df, category_lst, 'Test')
        assert 'Gender_Test' in return_df.columns
        logging.info(
            "Testing of response value of encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing of response value of encoder_helper incorrect")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        # generate dummy data
        generate_cols = [
            'Customer_Age',
            'Dependent_count',
            'Months_on_book',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio',
            'Gender_Churn',
            'Education_Level_Churn',
            'Marital_Status_Churn',
            'Income_Category_Churn',
            'Card_Category_Churn',
            'Churn']
        generate_data = []
        for _ in range(10):
            generate_data.append(list(range(20)))

        test_df = pd.DataFrame(generate_data, columns=generate_cols)

        X_train, X_test, y_train, y_test = perform_feature_engineering(test_df)

        # check sizes (with 0.3 test size)
        assert X_train.shape[0] == 7
        assert X_train.shape[1] == 19
        assert X_test.shape[0] == 3
        assert X_test.shape[1] == 19
        assert y_train.shape == (7,)
        assert y_test.shape == (3,)

        logging.info(
            "Testing of feature engineering df sizes: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Invalid df feature engineering sizes with expected test_size = 0.3\
                \nShapes: X_train %s, X_test %s, y_train %s, y_test %s",
            X_train.shape,
            X_test.shape,
            y_train.shape,
            y_test.shape)
        raise err

    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            test_df, 'Months_on_book')
        assert y_train.name == 'Months_on_book'
        assert 'Months_on_book' not in X_train.columns
    except AssertionError as err:
        logging.error(
            "Testing different response index y column failed, y_train.name == %s",
            y_train.name)
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        # delete existing eda images
        if os.path.exists('./images/results/rvc_classification_report.svg'):
            os.remove('./images/results/rvc_classification_report.svg')
        if os.path.exists('./images/results/lr_classification_report.svg'):
            os.remove('./images/results/lr_classification_report.svg')
        if os.path.exists('./images/results/rfc_lrc_roc_curves.svg'):
            os.remove('./images/results/rfc_lrc_roc_curves.svg')
        if os.path.exists('./images/results/feature_importance.svg'):
            os.remove('./images/results/feature_importance.svg')

        # delete models
        if os.path.exists('./models/rfc_model.pkl'):
            os.remove('./models/rfc_model.pkl')
        if os.path.exists('./models/logistic_model.pkl'):
            os.remove('./models/logistic_model.pkl')

        # generate dummy data
        np.random.seed(42)
        columns_list = list('ABCDEFGHIJKLMN')
        columns_list.extend(['Churn'])
        models_test_df = pd.DataFrame(
            np.random.randint(
                0,
                100,
                size=(
                    200,
                    15)),
            columns=columns_list)
        models_test_df['Churn'] = pd.Series(np.random.randint(2, size=200))
        x_models_df = pd.DataFrame()
        x_cols = list('ABCDEFGHIJKLMN')
        x_models_df[x_cols] = models_test_df[x_cols]
        y_models_df = models_test_df['Churn']

        train_models(x_models_df[:140],
                     x_models_df[141:],
                     y_models_df[:140],
                     y_models_df[141:])
        # check if classification report created
        assert os.path.exists('./images/results/rvc_classification_report.svg')
        assert os.path.exists('./images/results/lr_classification_report.svg')
        assert os.path.exists('./images/results/rfc_lrc_roc_curves.svg')
        assert os.path.exists('./images/results/feature_importance.svg')
        # check if models trained and saved
        assert os.path.exists('./models/rfc_model.pkl')
        assert os.path.exists('./models/logistic_model.pkl')
        logging.info(
            "Testing of model training, classification report, saving models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Generation of classification report or training and saving of the models failed")
        raise err


if __name__ == "__main__":
    test_import(import_data)
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(perform_feature_engineering)
    test_train_models(train_models)
