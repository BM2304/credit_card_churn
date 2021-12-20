import os
import logging
from churn_library import *
#import churn_library_solution as cls

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
        test_data = {'Churn': [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                     'Gender': ['m', 'm', 'm', 'm', 'm', 'm', 'w', 'w', 'w', 'w', 'w', 'w']
                     }
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

    # test if response is correct


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    test_import(import_data)
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
