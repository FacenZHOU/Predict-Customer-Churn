'''
Here are the unit tests of churn_library.py

- test data import
- test EDA
- test data encoder
- test feature enginering
- test train model

Author: Facen
Date: 12.03.2023
'''

import pandas as pd
from pathlib import Path
import logging
import churn_library as cl


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

PATH = './data/bank_data.csv'

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']

keep_cols = [
    'Churn',
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
    'Card_Category_Churn']


def test_import_data(path):  # def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = cl.import_data(path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return data_frame


def test_perform_eda(data_frame):
    '''
    test perform eda function
    '''
    cl.perform_eda(data_frame)

    path = Path('./images/eda')

    for image in [
        'Churn_distribution.png',
        'Custom_Age_distribution.png',
        'Marital_Status_distribution.png',
        'Total_Trans_Ct_distribution.png',
        'heatmap_distribution.png'
    ]:
        image_path = path.joinpath(f'{image}')
        try:
            assert image_path.is_file()
        except AssertionError as err:
            logging.error(f'ERROR: {image} not found.')
            raise err
    logging.info('SUCCESS: All EDA results have been successfully saved')


def test_encoder_helper(data_frame, cat_columns, keep_cols):
    '''
    test encoder helper
    '''
    assert isinstance(data_frame, pd.DataFrame)
    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error("ERROR: The dataframe doesn't have the correct frame")
        raise err
    encoded_data = cl.encoder_helper(data_frame, cat_columns)
    try:
        for col in keep_cols:
            assert col in encoded_data.columns
    except AssertionError as err:
        logging.error('ERROR: Missing categorical columns')
        raise err
    logging.info('SUCCESS: Categorical columns correctly encoded.')
    return encoded_data


def test_perform_feature_engineering(data_frame):
    '''
    test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
        data_frame)
    try:
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        logging.info(
            'SUCCESS: Datasets for training and testing have been correctly generated')
    except AssertionError as err:
        logging.error('ERROR: Datasets for training and testing do not match')
        raise err
    return (X_train, X_test, y_train, y_test)


def test_train_models(datasets):
    '''
    test train_models
    '''
    X_train, X_test, y_train, y_test = datasets
    cl.train_models(X_train, X_test, y_train, y_test)
    path = Path('./models')
    rfc_path = path.joinpath('rfc_model.pkl')
    lc_path = path.joinpath('logistic_model.pkl')
    try:
        assert rfc_path.is_file()
        assert lc_path.is_file()
        logging.info('SUCCESS: Models successfully saved')
    except AssertionError as err:
        logging.error('ERROR: Models incorrectly saved')
        raise err


if __name__ == "__main__":
    df = test_import_data(PATH)
    test_perform_eda(df)
    ed = test_encoder_helper(df, cat_columns, keep_cols)
    datasets = test_perform_feature_engineering(ed)
    test_train_models(datasets)
