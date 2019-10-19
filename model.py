import os
from consts import model as consts_model
import json
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
import pandas as pd
import train.preprocessor as train_pre_processor
import train.config.model as config_model
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from datetime import datetime
import utils.dates as utils_dates
import numpy as np


class Model:

    def __init__(self, model_folder_abs_path: str):
        self._model_folder_abs_path: str = model_folder_abs_path

        if not os.path.exists(self._model_folder_abs_path):
            os.makedirs(self._model_folder_abs_path)

        model_file_abs_path = self._get_model_abs_file_path()
        model_meta_data_file_abs_path = self._get_model_meta_data_abs_file_path()
        if os.path.exists(model_file_abs_path) and os.path.exists(model_meta_data_file_abs_path):
            self._arima_model: ARIMAResults = ARIMAResults.load(model_file_abs_path)
            with open(model_meta_data_file_abs_path) as file_meta_data:
                self._meta_data = json.load(file_meta_data)
        else:
            self._arima_model: ARIMAResults = None
            self._meta_data = None

    def _get_model_abs_file_path(self):
        return self._model_folder_abs_path + "/" + consts_model.MODEL_FILE_NAME

    def _get_model_meta_data_abs_file_path(self):
        return self._model_folder_abs_path + "/" + consts_model.META_DATA_FILE_NAME

    def train(self, train_data: pd.DataFrame):
        if self._arima_model is None:
            list_train_data = train_pre_processor.convert_to_model_data(train_data)

            self._arima_model = ARIMA(list_train_data, order=config_model.ORDER_FOR_MODEL)
            self._arima_model = self._arima_model.fit(disp=0)

            self._meta_data = {
                consts_model.META_DATA_KEY_DATA_END_DATE: train_data.index.max().strftime(consts_model.DEFAULT_DATETIME_FORMAT),
                consts_model.META_DATA_KEY_DATA_TRAIN_DATA_SET: [label[0] for label in list_train_data]
            }
        else:
            raise Exception("This directory contains a trained model.")

    def save(self):
        self._arima_model.save(self._get_model_abs_file_path())
        with open(self._get_model_meta_data_abs_file_path(), "w") as file_meta_data:
            json.dump(self._meta_data, file_meta_data)

    def test(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        list_train_data = train_pre_processor.convert_to_model_data(train_data)
        model_for_test: ARIMAResults = self._arima_model
        np_test_data = test_data.values

        test_results = []
        for test_index in range(len(np_test_data)):
            test_date = test_data.index[test_index]
            predicted_value = model_for_test.forecast()[0]
            expected_value = np_test_data[test_index][0]
            test_results.append({
                consts_model.TEST_COLUMN_KEY_DATE: test_date,
                consts_model.TEST_COLUMN_KEY_PREDICTED: predicted_value,
                consts_model.TEST_COLUMN_KEY_EXPECTED: expected_value
            })

            list_train_data.append(expected_value)
            model_for_test = ARIMA(list_train_data, order=config_model.ORDER_FOR_MODEL)
            model_for_test = model_for_test.fit(disp=0)

        return test_results

    def get_mean_squared_error(self, test_data: pd.DataFrame, test_results: list):
        np_test_data = test_data.values
        predictions = [test_result[consts_model.TEST_COLUMN_KEY_PREDICTED] for test_result in test_results]

        return mean_squared_error(np_test_data, predictions)

    def plot_results(self, test_data: pd.DataFrame, test_results: list, color="red"):
        np_test_data = test_data.values
        predictions = [test_result[consts_model.TEST_COLUMN_KEY_PREDICTED] for test_result in test_results]

        pyplot.plot(np_test_data)
        pyplot.plot(predictions, color=color)
        pyplot.show()

    def predict(self, on_date_str: str):
        if self._arima_model is None:
            raise Exception("This model has not been fitted.")
        else:
            on_date = datetime.strptime(on_date_str, consts_model.DEFAULT_DATETIME_FORMAT)
            model_train_end_date = datetime.strptime(self._meta_data[consts_model.META_DATA_KEY_DATA_END_DATE], consts_model.DEFAULT_DATETIME_FORMAT)
            if utils_dates.is_first_date_bigger(on_date, model_train_end_date):
                difference_dates = utils_dates.diff_days(on_date, model_train_end_date)
                model_for_predict = self._arima_model
                model_train_data: list = [np.array([label]) for label in self._meta_data[consts_model.META_DATA_KEY_DATA_TRAIN_DATA_SET]]
                last_prediction = None

                for _ in range(difference_dates):
                    last_prediction = model_for_predict.forecast()[0]

                    model_train_data.append(last_prediction)
                    model_for_predict = ARIMA(model_train_data, order=config_model.ORDER_FOR_MODEL)
                    model_for_predict = model_for_predict.fit(disp=0)

                return last_prediction[0]
            else:
                raise Exception("Please provide a date in the future. This model is not used to predict for the data in which it was trained to.")
