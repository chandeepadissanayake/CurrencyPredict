import pandas as pd
import train.config.data as config_data


def train_test_split_data(all_data: pd.DataFrame):
    train_dataset_size = int(len(all_data) * config_data.TRAIN_DATA_PERCENTAGE)
    train_data = all_data[0:train_dataset_size]
    test_data = all_data[train_dataset_size:len(all_data)]

    return train_data, test_data


def convert_to_model_data(data: pd.DataFrame):
    np_data = data.values
    return [example for example in np_data]
