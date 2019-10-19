import pandas as pd
import consts.data as consts_data
import train.config.data as config_data


def read_all_data():
    data_sources = pd.DataFrame(columns=consts_data.COLUMNS_TO_SET)
    for data_source in config_data.SOURCES_TO_READ:
        data_sources = data_sources.append(data_source.read_data(), ignore_index=True)

    # Do some required processing to prepare the dataframe to be fed into ARIMA toolkit
    data_sources.set_index(consts_data.DATA_COLUMN_NAME_DATE, inplace=True)
    data_sources.sort_index(inplace=True)

    return data_sources
