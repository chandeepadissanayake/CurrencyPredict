import pandas as pd
from train.data.sources.abstracts import DataSource
import consts.data as consts_data


class InvestingComDataSource(DataSource):

    _COL_NAME_DATE = "Date"
    _COL_NAME_CLOSE = "Price"

    def __init__(self, file_path: str):
        self._file_path = file_path

    def read_data(self):
        df = pd.read_csv(self._file_path, header=0, parse_dates=[0]).filter(
            [InvestingComDataSource._COL_NAME_DATE, InvestingComDataSource._COL_NAME_CLOSE],
            axis=1)
        df.columns = consts_data.COLUMNS_TO_SET
        return df
