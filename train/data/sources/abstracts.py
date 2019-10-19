import abc


class DataSource(abc.ABC):

    # This method should return data in as a Pandas Dataframe with two columns, Date and Close.
    @abc.abstractmethod
    def read_data(self):
        pass
