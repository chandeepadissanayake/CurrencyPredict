from train.data.sources.investingcom import InvestingComDataSource

# Change this file as needed

# Create an instance with new datasets as a source here.
SOURCES_TO_READ = [
    InvestingComDataSource("D:/Dev/Python/CurrencyPredict/data/GBP_USD_1980_01_02_1999_06_09.csv"),
    InvestingComDataSource("D:/Dev/Python/CurrencyPredict/data/GBP_USD_1999_06_10_2018_08_10.csv"),
]

# The percentage/ratio of dataset used to train the model.
TRAIN_DATA_PERCENTAGE = 0.8

OUTPUT_MODEL_FOLDER_PATH = "D:/Dev/Python/CurrencyPredict/data/models/GBP_USD_1980_01_02_2018_08_10"
