import train.data.reader as data_reader
import train.preprocessor as pre_processor
from model import Model
import train.config.data as config_data
import consts.model as consts_model

print("Reading data...")
_read_data = data_reader.read_all_data()
train_data, test_data = pre_processor.train_test_split_data(_read_data)

print("Training the model...")
model = Model(config_data.OUTPUT_MODEL_FOLDER_PATH)
model.train(train_data)
print("Completed training the model...")

print("Saving the Model...")
model.save()
print("Model saved...")

print("Testing for Accuracy...")
test_results = model.test(train_data, test_data)
for test_result in test_results:
    str_date = test_result[consts_model.TEST_COLUMN_KEY_DATE].strftime(consts_model.DEFAULT_DATETIME_FORMAT)
    predicted = test_result[consts_model.TEST_COLUMN_KEY_PREDICTED]
    expected = test_result[consts_model.TEST_COLUMN_KEY_EXPECTED]
    print("Date = %s, Expected Exchange Rate = %f, Predicted Exchange Rate = %f" % (str_date, expected, predicted))


ms_error = model.get_mean_squared_error(test_data, test_results)
print("Test Mean Squared Error (smaller the better fit): %.5f" % ms_error)

print("Plotting the Results...")
model.plot_results(test_data, test_results)
