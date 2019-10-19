import argparse
import consts.args as consts_args
from model import Model

parser = argparse.ArgumentParser(description="Predict Exchange Rates CLI")
parser.add_argument("-" + consts_args.KEY_PREDICT_MODEL_FOLDER_PATH, required=True, help=consts_args.DESC_PREDICT_MODEL_FOLDER_PATH)
parser.add_argument("-" + consts_args.KEY_PREDICT_DATE, help=consts_args.DESC_PREDICT_DATE)
parser.add_argument("-" + consts_args.KEY_PREDICT_DATE_FILE, help=consts_args.DESC_PREDICT_DATE_FILE)

args = vars(parser.parse_args())

model = Model(args[consts_args.KEY_PREDICT_MODEL_FOLDER_PATH])
if args[consts_args.KEY_PREDICT_DATE] is not None:
    predicted_value = model.predict(args[consts_args.KEY_PREDICT_DATE])
    print("Predicted Exchange Rate on %s is %f" % (args[consts_args.KEY_PREDICT_DATE], predicted_value))
elif args[consts_args.KEY_PREDICT_DATE_FILE] is not None:
    input_file = open(args[consts_args.KEY_PREDICT_DATE_FILE], "r")
    for date in input_file:
        date = date.strip()
        predicted_value_for_date = model.predict(date)
        print("Predicted Exchange Rate on %s is %f" % (date, predicted_value_for_date))

    input_file.close()
