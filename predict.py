import argparse
import consts.args as consts_args
from model import Model

parser = argparse.ArgumentParser(description="Predict Exchange Rates CLI")
parser.add_argument("-" + consts_args.KEY_PREDICT_MODEL_FOLDER_PATH, required=True, help=consts_args.DESC_PREDICT_MODEL_FOLDER_PATH)
parser.add_argument("-" + consts_args.KEY_PREDICT_DATE, required=True, help=consts_args.DESC_PREDICT_DATE)

args = vars(parser.parse_args())

model = Model(args[consts_args.KEY_PREDICT_MODEL_FOLDER_PATH])
predicted_value = model.predict(args[consts_args.KEY_PREDICT_DATE])

print("Predicted Exchange Rate on %s is %f" % (args[consts_args.KEY_PREDICT_DATE], predicted_value))
