from optparse import OptionParser,OptionGroup
import settings
import metrics.evaluation
import log
import pickle
import features.ver1
import features.ver2

import sklearn
import sklearn_crfsuite
import sklearn.model_selection as ms

logger = log.setup_logger(__name__)


def build_parser():
    # parse options
    parser = OptionParser("Usage: %prog --mode (" + "|".join(settings.MODES) + ") <arg1> <arg2>... "
                                                                               "\nUse -h for more details")

    parser.add_option('-m', '--mode', type='choice', action='store', dest='mode',
                      choices=settings.MODES, help="Select a mode between: '" + "', '".join(settings.MODES) + "'")

    train_opts = OptionGroup(
        parser, 'Train Options',
        'These options control the train mode.',
    )

    test_opts = OptionGroup(
        parser, 'Test Options',
        'These options control the test mode.',
    )

    demo_opts = OptionGroup(
        parser, 'Demo Options',
        "You can decide which demo you want execute among: '" + "', '".join(settings.DEMOS) + "'"
    )

    train_opts.add_option("--train_file", dest="train_file", default=None)
    train_opts.add_option("--dev_file", dest="dev_file", default=None)
    train_opts.add_option("--output_file", dest="output_file", default=None)
    train_opts.add_option("--delta", action="store", type="int", dest="delta", default=5,
                          help="Specify delta [default: 5]")
    train_opts.add_option("--feature_type", type="choice", action="store", dest="feature_type",
                          choices=settings.FEATURE_TYPES, default="ver1",
                          help="Specify the feature extraction strategy among: '" + "', '".join(settings.FEATURE_TYPES)
                               + ". See the docs for further details.")

    test_opts.add_option("--model_file", dest="model_file", default=None)
    test_opts.add_option("--test_file", dest="test_file", default=None)
    test_opts.add_option("--pred_file", dest="pred_file", default=None)
    train_opts.add_option("--delta_model", action="store", type="int", dest="delta_model", default=5,
                          help="Specify delta of the trained model. [default: 5]"
                               "N.B.: this value must be the same of the delta used for training the model!!!")


    demo_opts.add_option("--version", dest="version")

    parser.add_option_group(train_opts)
    parser.add_option_group(test_opts)
    parser.add_option_group(demo_opts)

    parser.add_option("--verbose", action="store_true", dest="verbose", default=False,
                      help="print a lot of stuff")

    return parser


def validate_options(parser, options):
    error = False
    if options.mode not in settings.MODES:
        parser.error("You have to decide between these values: " + "|".join(settings.MODES))
        error = True
    elif options.mode=="train" and not options.train_file:
        parser.error("If you want to train, there must be a train file... o.O?!")
        error = True
    elif options.mode=="test":
        if not options.model_file:
            parser.error("Please specify the --model_file")
        elif not options.test_file:
            parser.error("Please specify the --test_file")
        elif not options.pred_file:
            parser.error("Please specify the --pred_file")
    else:
        pass
    if error:
        raise ValueError

def print_log(thislogger, str):
    if settings.VERBOSE:
        thislogger.debug(str)



def labels2segments(feature_type, word, labels):
    segments = []
    if feature_type in ["ver1","ver3","ver4"]:
        segments = features.ver1.labels2segments(word, labels)
    elif feature_type in ["ver2"]:
        segments = features.ver2.labels2segments(word, labels)
    return segments

def write_predictions(feature_type, test_file, y_pred, out_file):
    print_log(logger,"Writing predictions in " + out_file.name)
    for row,preds in zip(test_file,y_pred):
        word = get_word_from_row(row)
        print(word + " " + " ".join(labels2segments(feature_type, word, preds)), file=out_file)


def write_evaluation(y_test, y_pred, metrics_file):
    print_log(logger,"Writing evaluation in " + metrics_file.name)

    flat_y_test = [i for subl in y_test for i in subl]
    flat_y_pred = [i for subl in y_pred for i in subl]

    print("E-count in test:",flat_y_test.count('E-SEG'),file=metrics_file)
    print("S-count in test:",flat_y_test.count('S-SEG'),file=metrics_file)
    print("E-count in pred:", flat_y_pred.count('E-SEG'),file=metrics_file)
    print("S-count in pred:", flat_y_pred.count('S-SEG'),file=metrics_file)

    H,I,D = metrics.evaluation.compute_HID(flat_y_test, flat_y_pred)
    print("H-score:",H,file=metrics_file)
    print("I-score:",I,file=metrics_file)
    print("D-score:",D,file=metrics_file)
    Precision, Recall, FScore = metrics.evaluation.compute_PRF(flat_y_test, flat_y_pred)
    print("Precision:",Precision, file=metrics_file)
    print("Recall:", Recall, file=metrics_file)
    print("F-score:", FScore, file=metrics_file)

def write_evaluation(evaluation, metrics_file):
    print_log(logger,"Writing evaluation in " + metrics_file.name)

    print("E-count test:", str(evaluation["E-count_test"]), file=metrics_file)
    print("S-count test:", str(evaluation["S-count_test"]), file=metrics_file)
    print("E-count pred:", str(evaluation["E-count_pred"]), file=metrics_file)
    print("S-count pred:", str(evaluation["S-count_pred"]), file=metrics_file)
    print("H:", str(evaluation["H"]), file=metrics_file)
    print("I:", str(evaluation["I"]), file=metrics_file)
    print("D:", str(evaluation["D"]), file=metrics_file)
    print("Precision:", str(evaluation["Precision"]), file=metrics_file)
    print("Recall:", str(evaluation["Recall"]), file=metrics_file)
    print("F-score:", str(evaluation["F-score"]), file=metrics_file)

def write_model(model, model_file):
    print_log(logger, "Writing the model in " + model_file.name)
    pickle.dump(model, model_file)

def write_fails(test_file, y_test, y_pred, fails_file, feature_type):
    n_fails = 0
    for i in range(len(y_test)):
        row =test_file.readline()
        word = get_word_from_row(row)

        y_test_inst = y_test[i]
        y_pred_inst = y_pred[i]
        if (y_test_inst != y_pred_inst):

            print(i, file=fails_file)
            print(word, file=fails_file)
            print(labels2segments(feature_type, word, y_test_inst), file=fails_file)
            print(labels2segments(feature_type, word, y_pred_inst), file=fails_file)
            # print(y_test_inst, file=fails_file)
            # print(y_pred_inst, file=fails_file)

            # print("marginals:",file=fails_file)
            # for j in range(len(y_test[i])):
            #     print(y_test[i][j],y_pred[i][j],file=fails_file)
            #     for item in crf.predict_marginals_single(X_test[i])[j].items(): print("\t",item, file=fails_file)
            n_fails += 1
    print(n_fails, file=fails_file)


def get_word_from_row(row):
    word = row.split(',')[0].split()[0]
    return word

def find_max_len(data):
    i = max(data,key=lambda x: len(x[0]))
    return len(i[0])


def extract_base_demo_params(demo_params):
    demo_id = demo_params["id"]
    name = demo_params["name"]
    train_file = demo_params["train_file"]
    dev_file = demo_params["dev_file"]
    test_file = demo_params["test_file"]
    output_folder = demo_params["output_folder"]
    return demo_id, name, train_file, dev_file, test_file, output_folder


def run_grid_search(X_train, y_train, X_dev, y_dev, param_grid, my_scorer):
    crf = sklearn_crfsuite.CRF(
        algorithm='ap',
        all_possible_transitions=True,
        all_possible_states=True,
    )

    validation_set_indexes = [-1] * len(X_train) + [0] * len(X_dev)
    ps = ms.PredefinedSplit(test_fold=validation_set_indexes)

    search = ms.GridSearchCV(
        estimator=crf,
        cv=ps,
        param_grid=param_grid,
        scoring=my_scorer,
        verbose=1,
        n_jobs=-1,
    )

    search.fit(X_train + X_dev, y_train + y_dev)
    return search
