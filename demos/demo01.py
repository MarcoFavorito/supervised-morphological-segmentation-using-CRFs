import log
import os
import utils
from utils import print_log
import features.extract_features
import sklearn
import sklearn.model_selection as ms
import sklearn_crfsuite
import metrics.my_scorer
import metrics.evaluation
import data_parsers.make_dataset
import details
import pickle
import numpy
import random
import copy

def exec_demo(demo_params):
    """
    Train the crf with different size of the train set
    Tune the hyperparameter over the development set
    Then test the best model
    :param demo_params:
    :return:
    """
    logger = log.setup_logger(__name__)

    #ignore this line... It's a long story.
    feature_type = "ver1"

    # extract base parameters
    demo_id, name, train_file, dev_file, test_file, output_folder = utils.extract_base_demo_params(demo_params)
    print_log(logger, "\n".join([ str((key, demo_params[key])) for key in list(demo_params)]))

    different_sizes_perc = list(range(10, 101,10))

    # define the scoring function for the grid search
    my_scorer = sklearn.metrics.make_scorer(
        metrics.my_scorer.get_evaluation
    )

    # track some result from the search used for tuning the hyperparameter delta
    size_evaluations = {}
    train_data_partitions = {}
    fscores = {}


    # pre-processing the data (remove tags and other stuff)
    print_log(logger, "Making datasets...")
    train_data = data_parsers.make_dataset.parse_file(open(train_file))
    dev_data = data_parsers.make_dataset.parse_file(open(dev_file))
    test_data = data_parsers.make_dataset.parse_file(open(test_file))

    # compute the maximum delta possible (from the length of the longest word
    # in the train and development set)
    max_delta = max(utils.find_max_len(train_data), utils.find_max_len(dev_data))
    print_log(logger, "max delta: %s, len train set:%s" %(max_delta, len(train_data)))

    # train the model for different train sizes
    for size in different_sizes_perc:
        print_log(logger, "train the model with percentage of the train set: %02d%%" % size)

        train_data_shuffled = copy.deepcopy(train_data)
        random.shuffle(train_data_shuffled)
        current_size = round(len(train_data)*size/100)
        print_log(logger, "current train set size: %d" % current_size)
        train_data_partition = train_data_shuffled[:current_size]
        print_log(logger, "train set: " + "; ".join(list(map(str,train_data_partition[0:5]))) + "...")

        size_evaluations[size] = {}
        train_data_partitions[size] = train_data_partition

        current_max_delta = utils.find_max_len(train_data_partition)
        print_log(logger, "current max delta: %s" % current_max_delta)

        for delta in range(1,current_max_delta+1):
            print_log(logger, "train the model with delta: %d" % delta)

            X_train, y_train = features.extract_features.get_features_and_labels(train_data_partition, delta, feature_type)
            X_dev, y_dev = features.extract_features.get_features_and_labels(dev_data, delta, feature_type)
            X_test, y_test = features.extract_features.get_features_and_labels(test_data, delta, feature_type)

            crf = sklearn_crfsuite.CRF(
                algorithm='ap',
                all_possible_transitions=True,
                all_possible_states=False,
            )
            crf.fit(X_train, y_train)
            y_dev_pred = crf.predict(X_dev)
            delta_evaluation = metrics.evaluation.get_evaluation(feature_type, y_dev, y_dev_pred)

            print_log(logger, "F-score on development set: %s" % delta_evaluation["F-score"])
            size_evaluations[size][delta] = (delta_evaluation["Precision"], delta_evaluation["Recall"], delta_evaluation["F-score"])

    # find delta that yields best F-score
    sizes = list(size_evaluations.keys())
    sizes.sort()
    deltas = []
    for size in sizes:
        max_fscore = max(size_evaluations[size].values())
        max_delta_for_size = [i for i in size_evaluations[size]
                              if size_evaluations[size][i] == max_fscore][0]
        deltas.append(max_delta_for_size)
        print_log(logger, "\nBest delta=%s for train size perc=%s%%. "
                          "\nOn development set:"
                          "\n\tPrecision=%s"
                          "\n\tRecall=%s"
                          "\n\tF-score=%s"
                  % (max_delta_for_size, size,
                     size_evaluations[size][max_delta_for_size][0],
                     size_evaluations[size][max_delta_for_size][1],
                     size_evaluations[size][max_delta_for_size][2])
                  )


    test_evaluations = {}
    print_log(logger, "Test models with different sizes of training set")
    for size, best_delta in zip(sizes, deltas):
        print_log(logger, "Train with size: %d and delta: %s" % (size, best_delta))
        cur_train_set = train_data_partitions[size]
        print_log(logger, "train set: " + "; ".join(list(map(str,cur_train_set[0:5]))) + "...")
        X_train, y_train = features.extract_features.get_features_and_labels(cur_train_set, best_delta, feature_type)
        X_test, y_test = features.extract_features.get_features_and_labels(test_data, best_delta, feature_type)

        crf = sklearn_crfsuite.CRF(
            algorithm='ap',
            all_possible_transitions=True,
            all_possible_states=False,
        )
        crf.fit(X_train, y_train)

        y_test_pred = crf.predict(X_test)
        delta_evaluation = metrics.evaluation.get_evaluation(feature_type, y_test, y_test_pred)

        test_evaluations[size] = (delta_evaluation["Precision"], delta_evaluation["Recall"], delta_evaluation["F-score"])
        print_log(logger, "train score (delta=%s): F-score, : %s" % (best_delta, delta_evaluation["F-score"]))

        # save some result from the tests
        curpath = output_folder+"/size_%02d_delta_%02d"%(size,best_delta)
        os.makedirs(curpath)

        curpath = curpath\
                  +"/"+name+"_" \
                  + "size_%02d_delta_%02d"%(size,best_delta)
        utils.write_model(crf, open(curpath + ".model", "wb+"))
        utils.write_predictions(feature_type, open(test_file), y_test_pred, open(curpath + ".pred", "w+"))
        utils.write_evaluation(delta_evaluation, open(curpath + ".eval","w+"))
        utils.write_fails(open(test_file), y_test, y_test_pred, open(curpath+".fails","w+"), feature_type)
        details.print_details(crf, file=open(curpath + ".details","w+"))

    freport = open(output_folder + "/report.txt", "w+")
    for size, best_delta in zip(sizes, deltas):
        print("Best delta=%s for train size perc=%s%%. "
              "\nOn development set:"
              "\n\tPrecision=%s"
              "\n\tRecall=%s"
              "\n\tF-score=%s"
              "\nOn test set:"
              "\n\tPrecision=%s"
              "\n\tRecall=%s"
              "\n\tF-score=%s"
              % (best_delta, size,
                 size_evaluations[size][best_delta][0],
                 size_evaluations[size][best_delta][1],
                 size_evaluations[size][best_delta][2],
                 test_evaluations[size][0],
                 test_evaluations[size][1],
                 test_evaluations[size][2])
              + "\n"+"-"*50,
              file=freport)






