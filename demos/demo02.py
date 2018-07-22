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

def exec_demo(demo_params):
    """
    Execute grid search over the param_grid defined in demo_params
    :param demo_params:
    :return:
    """
    logger = log.setup_logger(__name__)

    #ignore this line... It's a long story.
    feature_type = "ver1"

    # extract base parameters
    demo_id, name, train_file, dev_file, test_file, output_folder = utils.extract_base_demo_params(demo_params)
    print_log(logger, "\n".join([ str((key, demo_params[key])) for key in list(demo_params)]))

    param_grid = demo_params["param_grid"]

    # define the scoring function for the grid search
    my_scorer = sklearn.metrics.make_scorer(
        metrics.my_scorer.get_evaluation
    )

    # track some result from the grid search used for tuning the hyperparameter delta
    fscores = {}
    epsilons_list = {}
    max_iterations_list = {}
    best_eval = {"F-score": 0}


    # pre-processing the data (remove tags and other stuff)
    print_log(logger, "Making datasets...")
    train_data = data_parsers.make_dataset.parse_file(open(train_file))
    dev_data = data_parsers.make_dataset.parse_file(open(dev_file))
    test_data = data_parsers.make_dataset.parse_file(open(test_file))

    # compute the maximum delta possible (from the length of the longest word
    # in the train and development set)
    max_delta = max(utils.find_max_len(train_data), utils.find_max_len(dev_data))
    print_log(logger, "max delta: %s" % max_delta)

    # repeat the grid search for each possible value of delta
    for delta in range(1, max_delta+1):
        os.makedirs(output_folder + "/%02d" % delta, exist_ok=True)

        print_log(logger, "Training with delta=%s" % delta)
        X_train, y_train = features.extract_features.get_features_and_labels(train_data, delta, feature_type)
        X_dev, y_dev = features.extract_features.get_features_and_labels(dev_data, delta, feature_type)
        X_test, y_test = features.extract_features.get_features_and_labels(test_data, delta, feature_type)

        model = utils.run_grid_search(X_train, y_train, X_dev, y_dev, param_grid, my_scorer)

        best_cv_epsilon = model.best_params_["epsilon"]
        best_cv_max_iterations = model.best_params_["max_iterations"]

        # the best score will be considered in order to pick the best model
        fscores[delta] = model.best_score_
        epsilons_list[delta] = best_cv_epsilon
        max_iterations_list[delta] = best_cv_max_iterations

        print_log(logger, "Best params for delta %02d: max_iterations=%d\tepsilon=%.2E"
                  % (delta, best_cv_max_iterations, best_cv_epsilon))
        print_log(logger, "Best CV score: " + str(model.best_score_))



        # test the model on the test set. NOTICE: the result will not be considered for the choice
        # of the hyperparameter delta!
        print_log(logger, "***Predict test with the grid search model:***")

        y_test_pred = model.predict(X_test)
        test_eval = metrics.evaluation.get_evaluation(feature_type, y_test, y_test_pred)
        print_log(logger, "F-score on test (grid search with delta=%s): %s"
                  % (delta, test_eval["F-score"])
                  )


        # save some result from the grid search
        curpath = output_folder+"/%02d"%delta+"/"+name+"_" + "%02d" % delta
        utils.write_model(model, open(curpath + "_gridsearch.model", "wb+"))
        utils.write_predictions(feature_type, open(test_file), y_test_pred, open(curpath + ".pred", "w+"))
        utils.write_evaluation(test_eval, open(curpath + ".eval","w+"))
        utils.write_fails(open(test_file), y_test, y_test_pred, open(curpath+".fails","w+"), feature_type)
        details.print_gridsearch_details(model, file=open(curpath + "_gridsearch.details","w+"))

        print_log(logger, "#" * 50)

    print_log(logger, "-" * 50)
    max_fscore = max(fscores.values())
    max_fscore_delta = [i for i in fscores.keys() if fscores[i] == max_fscore][0]
    best_model_num = max_fscore_delta
    best_epsilon = epsilons_list[best_model_num]
    best_max_iterations = max_iterations_list[best_model_num]

    freport = open(output_folder + "/report.txt", "w+")
    print_log(logger, "The best model found is the one with delta: %s" % best_model_num)
    print_log(logger, "With best parameters: max_iterations=%s, epsilon=%s"
              % (best_max_iterations, best_epsilon))
    print_log(logger, "CV F-score: %s" % max_fscore)
    print("The best model found is the one with delta: %s" % best_model_num, file=freport)
    print("With best parameters: max_iterations=%s, epsilon=%s"
              % (best_max_iterations, best_epsilon), file=freport)
    print("CV F-score: %s" % max_fscore, file=freport)

    best_model_path = output_folder+"/%02d"%best_model_num+"/"+name+"_" + "%02d" % best_model_num + "_gridsearch.model"
    best_model = pickle.load(open(best_model_path, "rb"))
    X_test, y_test = features.extract_features.get_features_and_labels(test_data, best_model_num, feature_type)

    y_pred = best_model.predict(X_test)
    delta_evaluation = metrics.evaluation.get_evaluation(feature_type, y_test, y_pred)
    print_log(logger, "On test set: delta: %s\tF-score: %s" % (best_model_num, delta_evaluation["F-score"]))
    print("F-score on test set: %s" % delta_evaluation["F-score"], file=freport)














