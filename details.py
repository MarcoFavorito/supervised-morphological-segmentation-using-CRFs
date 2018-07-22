from collections import Counter
import sklearn_crfsuite.metrics as mtr

def print_transitions(trans_features, file=None):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight), file=file)

def print_details(crf, file=None):

    print("\nTop likely transitions:", file=file)
    print_transitions(Counter(crf.transition_features_).most_common(20), file=file)

    print("\nTop unlikely transitions:", file=file)
    print_transitions(Counter(crf.transition_features_).most_common()[-20:], file=file)

    def print_state_features(state_features, file=None):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr), file=file)

    print("Top positive:", file=file)
    print_state_features(Counter(crf.state_features_).most_common(100), file=file)

    print("\nTop negative:", file=file)
    print_state_features(Counter(crf.state_features_).most_common()[-100:], file=file)

    #print("\nflat_accuracy_score:")
    #mtr.flat_accuracy_score()

def print_gridsearch_details(grid_search, file=None):
    print('best params:', grid_search.best_params_, file=file)
    print('best CV score:', grid_search.best_score_, file=file)
    print('model size: {:0.2f}M'.format(grid_search.best_estimator_.size_ / 1000000), file=file)

    crf = grid_search.best_estimator_

    print_details(crf, file=file)