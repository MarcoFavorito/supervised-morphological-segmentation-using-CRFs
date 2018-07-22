import settings
from settings import TAGS
import features.ver1 as v1
import features.ver2 as v2
import features.ver3 as v3
import features.ver4 as v4

def get_features_and_labels(data, delta, feature_type):
    if feature_type in ["ver1"]:
        X_train, y_train = get_data(data, delta, v1.word2features, v1.morphlist2annotations_BMES)
    elif feature_type in ["ver2"]:
        X_train, y_train = get_data(data, delta, v1.word2features, v2.morphlist2annotations_BM)
    elif feature_type in ["ver3"]:
        X_train, y_train = get_data(data, delta, v3.word2features, v3.morphlist2annotations_BMES)
    elif feature_type in ["ver4"]:
        X_train, y_train = get_data(data, delta, v4.word2features, v4.morphlist2annotations_BMES)

    else:
        pass
    return X_train, y_train


def get_data(data, delta, w2f, m2a):
    XY_train = [
        (w2f(inst[0], delta), m2a(inst[1:])) for inst in data
        ]
    X_train = list(map(lambda x: x[0], XY_train))
    y_train = list(map(lambda x: x[1], XY_train))
    return X_train, y_train