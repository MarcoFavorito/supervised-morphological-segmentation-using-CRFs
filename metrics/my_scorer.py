import metrics.evaluation as e
import metrics.evaluation

def get_evaluation(y_test, y_pred):
    flat_y_test = [i for subl in y_test for i in subl]
    flat_y_pred = [i for subl in y_pred for i in subl]

    H,I,D = e.compute_HID(flat_y_test, flat_y_pred)
    Precision, Recall, FScore = e.compute_PRF(H,I,D)

    evaluation = {}
    evaluation["E-count_test"] = flat_y_test.count('E-SEG')
    evaluation["S-count_test"] = flat_y_test.count('S-SEG')
    evaluation["E-count_pred"] = flat_y_pred.count('E-SEG')
    evaluation["S-count_pred"] = flat_y_pred.count('S-SEG')
    evaluation["H"]=H
    evaluation["I"]=I
    evaluation["D"]=D
    evaluation["Precision"] = Precision
    evaluation["Recall"] = Recall
    evaluation["F-score"] = FScore

    return FScore


def get_evaluation_BM(y_test, y_pred):
    flat_y_test = [i for subl in y_test for i in subl]
    flat_y_pred = [i for subl in y_pred for i in subl]
    flat_y_test = metrics.evaluation.fromBM2BMES(flat_y_test)
    flat_y_pred = metrics.evaluation.fromBM2BMES(flat_y_pred)

    H,I,D = e.compute_HID(flat_y_test, flat_y_pred)
    Precision, Recall, FScore = e.compute_PRF(H,I,D)

    evaluation = {}
    evaluation["E-count_test"] = flat_y_test.count('E-SEG')
    evaluation["S-count_test"] = flat_y_test.count('S-SEG')
    evaluation["E-count_pred"] = flat_y_pred.count('E-SEG')
    evaluation["S-count_pred"] = flat_y_pred.count('S-SEG')
    evaluation["H"]=H
    evaluation["I"]=I
    evaluation["D"]=D
    evaluation["Precision"] = Precision
    evaluation["Recall"] = Recall
    evaluation["F-score"] = FScore

    return FScore