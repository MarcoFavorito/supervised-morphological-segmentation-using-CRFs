from settings import TAGS
import copy

def get_evaluation(feature_type, y_test, y_pred):
    flat_y_test = [i for subl in y_test for i in subl]
    flat_y_pred = [i for subl in y_pred for i in subl]
    if feature_type in ["ver2"]:
        flat_y_test = fromBM2BMES(flat_y_test)
        flat_y_pred = fromBM2BMES(flat_y_pred)
    H,I,D = compute_HID(flat_y_test, flat_y_pred)
    Precision, Recall, FScore = compute_PRF(H,I,D)

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

    return evaluation


def fromBM2BMES(y_pred_old):
    y_pred = copy.deepcopy(y_pred_old)
    for i in range(len(y_pred)):
        if y_pred[i]==TAGS['B-SEG']:
            if y_pred[i+1]==TAGS['B-SEG']:
                y_pred[i]=TAGS['S-SEG']
        if y_pred[i]==TAGS['M-SEG'] and y_pred[i+1]!=TAGS['M-SEG']:
            y_pred[i]=TAGS['E-SEG']
    return y_pred


def compute_HID(y_test, y_pred):
    return compute_HID_BMES(y_test,y_pred)


def compute_HID_BMES_hide(y_test, y_pred):
    H=I=D=0
    for i in range(len(y_test)):
        t_seg=y_test[i]
        p_seg = y_pred[i]
        if t_seg=='E-SEG':
            if y_pred[i]=='E-SEG':
                H+=1
            else:
                D+=1
        elif y_test[i]=='S-SEG':
            if y_pred[i]!='S-SEG':
                D+=1
        elif y_pred[i]=='E-SEG' and y_test[i]!='E-SEG':
            I+=1
    return H,I,D

def compute_HID_BMES(y_test, y_pred):
    H=I=D=0
    for i in range(len(y_test)):
        t_seg=y_test[i]
        p_seg = y_pred[i]
        if p_seg==TAGS['E-SEG'] and t_seg==TAGS['E-SEG']:
            H+=1
        elif p_seg==TAGS['E-SEG'] and t_seg!=TAGS['E-SEG']:
            I+=1
        elif (t_seg==TAGS['S-SEG'] and p_seg!=TAGS['S-SEG']) or (t_seg==TAGS['E-SEG'] and p_seg!=TAGS['E-SEG']):
            D+=1
        else:
            pass
    return H,I,D



def compute_PRF(H,I,D):
    P = H/(H+I)
    R = H/(H+D)
    D = 2*P*R/(P+R)
    return P,R,D


