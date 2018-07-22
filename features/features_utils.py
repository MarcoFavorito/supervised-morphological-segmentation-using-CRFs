import settings
from settings import TAGS

def initialize_features_to_BIAS(len_charlist):
    features_sets = {}
    for t in range(len_charlist):
        features_sets[t] = [('BIAS', 0.1)]
    return features_sets


def extract_paper_features(charlist, delta):
    """
    This is the implementation for the feature extraction process suggested by the paper
    return a list of pairs: (feature, activation)
    :param charlist:
    :return:
    """
    len_charlist = len(charlist)
    feature_sets = {}
    feature_sets[0] = [('BIAS', 1), ('substring right '+TAGS['S-TAG'], 1)]
    feature_sets[len_charlist-1] = [('BIAS', 1), ('substring right '+TAGS['E-TAG'], 1)]

    for t in range(1, len_charlist - 1):

        # bias
        feature_sets[t] = [('BIAS', 1)]
        # transition_feature_sets[t] = [('BIAS', 1)]


        # left substring
        for start in range(max(t - delta, 0), t):
            feature = 'substring left ' + ''.join(charlist[start:t])
            feature_sets[t].append((feature, 1))

        # right substring
        for stop in range(t + 1, min(t + delta, len_charlist) + 1):
            feature = 'substring right ' + ''.join(charlist[t:stop])
            feature_sets[t].append((feature, 1))

    return feature_sets


def string2annotations_BMES(string):
    if len(string)==1:
        return [TAGS['E-SEG']]
    else:
        return [TAGS['B-SEG']]+[TAGS['M-SEG']]*(len(string)-2)+[TAGS['E-SEG']]

def morphlist2annotations_BMES(morph_list):
    return [TAGS['START']]+[ label for sublabels in [string2annotations_BMES(morph) for morph in morph_list] for label in sublabels]+[TAGS['STOP']]

def labels2segments(word, labels):
    segments = []
    covered = 0
    new_labels = labels[1:len(labels)-1]
    for i in range(0,len(new_labels)):
        if new_labels[i]==TAGS['B-SEG']:
            segments.append(word[covered:i+1])
            covered = i+1
    if covered != len(word):
        segments.append(word[covered:])
    return segments


def string2annotations_BM(string):
    if len(string)==1:
        return [TAGS['B-SEG']]
    else:
        return [TAGS['B-SEG']]+[TAGS['M-SEG']]*(len(string)-1)

def morphlist2annotations_BM(morph_list):
    return [TAGS['START']]+[ label for sublabels in [string2annotations_BM(morph) for morph in morph_list] for label in sublabels]+[TAGS['STOP']]


def merge_feature_sets(fs1,fs2):
    assert len(fs1)!= len(fs2)
    new_fs = {}
    for t in range(len(fs1)):
        new_fs[t] += fs1[t]
        new_fs[t] += fs2[t]
    return new_fs