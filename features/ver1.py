import settings
from settings import TAGS
import features.features_utils as fu


def word2features(word, delta):
    """
    This is the implementation for the feature extraction process suggested by the paper
    return a list of pairs: (feature, activation)
    :param word:
    :param delta:
    :return:
    """
    charlist = [settings.start_tag]+list(word)+[settings.end_tag]
    len_charlist = len(charlist)

    # return feature sets in dictionary
    feature_sets = {}

    feature_sets = fu.extract_paper_features(charlist,delta)
    feature_sets_for_char = [dict(x) for x in list(feature_sets.values())]

    return feature_sets_for_char
        #, transition_feature_sets

def string2annotations(string):
    if len(string)==1:
        return [TAGS['S-SEG']]
    else:
        return [TAGS['B-SEG']]+[TAGS['M-SEG']]*(len(string)-2)+[TAGS['E-SEG']]

def morphlist2annotations_BMES(morph_list):
    return [TAGS['START']]+[ label for sublabels in [string2annotations(morph) for morph in morph_list] for label in sublabels]+[TAGS['STOP']]


def labels2segments(word, labels):
    segments = []
    covered = 0
    new_labels = labels[1:len(labels)-1]
    for i in range(0,len(new_labels)):
        if new_labels[i]==TAGS['S-SEG']:
            segments.append(word[i])
            covered=i+1
        elif new_labels[i]==TAGS['E-SEG']:
            segments.append(word[covered:i+1])
            covered = i+1
    if covered != len(word):
        segments.append(word[covered:])
    return segments