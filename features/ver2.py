import settings
from settings import TAGS
import features.features_utils as fu

def string2annotations_BM(string):
    return fu.string2annotations_BM(string)

def morphlist2annotations_BM(morph_list):
    return fu.morphlist2annotations_BM(morph_list)


def labels2segments(word, labels):
    segments = fu.labels2segments(word,labels)
    return segments
