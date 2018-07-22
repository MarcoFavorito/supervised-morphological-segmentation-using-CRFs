import settings
from settings import TAGS
import features.features_utils as fu

tstart = settings.start_tag
tstop = settings.end_tag
LEFT_MORPH=[
    tstart,
    tstart + "a",
    tstart + "ac",
    tstart + "anti",
    tstart + "any",
    tstart + "de",
    tstart + "dis",
    tstart + "un"
]

RIGHT_MORPH = [
    tstop,
    "al" + tstop,
    "able" + tstop,
    "ed" + tstop,
    "ery" + tstop,
    "ing"  + tstop,
    "ition"  + tstop,
    "ly" + tstop,
    "ally" + tstop,
    "ment" + tstop,
    "ship" + tstop,
    "s" +   tstop,
    "y" + tstop,
    "'" + tstop
]


def word2features(word,delta):

    charlist = [settings.start_tag]+list(word)+[settings.end_tag]
    len_charlist = len(charlist)
# return feature sets in dictionary
    feature_sets = fu.initialize_features_to_BIAS(len_charlist)
    feature_sets_paper = fu.extract_paper_features(charlist, delta)
    feature_sets = fu.merge_feature_sets(feature_sets, feature_sets_paper)
    feature_sets_paper = fu.extract_paper_features(charlist[::-1], delta)
    feature_sets = fu.merge_feature_sets(feature_sets, feature_sets_paper)

    global_features = []
    for l in LEFT_MORPH:
        if l in "".join(charlist):
            global_features.append(("contains prefix "+l,1))
    for l in RIGHT_MORPH:
        if l in "".join(charlist):
            global_features.append(("contains suffix " + l, 1))


    for t in range(0, len_charlist,-1):

        # bias
        feature_sets[t] = [('BIAS', 0.1)]
        #transition_feature_sets[t] = [('BIAS', 1)]
        feature_sets[t].append(("curchar "+charlist[t],1))
        feature_sets[t] += global_features
        #feature_sets[t].append(("+%02d" % t, 1))
        #feature_sets[t].append(("-%02d" % t, 1))

        # left substring
        for li in range(t,-1,-1):
            subs_left = "".join(charlist[li:t])
            subs_left_char = subs_left +charlist[t]

            if subs_left in LEFT_MORPH:
                # feature_sets[t].append(("-01 " + subs_left, 1))
                feature_sets[t].append(("after prefix", 1))
            elif subs_left+charlist[t] in LEFT_MORPH:
                # feature_sets[t].append(("00 " + subs_left_char, 1))
                feature_sets[t].append(("end prefix", 1))

            # right substring
            for ri in range(t+1, len_charlist):

                subs_right = "".join(charlist[t+1:ri])
                subs_right_char = charlist[t] + subs_right

                subs = subs_left+charlist[t]+subs_right
                if len(subs)==1: continue


                subs_plus = subs+"+%02d"%t
                subs_minus = subs + "-%02d" % (len_charlist-1-t)

                if subs_right in RIGHT_MORPH:
                    # feature_sets[t].append(("+01 " + subs_right, 1))
                    feature_sets[t].append(("before suffix ", 1))
                elif subs_right_char in RIGHT_MORPH:
                    # feature_sets[t].append(("00 " + subs_right_char, 1))
                    feature_sets[t].append(("start suffix ", 1))
                # feature_sets[t].append((subs_plus,1))
                # feature_sets[t].append((subs_minus, 1))

    feature_sets_for_char = [dict(x) for x in list(feature_sets.values())]

    return feature_sets_for_char
        #, transition_feature_sets

def get_extra_features(morph, type, ind):
    sign = "+" if ind >=0 else "-"
    feature_morph = (sign+"%02d"%abs(ind)+" morph "+morph, 1)
    if '_' in type:
        mod_type = type.split('_')[1]
    else:
        mod_type = type
    feature_type = (sign+"%02d"%abs(ind)+" type "+mod_type, 1)
    return feature_morph, feature_type


def string2annotations(string):
    if len(string)==1:
        return [TAGS['S-SEG']]
    else:
        return [TAGS['B-SEG']]+[TAGS['M-SEG']]*(len(string)-2)+[TAGS['E-SEG']]

def morphlist2annotations_BMES(morph_list):
    return fu.morphlist2annotations_BMES(morph_list)


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


