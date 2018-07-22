import settings
from settings import TAGS
import features.features_utils as fu
import features.ver1 as v1

def word2features(word,delta):

    charlist = [settings.start_tag]+list(word)+[settings.end_tag]
    len_charlist = len(charlist)
# return feature sets in dictionary
    feature_sets = {}

    # transition_feature_sets = {}
    # transition_feature_sets[0] = [('BIAS', 1)]
    # transition_feature_sets[len_charlist-1] = [('BIAS', 1)]

    #inst[1][0]= TAGS['START']+inst[1][0]
    #inst[-1][0] = inst[-1][0] + inst[-1][0]
    feature_sets = fu.initialize_features_to_BIAS(len_charlist)
    feature_sets_paper = fu.extract_paper_features(charlist, delta)
    for t in range(len_charlist):
        feature_sets[t] += feature_sets_paper[t]


    for t in range(1, len_charlist-1):

        #transition_feature_sets[t] = [('BIAS', 1)]

        feature_sets[t].append(('curchar %s' % charlist[t],1))
        feature = ("pos +%02d" % t, 1)
        feature_sets[t].append(feature)
        feature = ("pos -%02d" % (len_charlist-1-t), 1)
        feature_sets[t].append(feature)



        for left_right in range(1,min(t+1,len_charlist-1-t)):
            left_sub = charlist[t-left_right:t]
            curch = [charlist[t]]
            right_sub = charlist[t+1:t+1+left_right]
            subs = left_sub+curch+right_sub

            # imp_morph = [m for m in IMP_MORPH if m in "".join(subs) and curch[0] in m]
            # ftrs = [(m +" "+ str(m.index(curch[0])),1) for m in imp_morph]
            #
            # for f in ftrs:
            #     feature_sets[t].append(f)

            feature = 'left_right ' + ''.join(subs)
            feature_sets[t].append((feature, 1))

    # for i, (morph,type) in enumerate(morph_list):
    #     if morph=='~': continue
    #     for char in morph:
    #         feature_morph, feature_type = get_extra_features(morph,type, 0)
    #         feature_sets[index].append(feature_morph)
    #         feature_sets[index].append(feature_type)
    #         pr = morph_list[:i][::-1]
    #         su = morph_list[i+1:len(inst)]
    #         for prec_i, (preced_morph, preced_type) in enumerate(morph_list[:i][::-1]):
    #             feature_morph, feature_type = get_extra_features(preced_morph, preced_type, -(prec_i+1))
    #             feature_sets[index].append(feature_morph)
    #             feature_sets[index].append(feature_type)
    #         for succ_i, (succ_morph, succ_type) in enumerate(morph_list[i+1:len(inst)]):
    #             feature_morph, feature_type = get_extra_features(succ_morph, succ_type, (succ_i+1))
    #             feature_sets[index].append(feature_morph)
    #             feature_sets[index].append(feature_type)

    #feature = ("pos %02d"%index, 1)
    #feature_sets[index].append(feature)
    # index += 1

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
    return v1.string2annotations(string)

def morphlist2annotations_BMES(morph_list):
    return v1.morphlist2annotations_BMES(morph_list)


def labels2segments(word, labels):
    segments = v1.labels2segments(word,labels)
    return segments


