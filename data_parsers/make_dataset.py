import settings


def parse_row(row):
    spl_row = row.split(',')[0]
    spl_row = spl_row.split()

    ann_row = list(map(lambda x: x.split(":")[0], spl_row))
    while "~" in ann_row:
        del ann_row[ann_row.index("~")]
    return ann_row

def parse_row_ver3(row):
    spl_row = row.split(',')[0]
    spl_row_sep = spl_row.split()
    ann_row = [x.split(":") for x in spl_row_sep]
    return ann_row


def parse_file_no_word_annotations(filedesc):
    return list(map(parse_row, filedesc))


def parse_file_with_word_annotations(filedesc):
    return list(map(parse_row_ver3, filedesc))
    pass


#def parse_file_by_feature_type(filedesc, feature_type):
def parse_file(filedesc):
    dataset = parse_file_no_word_annotations(filedesc)
    return dataset

