import config

HW_DATA_TRAIN = config.HW_DATA_TRAIN
HW_DATA_DEV = config.HW_DATA_DEV
HW_DATA_TEST = config.HW_DATA_TEST
HW_DATA_TRAIN_EXTRA_POINTS = config.HW_DATA_TRAIN_EXTRA_POINTS

MAX_ALLOWABLE_DELTA = 25

VERBOSE = False
log_fmt = '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'

MODES = ["train", "test", "demo"]
FEATURE_TYPES = ["ver1", "ver2"]
DEMOS = ["demo01", "demo02", "demo03"]


label_set=['START','STOP','B_SEG','M_SEG','E_SEG','S_SEG']
start_tag = '<s>'
end_tag = '</s>'
TAGS = {
    'START':'START',
    'STOP' :'STOP',
    'B-SEG':'B-SEG',
    'M-SEG':'M-SEG',
    'E-SEG':'E-SEG',
    'S-SEG':'S-SEG',
    'S-TAG':'<s>',
    'E-TAG':'</s>'
}

METRICS_DIR = "output/metrics"


class VersionSettings(object):

    def __init__(self, feature_extraction_function):
        self.feature_extraction_function = feature_extraction_function

DEMO_01_PARAMS = {
    "id": 1,
    "name": "demo_01",
    "description": "Try different size of the training set and compare performances",
    "train_file":   HW_DATA_TRAIN,
    "dev_file":   HW_DATA_DEV,
    "test_file":    HW_DATA_TEST,
    "output_folder": "output/demo_01"
}
DEMO_02_PARAMS = {
    "id": 2,
    "name": "demo_02",
    "description": "Execute grid search over param_grid",
    "train_file":   HW_DATA_TRAIN,
    "dev_file":   HW_DATA_DEV,
    "test_file":    HW_DATA_TEST,
    "output_folder": "output/demo_02",
    "param_grid": config.DEMO_02_PARAM_GRID

}

DEMO_03_PARAMS = {
    "id": 3,
    "name": "demo_03",
    "description": "Use the extra point dataset and compare performances",
    "train_file":   HW_DATA_TRAIN,
    "dev_file":   HW_DATA_DEV,
    "test_file":    HW_DATA_TEST,
    "output_folder": "output/demo_03",
    "param_grid": config.DEMO_03_PARAM_GRID,
    "train_file_extra_points":   HW_DATA_TRAIN_EXTRA_POINTS,
}
DEMO_PARAMS = {
    1: DEMO_01_PARAMS,
    2: DEMO_02_PARAMS,
    3: DEMO_03_PARAMS
}

def get_demo_settings(demo_id):
    return DEMO_PARAMS[demo_id]
