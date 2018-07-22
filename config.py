LIGHTWEIGHT_SETTING_FLAG = False


HW_DATA_TRAIN = "data/task1_data/training.eng.txt"
HW_DATA_DEV = "data/task1_data/dev.eng.txt"
HW_DATA_TEST = "data/task1_data/test.eng.txt"
HW_DATA_TRAIN_EXTRA_POINTS = "./data/extra_point_data/crowd-sourced-annotations.txt"

LIGHTWEIGHT_PARAM_GRID = {
    'max_iterations': [i for i in range(1, 19, 3)] + [i for i in range(20, 101, 20)],
    'epsilon': [1 * 10 ** i for i in range(0, -17, -4)]
}

HEAVYWEIGHT_PARAM_GRID = {
        'max_iterations': [i for i in range(1, 30, 1)] + [i for i in range(30, 100, 10)] + [i for i in range(100, 1000, 50)],
        'epsilon': [1 * 10 ** i for i in range(0, -15, -1)]
    }

if LIGHTWEIGHT_SETTING_FLAG:
    DEMO_02_PARAM_GRID = DEMO_03_PARAM_GRID = LIGHTWEIGHT_PARAM_GRID
else:
    DEMO_02_PARAM_GRID = DEMO_03_PARAM_GRID = HEAVYWEIGHT_PARAM_GRID

