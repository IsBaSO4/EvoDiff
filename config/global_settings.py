# ig_with_ea_sample.py
SAMPLE_BATCH_SIZE = 50 # batchsize
SAMPLE_LOG_ROOT = './logs_sample' # Path of logs
SAMPLE_MODEL_PATH = '' # diffusion model checkpoint
SAMPLE_CLASSIFIER_PATH = '' # gudiance classifier checkpoint
SAMPLE_CLASSIFIER_SCALE = 0.00 # information gain gudiance scale. The default value is 0.0, which means it is not used.
CFG = 0.0 # classifier free scale. The default value is 0.0, which means it is not used.
SAMPLE_DATASET_DIR = './logs_sample' # Path of generated data
SAMPLE_CATEGORY_NAME_LIST = ["0","1","2","3","4","5","6","7","8","9","10","11","12",] # List of image category names, len of the list is the num of categories.
SAMPLE_CATEGORY_NUM_LIST =  [100,100,100,100,100,100,100,100,100,100,100,100,100]  # Number of images generated per category
SAMPLE_DATASET_NAME = '' # Name of generated data