VOC_CLASSES_LABEL_TO_ID = {
    'aeroplane':    0, 'bicycle': 1, 'bird':   2, 'boat':       3, 'bottle':     4, 
    'bus':          5, 'car':     6, 'cat':    7, 'chair':      8, 'cow':        9, 
    'diningtable': 10, 'dog':    11, 'horse': 12, 'motorbike': 13, 'person':    14, 
    'pottedplant': 15,'sheep':   16, 'sofa':  17, 'train':     18, 'tvmonitor': 19, 
    'background':  20}

VOC_CLASSES_ID_TO_LABEL = {
    0:'aeroplane'   , 1:'bicycle', 2:'bird'  , 3:'boat'      , 4:'bottle', 
    5:'bus'         , 6:'car'    , 7:'cat'   , 8:'chair'     , 9:'cow' , 
    10:'diningtable', 11:'dog'   , 12:'horse', 13:'motorbike', 14:'person', 
    15:'pottedplant', 16:'sheep' , 17:'sofa' , 18:'train'    , 19:'tvmonitor', 
    20:'background'}

SSD300_DEFAULTBOX_CONFIG = {
    'steps':     [8., 16., 32., 64., 100., 300.],
    'image_size': 300.,
    'min_sizes': [30., 60., 111., 162., 213., 264.],
    'max_sizes': [60., 111., 162., 213., 264., 315.],
    'aspect_ratios': [[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
    'extra_layer_size': [38, 19, 10, 5, 3, 1]}

DATA_CODER_CONFIG = {
    'conf_threshold': 0.01,
    'iou_threshold': 0.5,
    'nms_threshold': 0.5
}

DATASET_CONFIG = {
    'mean': [104., 117., 123.],
    'num_classes': 21
}