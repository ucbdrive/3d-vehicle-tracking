import os
from os.path import join, dirname, abspath

import progressbar
from easydict import EasyDict as edict

cfg = edict()
# ROOT path is at 3d-vehicle-tracking/3d-tracking
cfg.ROOT = abspath(join(dirname(__file__), os.pardir))
cfg.CHECKPOINT_PATH = join(cfg.ROOT, 'checkpoint')
cfg.OUTPUT_PATH = join(cfg.ROOT, 'output')
cfg.DATA_PATH = join(cfg.ROOT, 'data')

cfg.PGBAR_WIDGETS = [
    progressbar.Percentage(), ' ',
    progressbar.Timer(), ' ',
    progressbar.ETA(),
    progressbar.Bar(),
    progressbar.DynamicMessage('loss'),
]

# KITTI
cfg.KITTI = edict()
cfg.KITTI.FOCAL_LENGTH = 721.5377
cfg.KITTI.H = 384
cfg.KITTI.W = 1248
cfg.KITTI.NUM_OBJECT = 100

# KITTI object
cfg.KITTI.OBJECT = edict()
cfg.KITTI.OBJECT.CLASS_LIST = ['Car', 'Van', 'Truck']
cfg.KITTI.OBJECT.PATH = join(cfg.DATA_PATH, 'kitti_object')
cfg.KITTI.OBJECT.IM_PATH = join(cfg.KITTI.OBJECT.PATH, 'training', 'image_02')
cfg.KITTI.OBJECT.LABEL_PATH = join(cfg.KITTI.OBJECT.PATH, 'training', 'label_02')
cfg.KITTI.OBJECT.PRED_PATH = join(cfg.OUTPUT_PATH, 'kitti_object', 'training', 'pred_02')
cfg.KITTI.OBJECT.PKL_FILE = join(cfg.KITTI.OBJECT.LABEL_PATH,
                                 'kitti_train_obj_labels.pkl')
cfg.KITTI.OBJECT.OXT_PATH = join(cfg.KITTI.OBJECT.PATH,
                                 'training', 'oxts')  # Not available
cfg.KITTI.OBJECT.CALI_PATH = join(cfg.KITTI.OBJECT.PATH,
                                  'training', 'calib')  # Not available

# KITTI tracking
cfg.KITTI.TRACKING = edict()
cfg.KITTI.TRACKING.CLASS_LIST = ['Car', 'Van', 'Truck']
cfg.KITTI.TRACKING.PATH = join(cfg.DATA_PATH, 'kitti_tracking')
cfg.KITTI.TRACKING.IM_PATH = join(cfg.KITTI.TRACKING.PATH, 'training', 'image_02')
cfg.KITTI.TRACKING.LABEL_PATH = join(cfg.KITTI.TRACKING.PATH,
                                     'training', 'label_02')
cfg.KITTI.TRACKING.PRED_PATH = join(cfg.OUTPUT_PATH, 'kitti_tracking', 'training', 'pred_02')
cfg.KITTI.TRACKING.PKL_FILE = join(cfg.KITTI.TRACKING.LABEL_PATH,
                                   'kitti_train_trk_labels.pkl')
cfg.KITTI.TRACKING.OXT_PATH = join(cfg.KITTI.TRACKING.PATH, 'training', 'oxts')
cfg.KITTI.TRACKING.CALI_PATH = join(cfg.KITTI.TRACKING.PATH, 'training', 'calib')

tracking_vid_list = [str(num).zfill(4) for num in range(21)]
cfg.KITTI.TRACKING.TRAIN_SPLIT = tracking_vid_list[:15]
cfg.KITTI.TRACKING.VAL_SPLIT = tracking_vid_list[15:]

# GTA dataset
cfg.GTA = edict()
cfg.GTA.FOCAL_LENGTH = 935.3074360871937
cfg.GTA.H = 1080
cfg.GTA.W = 1920

# GTA object
cfg.GTA.TRACKING = edict()
cfg.GTA.TRACKING.CLASS_LIST = ['Car', 'Truck']
cfg.GTA.TRACKING.PATH = join(cfg.DATA_PATH, 'gta5_tracking')
cfg.GTA.TRACKING.IM_PATH = join(cfg.GTA.TRACKING.PATH, 'train', 'image')
cfg.GTA.TRACKING.LABEL_PATH = join(cfg.GTA.TRACKING.PATH, 'train', 'label')
cfg.GTA.TRACKING.PRED_PATH = join(cfg.OUTPUT_PATH, 'gta5_tracking', 'train', 'pred')
cfg.GTA.TRACKING.PKL_FILE = join(cfg.GTA.TRACKING.LABEL_PATH, 'gta_train_labels.pkl')
cfg.GTA.TRACKING.CLASS_PARSER = {
    'Compacts': 'Car',
    'Sedans': 'Car',
    'SUVs': 'Car', #'Van'
    'Coupes': 'Car',
    'Muscle': 'Car',
    'Sports Classics': 'Car',
    'Sports': 'Car',
    'Super': 'Car',
    # 8: 'Motorcycles',
    'Industrial': 'Truck',
    'Utility': 'Truck',  # usally truck
    'Vans': 'Car', #'Van'
    'Off-road': 'Car',
    'Service': 'Car',  # usually taxi
    'Emergency': 'Car',  # usually police car
    'Military': 'Car',
    'Commercial': 'Truck'
}
