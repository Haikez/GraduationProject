"""STIDGCN configuration for PEMS04. Run from the BasicTS repository root."""
import os
from easydict import EasyDict

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings
from .arch import STIDGCN
from .ranger21 import Ranger

DATA_NAME = "PEMS04"
settings = get_regular_settings(DATA_NAME)
INPUT_LEN = settings["INPUT_LEN"]
OUTPUT_LEN = settings["OUTPUT_LEN"]
RATIO = settings["TRAIN_VAL_TEST_RATIO"]

CFG = EasyDict()
CFG.DESCRIPTION = "STIDGCN on PEMS04"
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET = EasyDict(TYPE=TimeSeriesForecastingDataset, NAME=DATA_NAME,
                       PARAM=EasyDict(dataset_name=DATA_NAME, train_val_test_ratio=RATIO,
                                      input_len=INPUT_LEN, output_len=OUTPUT_LEN))
CFG.SCALER = EasyDict(TYPE=ZScoreScaler,
                      PARAM=EasyDict(dataset_name=DATA_NAME, train_ratio=RATIO[0],
                                     norm_each_channel=settings["NORM_EACH_CHANNEL"],
                                     rescale=settings["RESCALE"]))
CFG.MODEL = EasyDict(NAME="STIDGCN", ARCH=STIDGCN,
                     PARAM=dict(input_dim=3, num_nodes=307, channels=48,
                                granularity=288, input_len=INPUT_LEN,
                                output_len=OUTPUT_LEN, dropout=0.1),
                     FORWARD_FEATURES=[0, 1, 2], TARGET_FEATURES=[0])
CFG.METRICS = EasyDict(FUNCS=EasyDict(MAE=masked_mae, MAPE=masked_mape,
                                      RMSE=masked_rmse), TARGET="MAE",
                       NULL_VAL=settings["NULL_VAL"])
CFG.TRAIN = EasyDict(NUM_EPOCHS=300,
                     CKPT_SAVE_DIR=os.path.join("checkpoints", "STIDGCN", DATA_NAME),
                     LOSS=masked_mae,
                     OPTIM=EasyDict(TYPE=Ranger, PARAM=dict(lr=0.001, weight_decay=0.0001)),
                     CLIP_GRAD_PARAM=dict(max_norm=5.0),
                     DATA=EasyDict(BATCH_SIZE=64, SHUFFLE=True))
CFG.VAL = EasyDict(INTERVAL=1, DATA=EasyDict(BATCH_SIZE=64))
CFG.TEST = EasyDict(INTERVAL=300, DATA=EasyDict(BATCH_SIZE=64))
CFG.EVAL = EasyDict(HORIZONS=[3, 6, 12], USE_GPU=True)
