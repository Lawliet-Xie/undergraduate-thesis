import os
from yacs.config import CfgNode as CN
import time
from runx.logx import logx
from utils import get_global_rank

time_str = time.strftime('%m-%d-%H-%M')
# ================= basic ====================
_C = CN()
_C.SEED = 42
_C.WORKERS = 8
_C.ROOT = ''

# ================= dataset ====================
_C.DATA = CN()
_C.DATA.SOURCE = 'mr'
_C.DATA.TARGET = 'ct'
_C.DATA.ROOTA_TEST = _C.ROOT + 'data/mr_test/'
_C.DATA.ROOTB_TEST = _C.ROOT + 'data/ct_test/'

# ================= training ====================
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.NEPOCHS = 50
_C.TRAIN.DISPLAY_FREQ = 5000
_C.TRAIN.SAVE_EPOCH_FREQ = 5
_C.TRAIN.OUTPUT_ROOT = _C.ROOT + 'outputs/uda/'
_C.TRAIN.OUTPUT_LOG = _C.TRAIN.OUTPUT_ROOT + f'{time_str}/'
_C.TRAIN.OUTPUT_CKPT = _C.TRAIN.OUTPUT_LOG + 'ckpt/'
_C.TRAIN.PRETRAINED_CKPT = _C.ROOT + 'pretrained/'

# ================= optimizer ====================
_C.OPTIM = CN()
_C.OPTIM.NAME = 'Adam'
_C.OPTIM.GEN_LR = 2e-4
_C.OPTIM.DIS_LR = 2e-4
_C.OPTIM.BETA1 = 0.5
_C.OPTIM.WEIGHT_DECAY = 2e-4

# ================= models ====================
_C.MODEL = CN()
_C.MODEL.NAME = 'Model'
_C.MODEL.IN_CH = 3
_C.MODEL.NUM_CLASSES = 5
_C.MODEL.NUM_CONTENTS = 8
_C.MODEL.Z_DIM = 256

# ================= loss ======================
_C.LOSS = CN()
_C.LOSS.LAMBDA_DICE = 1
_C.LOSS.LAMBDA_CYCLE = 10
_C.LOSS.LAMBDA_SEG = 0.5
_C.LOSS.LAMBDA_ADV = 2
_C.LOSS.LAMBDA_ID = 2.5
_C.LOSS.LAMBDA_PSEUDO = 0.1
_C.LOSS.LAMBDA_ORTHO = 1
_C.LOSS.LAMBDA_DIS = 0.5 * _C.LOSS.LAMBDA_ADV
_C.LOSS.LAMBDA_KL = 0.01
_C.LOSS.GAN_TYPE = 'lsgan'


logx.initialize(
    logdir=_C.TRAIN.OUTPUT_LOG,   # 日志存储目录，不存在会自动创建
    hparams=None,  # 配合runx使用，超参数都存下来
    tensorboard=True,  # 是否写入tensorboard文件，无需手动writer
    no_timestamp=False,  # 是否不启用时间戳命名
    global_rank=get_global_rank(),  # 分布式训练防止多输出
    eager_flush=True  # 打开后即使tensorboard写入过快也不会丢失
)


def get_default_cfg():
    cfg = _C.clone()
    os.makedirs(cfg.TRAIN.OUTPUT_LOG, exist_ok=True)
    os.makedirs(cfg.TRAIN.OUTPUT_CKPT, exist_ok=True)

    return cfg


