import argparse
import torch
import torch.backends.cudnn as cudnn
from configs import get_default_cfg, logx
from utils.utils import set_seed
from utils import get_global_rank, get_local_rank, get_world_size
import torch.distributed as dist
from trainer import Trainer
from newtrainer import NewTrainer
# from evaluate import evaluate


def train(cfg):
    set_seed(cfg.SEED + get_global_rank())
    cudnn.deterministic = True
    cudnn.benchmark = False

    device = 'cuda'
    logx.msg('======================= cfg =======================\n' + cfg.dump(indent=4))

    word_size = get_world_size()
    local_rank = get_local_rank()
    rank = get_global_rank()
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=word_size,
                            rank=rank)
    logx.msg(f'Distribute initialize......using device: {device}, world size: {word_size}.')
    torch.cuda.set_device(local_rank)
    opt = {'device':device, 'word_size':word_size, 'local_rank':local_rank, 'rank':rank}

    trainer = Trainer(cfg, opt)
    trainer.train()

if __name__ == '__main__':
    cfg = get_default_cfg()
    cfg.freeze()
    train(cfg)

