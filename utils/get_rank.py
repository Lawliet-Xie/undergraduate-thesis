import os

def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))

def get_global_rank():
    return int(os.environ.get('RANK', 0))

def get_world_size():
    return int(os.environ.get('WORLD_SIZE', 0))


