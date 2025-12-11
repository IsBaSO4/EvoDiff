"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 2

SETUP_RETRY_COUNT = 2

CUDA_VISIBLE_DEVICES = [1,2,6,7]  # 指定要使用的 GPU
GPUS_PER_NODE = len(CUDA_VISIBLE_DEVICES)  # 现在是 4

def setup_dist(single_gpu=False):
    if dist.is_initialized():
        return
    if not single_gpu:
        # 根据 rank 选择对应的 GPU
        rank = MPI.COMM_WORLD.Get_rank()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES[rank % GPUS_PER_NODE])
#----------3.21修改------------
# 添加这个变量到文件开头
# CUDA_VISIBLE_DEVICES = [1,2,5,7]  # 指定要使用的 GPU
# GPUS_PER_NODE = len(CUDA_VISIBLE_DEVICES)  # 现在是 4

# def setup_dist(single_gpu=False):
#     if dist.is_initialized():
#         return
#     if not single_gpu:
#         # 根据 rank 选择对应的 GPU
#         rank = MPI.COMM_WORLD.Get_rank()
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES[rank % GPUS_PER_NODE])
        # comm = MPI.COMM_WORLD
        # backend = "gloo" if not th.cuda.is_available() else "nccl"

        # if backend == "gloo":
        #     hostname = "localhost"
        # else:
        #     hostname = socket.gethostbyname(socket.getfqdn())
        # os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
        # os.environ["RANK"] = str(comm.rank)
        # os.environ["WORLD_SIZE"] = str(comm.size)

        # port = comm.bcast(_find_free_port(), root=0)
        # os.environ["MASTER_PORT"] = str(port)
        # dist.init_process_group(backend=backend, init_method="env://")

# def setup_dist(single_gpu=False):
#     if dist.is_initialized():
#         return
        
#     if not single_gpu:
#         # 获取 MPI 的 rank 和 world size
#         rank = MPI.COMM_WORLD.Get_rank()
#         world_size = MPI.COMM_WORLD.Get_size()
        
#         # 根据 rank 选择对应的 GPU
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES[rank % GPUS_PER_NODE])
        
#         # 设置主节点地址和端口
#         os.environ["MASTER_ADDR"] = "localhost"  # 如果是多机训练，需要设置为主节点的IP
#         os.environ["MASTER_PORT"] = "29500"      # 可以自定义端口号
        
#         # 初始化进程组
#         dist.init_process_group(
#             backend="nccl",  # 使用NCCL后端
#             world_size=world_size,
#             rank=rank
#         )
#     else:
#         # 单GPU模式
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES[0])
    
#----------3.21修改------------


#------原来的是这个--------
def setup_dist1(single_gpu=False):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    if single_gpu == False:
       os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if not dist.is_initialized():
        return
    
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()