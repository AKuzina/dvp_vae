# """
# Code from https://github.com/openai/vdvae/blob/ea35b490313bc33e7f8ac63dd8132f3cc1a729b4/utils.py#L117
# """
import os
import socket

import torch
import torch.distributed as dist
import omegaconf
#
# # Change this to reflect your cluster layout.
# # The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 4
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# SETUP_RETRY_COUNT = 3
# GPU_ID = ""

def cleanup():
    dist.destroy_process_group()

def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

def setup_mpi(args):
    if dist.is_initialized():
        return
    print(torch.cuda.device_count(), 'GPUs available')
    with omegaconf.open_dict(args):
        args.mpi_size = mpi_size()
        args.local_rank = local_mpi_rank()
        args.rank = mpi_rank()
    print('Worls size:', args.mpi_size)
    print('RANK:', args.rank)
    hostname = "localhost"
    from mpi4py import MPI
    os.environ['MASTER_ADDR'] = MPI.COMM_WORLD.bcast(hostname, root=0)

    os.environ["RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(args.mpi_size)
    port = MPI.COMM_WORLD.bcast(_find_free_port(), root=0)
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(
        backend="nccl", init_method=f"env://",world_size=args.mpi_size, rank=args.rank)
    torch.cuda.set_device(args.local_rank)
    return args

def mpi_size():
    from mpi4py import MPI
    return MPI.COMM_WORLD.Get_size()

def mpi_rank():
    from mpi4py import MPI
    return MPI.COMM_WORLD.Get_rank()

def num_nodes():
    nn = mpi_size()
    if nn % GPUS_PER_NODE == 0:
        return nn // GPUS_PER_NODE
    return nn // GPUS_PER_NODE + 1

def gpus_per_node():
    size = mpi_size()
    if size > 1:
        return max(size // num_nodes(), 1)
    return 1

def local_mpi_rank():
    return mpi_rank() % gpus_per_node()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0