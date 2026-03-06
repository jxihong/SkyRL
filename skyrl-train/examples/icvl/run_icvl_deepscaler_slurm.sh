#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=zip_icvl_deepscaler
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --output=/home/rohin/SkyRL/logs/zip_icvl_deepscaler-%x-%j.out
#SBATCH --error=/home/rohin/SkyRL/logs/zip_icvl_deepscaler-%x-%j.err
#SBATCH --account=liquidai
#SBATCH --exclude=liquid-gpu-[030]

set -euo pipefail

# Network configuration
export PMI_DEBUG=1
export MPI_ROOT=/usr/mpi/gcc/openmpi-4.1.7a1/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_ROOT/lib
export OMPI_MCA_btl_tcp_if_include=bond0
export UCX_TLS=self,shm,tcp
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_13,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8"
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=1
export NCCL_SOCKET_IFNAME=bond0
export LC_CTYPE=en_US.UTF-8
export PYTHONUNBUFFERED=1

export WANDB_API_KEY=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
export UV_CACHE_DIR=/lambdafs/cache/rohin

cd $HOME/SkyRL/skyrl-train
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate && conda deactivate
conda activate sky_rl

ESTIMATOR=icvl ENABLE_THINKING=false bash examples/icvl/run_icvl_deepscaler.sh "$@"
