#!/bin/bash
#SBATCH -p debug        # 指定GPU队列
#SBATCH -o ./temp/output.txt  # 指定作业标准输出文件，%j为作业号  SBATCH -o output_%j.txt
#SBATCH -e ./temp/err.txt    # 指定作业标准错误输出文件  SBATCH -e err_%j.txt
#SBATCH -n 1            # 指定CPU总核心数
#SBATCH --gres=gpu:1    # 指定GPU卡数
#SBATCH -D .        # 指定作业执行路径为当前目录

# 加载CUDA模块（如果需要）
module load cuda/12.1

# 激活 Conda 环境
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate RAG-Diffusion

# 设置 LD_LIBRARY_PATH，只使用 conda 下的 .so 文件
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
echo "[INFO] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# 可选调试：查看 .so 是否在位
echo "[DEBUG] Verifying libcudnn_graph.so presence:"
ls -l $CONDA_PREFIX/lib/libcudnn_graph.so*

# Job 执行主体
echo "Job started at $(date)"
python Quick_Start.py
echo "Job ended at $(date)"

# 退出环境
conda deactivate