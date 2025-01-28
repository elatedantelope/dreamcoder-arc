#!/bin/sh
#SBATCH -p gpu
#SBATCH -t 9:00:00
#SBATCH --gpus=1

# Explicitly set the temporary directory to use (see below)
module purge
module load 2023
#module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load PyTorch/2.1.2-foss-2023a
module load SWIG/4.1.1-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
 
 
# Note the necessary --fakeroot option
#../container_mod.img wandb offline

#../container_mod.img python -u arcbin/arc_mikel2.py -c 18 -t 3600 -R 2400 -i 2
source ../venv/bin/activate
wandb offline
python -u arcbin/arc_mikel2.py -c 18 -t 3600 -R 2400 -i 2
# Submit the job