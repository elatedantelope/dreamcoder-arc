#!/bin/sh
#SBATCH -p cbuild
#SBATCH -t 50:00
 
# Explicitly set the temporary directory to use (see below)
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a
module load SWIG/4.1.1-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
 
# Note the necessary --fakeroot option
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip 
pip install dill pyzmq matplotlib protobuf scikit-learn numpy scipy
#pip install git+https://github.com/MaxHalford/vose@fae179e5afa45f224204519c10957d087633ae60
pip install vose
pip install dill sexpdata pygame pycairo cairocffi psutil pypng Box2D-kengz graphviz frozendict pathos
pip install rich drawsvg pytest line_profiler wandb==0.18.7
pip install drawsvg
# Submit the job