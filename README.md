# ARC DreamCoder --Modified,
Based on the repository:
https://github.com/mxbi/dreamcoder-arc
which is described in the paper 
> **Neural networks for abstraction and reasoning: Towards broad generalization in machines**  
> *Mikel Bober-Irizar & Soumya Banerjee*

https://arxiv.org/abs/2402.03507

## Running on Snellius
Note: Although the scripts ```build_and_run.sh``` and ```ec/run.sh``` use the partition gpu, they do not actually use gpu, (there were to many errors in the dreamcoder code getting cuda to work). If these scripts are modified to run on a different partition, then the number of cores ```-c``` should be changed to the number of cores available.


Clone the repository, and then enter it. 

```bash
git clone https://github.com/elatedantelope/dreamcoder-arc.git
cd dreamcoder-arc 
```
Then you can either run 

```bash_
sbatch build_venv.sh
``` 
This should take less than 30 minutes and builds a python virtual environment ```venv``` in the current folder. After this job completes run
```bash
cd ec
sbatch run.sh
```
The output file will be called "results.txt"

Alternatively you can run ```sbatch build_and_run.sh``` which will first build the virtual enviroment and then start project. 


## Building the DreamCoder environment in MacOs

Install dependencies for the python packages:
```bash 
brew install swig
brew install libomp
brew cask install xquartz
brew install feh
brew install imagemagick
brew install pkg-config
```

Navigate with your terminal to the directory where you have cloned this repository. 

Create a conda environment, make it look nicer and activate it: 
```bash
conda create --prefix ./envs
conda config --set env_prompt '({name})'
conda activate ./envs
```
Install the following dependencies in the conda virtual environment.
```bash
#Needed for the original dreamcoder.
conda install -y  numpy dill pyzmq matplotlib protobuf scikit-learn scipy
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install dill sexpdata pygame pycairo cairocffi psutil pypng Box2D-kengz graphviz frozendict pathos vose
#Needed for the modifications from the original dreamcoder.
pip install rich drawsvg pytest wandb arckit line_profiler
```
You will need to create a wandb account, the academic plan is free. 

Run  ``` wandb login``` and enter your api key when prompted. 

Now you are ready and can try to run the experiment:
```bash
cd ec
python -u arcbin/arc_mikel2.py -c 1 -t 3600 -R 2400 -i 1
# -c 1: Run on 1 cores
# -t 3600: 3600 core-seconds per task
# -R 2400: Train recognition model for 2400s per iteration (all cores)
# -i 1: Run for one iteration
```
## Repo overview

Most of this repo follows the primary DreamCoder repo: https://github.com/ellisk42/ec.

Some helpful ARC-specific files:
- `ec/arcbin/arc_mikel2.py`: The main entry-point for DreamCoder on ARC
- `ec/dreamcoder/domains/arc/arcPrimitivesIC2.py`: PeARL definitions (domain-specific language).
- `ec/dreamcoder/domains/arc/main.py`: Recognition model
- `ec/arcbin/test_primitives_mikel2.py`: Very rough test harness to check that primitives aren't broken
- `arckit/`: Vendored early version of the [arckit](https://github.com/mxbi/arckit) library.
- `solved_tasks.md` shows a list of tasks solved by DreamCoder with corresponding programs.

## Building the DreamCoder environment

Since DreamCoder requires a complex set of dependencies, we follow the original repo in using [Singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html) containers. If you're familiar with Docker, this is quite similar.

The build is a 2-stage process. To use wandb, add a key in `singularity_mod` and create an `arc` project in your repo (or modify the entrypoint script to disable wandb).

```bash
cd ec/
# Build original DreamCoder (with fixes)
sudo singularity build container.img singularity

# Build additional packages and environment variables.
cd ..
sudo singularity build container_mod.img singularity_mod
```

Now, you have a `container.img` in the root of the repo which can be used to run the DreamCoder environment.

## Running experiments

```bash
# See all command-line arguments
../container_mod.img python -u arcbin/arc_mikel.py --help

# Getting 70/400 on training set
../container_mod.img python -u arcbin/arc_mikel2.py -c 76 -t 3600 -R 2400 -i 1
# -c 76: Run on 76 cores
# -t 3600: 3600 core-seconds per task
# -R 2400: Train recognition model for 2400s per iteration (all cores)
# -i 1: Run for one iteration

# 18/400 on evaluation set:
../container_mod.img python -u arcbin/arc_mikel2.py -c 76 -t 3600 -R 2400 -i 1 --evalset
# --evalset: Run on ARC-Hard

# Ablation without recognition model (1min per task)
../container_mod.img python -u arcbin/arc_mikel2.py -c 76 -t 60 -g -i 5 --task-isolation
# -g: disable recognition model
# --task-isolation: Don't share programs across multiple tasks
```

## Acknowledgements

The codebase in this repo is primarily based on the original [DreamCoder](https://github.com/ellisk42/ec) repository, licensed under MIT.

Additionally, I brought in some changes from Simon Alford's [bidir-synth](https://github.com/simonalford42/bidir-synth) repository as a starting point ([https://github.com/mxbi/arc/commit/a04da2471d327c7e39352048fed2fcd63408c3fd](commit)). The starting point was a combination of these two repos with some additional patches to get it compiling again after a couple years of changes in dependencies.

## License

The code in this repository is licensed under the MIT license. The original DreamCoder and bidir-synth repos are licened under the same license from their respective authors.

The ARC dataset (arckit/arc1.json) is licensed instead under the Apache license.