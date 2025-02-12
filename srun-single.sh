#/bin/bash


PROJECT_HOME="/home/pt202342/projects/lidl"
CONDA_HOME="/home/pt202342/miniconda3"

source $CONDA_HOME/bin/activate id-diff
cd $PROJECT_HOME

python run_experiments.py \
	--dataset $dataset \
	--algorithm $algorithm \
	--size $size \
	--seed $seed \
	--deltas $deltas \
	--bs $bs \
	--layers $layers \
	--hidden $hidden \
	--device cuda 

