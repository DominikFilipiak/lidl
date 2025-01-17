#/bin/bash


PROJECT_HOME="/home/pt202342/projects/lidl"
CONDA_HOME="/home/pt202342/miniconda3"

source $CONDA_HOME/bin/activate id-diff
cd $PROJECT_HOME

# dataset=aldi-sphere-20-21-0.1
# for algorithm in "rqnsf" "maf" #gm
# do
	# size=12000
python run_experiments.py \
	--dataset $dataset \
	--algorithm $algorithm \
	--size $size \
	--seed $seed \
	--deltas $deltas \
	--device cuda 
# done
