#/bin/bash

cd /home/pt202342/projects/lidl

dataset=aldi-sphere-20-21-0.1
for algorithm in "rqnsf" "maf" #gm
do
	size=12000
	python run_experiments.py \
		--dataset $dataset \
		--algorithm $algorithm \
		--size $size \
		--seed 42 \
		--deltas "0.009,0.01,0.011" \
		--device cuda \
		--epochs 100
done
