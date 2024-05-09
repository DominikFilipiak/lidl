
PROJECT_HOME="/home/pt202342/projects/lidl"
CONDA_HOME="/home/pt202342/miniconda3"

source $CONDA_HOME/bin/activate id-diff
cd $PROJECT_HOME

for algorithm in "rqnsf" "maf"
do
    seed=42
    deltas="0.009,0.01,0.011"

    for dataset in "aldi-sphere-20-21-0.1" "aldi-sphere-20-21-0.2" "aldi-sphere-20-21-0.4" "aldi-sphere-20-21-0.8" "aldi-sphere-20-40-0.1" "aldi-sphere-20-40-0.2" "aldi-sphere-20-40-0.4" "aldi-sphere-20-40-0.8" "aldi-gaussian-20-20-0.1" "aldi-gaussian-20-20-0.2" "aldi-gaussian-20-20-0.4" "aldi-gaussian-20-20-0.8" "aldi-gaussian-20-40-0.01" "aldi-gaussian-20-40-0.05" "aldi-gaussian-20-40-0.1" "aldi-gaussian-20-40-0.2" "aldi-gaussian-20-40-0.4" "aldi-gaussian-20-40-0.8" "aldi-gaussian_saw-1-3-02-11"
    do
        size=12000
        epochs=500
        srun --time=0-12:00:00 --partition=common --qos=32gpu14d --export=ALL,algorithm=${algorithm},seed=${seed},dataset=${dataset},size=${size},epochs=${epochs},deltas="${deltas}" --gres=gpu:1 --job-name="${algorithm}-${dataset}" bash srun-single.sh &
    done

    for dataset in "aldi-gaussian_saw-10-15-0.2-11" "aldi-gaussian_saw-10-16-0.2-11" "aldi-gaussian_saw-10-17-0.2-11" "aldi-gaussian_saw-10-18-0.2-11" "aldi-gaussian_saw-10-20-0.2-11" "aldi-gaussian_saw-10-30-0.2-11" "aldi-gaussian_saw-15-15-0.2-11" "aldi-gaussian_saw-15-16-0.2-11" "aldi-gaussian_saw-15-17-0.2-11" "aldi-gaussian_saw-15-18-0.2-11" "aldi-gaussian_saw-15-20-0.2-11" "aldi-gaussian_saw-15-30-0.2-11"
    do
        size=120000
        epochs=10000
        srun --time=0-12:00:00 --partition=common --qos=32gpu14d --export=ALL,algorithm=${algorithm},seed=${seed},dataset=${dataset},size=${size},epochs=${epochs},deltas="${deltas}" --gres=gpu:1 --job-name="${algorithm}-${dataset}" bash srun-single.sh &
    done
done