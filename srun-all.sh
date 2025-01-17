
PROJECT_HOME="/home/pt202342/projects/lidl"
CONDA_HOME="/home/pt202342/miniconda3"

source $CONDA_HOME/bin/activate id-diff
cd $PROJECT_HOME

for algorithm in "maf" #"rqnsf" #"maf"
do
    seed=42
    # deltas="0.009,0.01,0.011"

    # for dataset in "aldi-sphere-20-21-0.1" "aldi-sphere-20-21-0.2" "aldi-sphere-20-21-0.4" "aldi-sphere-20-21-0.8" "aldi-sphere-20-40-0.1" "aldi-sphere-20-40-0.2" "aldi-sphere-20-40-0.4" "aldi-sphere-20-40-0.8" "aldi-gaussian-20-20-0.1" "aldi-gaussian-20-20-0.2" "aldi-gaussian-20-20-0.4" "aldi-gaussian-20-20-0.8" "aldi-gaussian-20-40-0.01" "aldi-gaussian-20-40-0.05" "aldi-gaussian-20-40-0.1" "aldi-gaussian-20-40-0.2" "aldi-gaussian-20-40-0.4" "aldi-gaussian-20-40-0.8" "aldi-gaussian_saw-1-3-02-11"
    # do
    #     size=12000
    #     epochs=500
    #     srun --time=0-12:00:00 --partition=common --qos=32gpu14d --export=ALL,algorithm=${algorithm},seed=${seed},dataset=${dataset},size=${size},epochs=${epochs},deltas="${deltas}" --gres=gpu:1 --job-name="${algorithm}-${dataset}" bash srun-single.sh &
    # done
    

    # for dataset in "aldi-sphere-20-21-0.1" "aldi-sphere-20-21-0.2" "aldi-sphere-20-21-0.4" "aldi-sphere-20-21-0.8" "aldi-sphere-20-40-0.1" "aldi-sphere-20-40-0.2" "aldi-sphere-20-40-0.4" "aldi-sphere-20-40-0.8" "aldi-gaussian-20-20-0.1" "aldi-gaussian-20-20-0.2" "aldi-gaussian-20-20-0.4" "aldi-gaussian-20-20-0.8" "aldi-gaussian-20-40-0.01" "aldi-gaussian-20-40-0.05" "aldi-gaussian-20-40-0.1" "aldi-gaussian-20-40-0.2" "aldi-gaussian-20-40-0.4" "aldi-gaussian-20-40-0.8" "aldi-gaussian_saw-1-3-02-11"
    # do
    #     size=12000
    #     epochs=500
    #     srun --time=0-12:00:00 --partition=common --qos=32gpu14d --export=ALL,algorithm=${algorithm},seed=${seed},dataset=${dataset},size=${size},epochs=${epochs},deltas="${deltas}" --gres=gpu:1 --job-name="${algorithm}-${dataset}" bash srun-single.sh &
    # done

    # for dataset in "aldi-gaussian_saw-10-15-0.2-11" "aldi-gaussian_saw-10-16-0.2-11" "aldi-gaussian_saw-10-17-0.2-11" "aldi-gaussian_saw-10-18-0.2-11" "aldi-gaussian_saw-10-20-0.2-11" "aldi-gaussian_saw-10-30-0.2-11" "aldi-gaussian_saw-15-15-0.2-11" "aldi-gaussian_saw-15-16-0.2-11" "aldi-gaussian_saw-15-17-0.2-11" "aldi-gaussian_saw-15-18-0.2-11" "aldi-gaussian_saw-15-20-0.2-11" "aldi-gaussian_saw-15-30-0.2-11"
    # do
    #     size=120000
    #     epochs=10000
    #     srun --time=0-12:00:00 --partition=common --qos=32gpu14d --export=ALL,algorithm=${algorithm},seed=${seed},dataset=${dataset},size=${size},epochs=${epochs},deltas="${deltas}" --gres=gpu:1 --job-name="${algorithm}-${dataset}" bash srun-single.sh &
    # done

    
    # dataset_prefix="quantized_uniform"
    # for dataset_num in 3 5 9 17 33 65 129 257
    # do
    #     for deltas in "0.00015625;0.0003125;0.000625" "0.0003125;0.000625;0.00125" # "0.000625;0.00125;0.0025" "0.00125;0.0025;0.005" "0.0025;0.005;0.01" "0.005;0.01;0.02" # "0.01;0.02;0.04" "0.02;0.04;0.08" "0.04;0.08;0.16" "0.08;0.16;0.32" "0.16;0.32;0.64" "0.32;0.64;1.28"
    #     do
    #         dataset="${dataset_prefix}-${dataset_num}"
    #         epochs=500
    #         srun --time=0-6:00:00 --partition=common --qos=32gpu14d --export=ALL,algorithm=${algorithm},seed=${seed},dataset=${dataset},epochs=${epochs},deltas=${deltas},size=10000 --gres=gpu:1 --job-name="${algorithm}-${dataset}" bash srun-single.sh &
    #     done
    # done


    # dataset_prefix="quantized_uniform"
    for dataset in "e1/spiral_pca" "e2/uniform_pca" "e2/arrows" "e3/gaussian_pca" "e4/sphere_pca_radius025" "e4/sphere_pca_radius050" "e4/sphere_pca_radius100" "e4/sphere_pca_radius200" "e5/padded_fmnist_adddim0" "e5/padded_fmnist_adddim4" "e5/padded_fmnist_adddim8" "e5/upscaled_fmnist" "e6/exp_pca" "e7/crescent_moon_pca" "e1/sampled_fmnist_step1" "e1/sampled_fmnist_step2" "e1/sampled_fmnist_step3" "e1/sampled_fmnist_step4" "e1/sampled_fmnist_step5" "e1/sampled_fmnist_step6" "e1/sampled_fmnist_step7" "e1/sampled_fmnist_step8" "e1/sampled_fmnist_step9" "e1/sampled_fmnist_step10" "e1/sampled_fmnist_step11" "e1/sampled_fmnist_step12" 
    do
        # for deltas in "0.00015625;0.0003125;0.000625" #"0.0003125;0.000625;0.00125" # "0.000625;0.00125;0.0025" "0.00125;0.0025;0.005" "0.0025;0.005;0.01" "0.005;0.01;0.02" # "0.01;0.02;0.04" "0.02;0.04;0.08" "0.04;0.08;0.16" "0.08;0.16;0.32" "0.16;0.32;0.64" "0.32;0.64;1.28"
        # for deltas in "0.0039;0.0078;0.0156;0.0312;0.0625;0.1250"
        for deltas in "0.2500;0.5000;1.0000"
        do
            # dataset="${dataset_prefix}-${dataset_num}"
            epochs=500
            srun --time=1-00:00:00 --partition=common --qos=32gpu14d --export=ALL,algorithm=${algorithm},seed=${seed},dataset=${dataset},epochs=${epochs},deltas=${deltas},size=100000 --gres=gpu:1 --job-name="${algorithm}-${dataset}" bash srun-single.sh &
        done
    done

done