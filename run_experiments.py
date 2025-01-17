import argparse
from pathlib import Path
import datasets
from datasets import normalize
from dim_estimators import mle_skl, corr_dim, LIDL, mle_inv
import numpy as np
import neptune.new as neptune
import skdim
import json
import time

inputs = {
    "aldi-sphere-20-21-0.1": lambda size, seed: datasets.sphere_dataset(size, dim=20, ambient_dim=21, radius=0.1, seed=seed),
    "aldi-sphere-20-21-0.2": lambda size, seed: datasets.sphere_dataset(size, dim=20, ambient_dim=21, radius=0.2, seed=seed),
    "aldi-sphere-20-21-0.4": lambda size, seed: datasets.sphere_dataset(size, dim=20, ambient_dim=21, radius=0.4, seed=seed),
    "aldi-sphere-20-21-0.8": lambda size, seed: datasets.sphere_dataset(size, dim=20, ambient_dim=21, radius=0.8, seed=seed),
    "aldi-sphere-20-40-0.1": lambda size, seed: datasets.sphere_dataset(size, dim=20, ambient_dim=40, radius=0.1, seed=seed),
    "aldi-sphere-20-40-0.2": lambda size, seed: datasets.sphere_dataset(size, dim=20, ambient_dim=40, radius=0.2, seed=seed),
    "aldi-sphere-20-40-0.4": lambda size, seed: datasets.sphere_dataset(size, dim=20, ambient_dim=40, radius=0.4, seed=seed),
    "aldi-sphere-20-40-0.8": lambda size, seed: datasets.sphere_dataset(size, dim=20, ambient_dim=40, radius=0.8, seed=seed),
    "aldi-gaussian-20-20-0.1": lambda size, seed: datasets.gaussian_dataset(size, dim=20, ambient_dim=20, std=0.1, seed=seed),
    "aldi-gaussian-20-20-0.2": lambda size, seed: datasets.gaussian_dataset(size, dim=20, ambient_dim=20, std=0.2, seed=seed),
    "aldi-gaussian-20-20-0.4": lambda size, seed: datasets.gaussian_dataset(size, dim=20, ambient_dim=20, std=0.4, seed=seed),
    "aldi-gaussian-20-20-0.8": lambda size, seed: datasets.gaussian_dataset(size, dim=20, ambient_dim=20, std=0.8, seed=seed),
    "aldi-gaussian-20-40-0.01": lambda size, seed: datasets.gaussian_dataset(size, dim=20, ambient_dim=40, std=0.01, seed=seed),
    "aldi-gaussian-20-40-0.05": lambda size, seed: datasets.gaussian_dataset(size, dim=20, ambient_dim=40, std=0.05, seed=seed),
    "aldi-gaussian-20-40-0.1": lambda size, seed: datasets.gaussian_dataset(size, dim=20, ambient_dim=40, std=0.1, seed=seed),
    "aldi-gaussian-20-40-0.2": lambda size, seed: datasets.gaussian_dataset(size, dim=20, ambient_dim=40, std=0.2, seed=seed),
    "aldi-gaussian-20-40-0.4": lambda size, seed: datasets.gaussian_dataset(size, dim=20, ambient_dim=40, std=0.4, seed=seed),
    "aldi-gaussian-20-40-0.8": lambda size, seed: datasets.gaussian_dataset(size, dim=20, ambient_dim=40, std=0.8, seed=seed),
    "aldi-gaussian_saw-1-3-02-11": lambda size, seed: datasets.gaussian_saw_dataset(size, dim=1, ambient_dim=3, std=0.2, n_peaks=11, seed=seed),
    "aldi-gaussian_saw-10-15-0.2-11": lambda size, seed: datasets.gaussian_saw_dataset(size, dim=10, ambient_dim=15, std=0.2, n_peaks=11, seed=seed),
    "aldi-gaussian_saw-10-16-0.2-11": lambda size, seed: datasets.gaussian_saw_dataset(size, dim=10, ambient_dim=16, std=0.2, n_peaks=11, seed=seed),
    "aldi-gaussian_saw-10-17-0.2-11": lambda size, seed: datasets.gaussian_saw_dataset(size, dim=10, ambient_dim=17, std=0.2, n_peaks=11, seed=seed),
    "aldi-gaussian_saw-10-18-0.2-11": lambda size, seed: datasets.gaussian_saw_dataset(size, dim=10, ambient_dim=18, std=0.2, n_peaks=11, seed=seed),
    "aldi-gaussian_saw-10-20-0.2-11": lambda size, seed: datasets.gaussian_saw_dataset(size, dim=10, ambient_dim=20, std=0.2, n_peaks=11, seed=seed),
    "aldi-gaussian_saw-10-30-0.2-11": lambda size, seed: datasets.gaussian_saw_dataset(size, dim=10, ambient_dim=30, std=0.2, n_peaks=11, seed=seed),
    "aldi-gaussian_saw-15-15-0.2-11": lambda size, seed: datasets.gaussian_saw_dataset(size, dim=15, ambient_dim=15, std=0.2, n_peaks=11, seed=seed),
    "aldi-gaussian_saw-15-16-0.2-11": lambda size, seed: datasets.gaussian_saw_dataset(size, dim=15, ambient_dim=16, std=0.2, n_peaks=11, seed=seed),
    "aldi-gaussian_saw-15-17-0.2-11": lambda size, seed: datasets.gaussian_saw_dataset(size, dim=15, ambient_dim=17, std=0.2, n_peaks=11, seed=seed),
    "aldi-gaussian_saw-15-18-0.2-11": lambda size, seed: datasets.gaussian_saw_dataset(size, dim=15, ambient_dim=18, std=0.2, n_peaks=11, seed=seed),
    "aldi-gaussian_saw-15-20-0.2-11": lambda size, seed: datasets.gaussian_saw_dataset(size, dim=15, ambient_dim=20, std=0.2, n_peaks=11, seed=seed),
    "aldi-gaussian_saw-15-30-0.2-11": lambda size, seed: datasets.gaussian_saw_dataset(size, dim=15, ambient_dim=30, std=0.2, n_peaks=11, seed=seed),
    "quantized_uniform-3": lambda root_data_path: datasets.quantized_uniform(root_data_path, 3),
    "quantized_uniform-5": lambda root_data_path: datasets.quantized_uniform(root_data_path, 5),
    "quantized_uniform-9": lambda root_data_path: datasets.quantized_uniform(root_data_path, 9),
    "quantized_uniform-17": lambda root_data_path: datasets.quantized_uniform(root_data_path, 17),
    "quantized_uniform-33": lambda root_data_path: datasets.quantized_uniform(root_data_path, 33),
    "quantized_uniform-65": lambda root_data_path: datasets.quantized_uniform(root_data_path, 65),
    "quantized_uniform-129": lambda root_data_path: datasets.quantized_uniform(root_data_path, 129),
    "quantized_uniform-257": lambda root_data_path: datasets.quantized_uniform(root_data_path, 257),

    "e1/sampled_fmnist_step1": lambda root_data_path: datasets.aldi_generic(root_data_path, "e1_sampled_fmnist_step1"),
    "e1/sampled_fmnist_step2": lambda root_data_path: datasets.aldi_generic(root_data_path, "e1_sampled_fmnist_step2"),
    "e1/sampled_fmnist_step3": lambda root_data_path: datasets.aldi_generic(root_data_path, "e1_sampled_fmnist_step3"),
    "e1/sampled_fmnist_step4": lambda root_data_path: datasets.aldi_generic(root_data_path, "e1_sampled_fmnist_step4"),
    "e1/sampled_fmnist_step5": lambda root_data_path: datasets.aldi_generic(root_data_path, "e1_sampled_fmnist_step5"),
    "e1/sampled_fmnist_step6": lambda root_data_path: datasets.aldi_generic(root_data_path, "e1_sampled_fmnist_step6"),
    "e1/sampled_fmnist_step7": lambda root_data_path: datasets.aldi_generic(root_data_path, "e1_sampled_fmnist_step7"),
    "e1/sampled_fmnist_step8": lambda root_data_path: datasets.aldi_generic(root_data_path, "e1_sampled_fmnist_step8"),
    "e1/sampled_fmnist_step9": lambda root_data_path: datasets.aldi_generic(root_data_path, "e1_sampled_fmnist_step9"),
    "e1/sampled_fmnist_step10": lambda root_data_path: datasets.aldi_generic(root_data_path, "e1_sampled_fmnist_step10"),
    "e1/sampled_fmnist_step11": lambda root_data_path: datasets.aldi_generic(root_data_path, "e1_sampled_fmnist_step11"),
    "e1/sampled_fmnist_step12": lambda root_data_path: datasets.aldi_generic(root_data_path, "e1_sampled_fmnist_step12"),
    "e1/spiral_pca": lambda root_data_path: datasets.aldi_generic(root_data_path, "e1_spiral_pca"),
    "e2/uniform_pca": lambda root_data_path: datasets.aldi_generic(root_data_path, "e2_uniform_pca"),
    "e3/gaussian_pca": lambda root_data_path: datasets.aldi_generic(root_data_path, "e3_gaussian_pca"),
    "e4/sphere_pca_radius025": lambda root_data_path: datasets.aldi_generic(root_data_path, "e4_sphere_pca_radius025"),
    "e4/sphere_pca_radius050": lambda root_data_path: datasets.aldi_generic(root_data_path, "e4_sphere_pca_radius050"),
    "e4/sphere_pca_radius100": lambda root_data_path: datasets.aldi_generic(root_data_path, "e4_sphere_pca_radius100"),
    "e4/sphere_pca_radius200": lambda root_data_path: datasets.aldi_generic(root_data_path, "e4_sphere_pca_radius200"),
    "e6/exp_pca": lambda root_data_path: datasets.aldi_generic(root_data_path, "e6_exp_pca"),
    "e7/crescent_moon_pca": lambda root_data_path: datasets.aldi_generic(root_data_path, "e7_crescent_moon_radius3.0"),

    "e2/arrows": lambda root_data_path: datasets.aldi_generic(root_data_path, "e2_arrows"),
    "e5/padded_fmnist_adddim0": lambda root_data_path: datasets.aldi_generic(root_data_path, "e5_padded_fmnist_adddim0"),
    "e5/padded_fmnist_adddim4": lambda root_data_path: datasets.aldi_generic(root_data_path, "e5_padded_fmnist_adddim4"),
    "e5/padded_fmnist_adddim8": lambda root_data_path: datasets.aldi_generic(root_data_path, "e5_padded_fmnist_adddim8"),
    "e5/upscaled_fmnist": lambda root_data_path: datasets.aldi_generic(root_data_path, "e5_upscaled_fmnist"),
    #  "e4/sphere_pca_radius050" "e4/sphere_pca_radius100" "e4/sphere_pca_radius200"

    "uniform-1": lambda size, seed: datasets.uniform_N(1, size, seed=seed),
    "uniform-10": lambda size, seed: datasets.uniform_N(10, size, seed=seed),
    "uniform-12": lambda size, seed: datasets.uniform_N(12, size, seed=seed),
    "uniform-100": lambda size, seed: datasets.uniform_N(100, size, seed=seed),
    "uniform-500": lambda size, seed: datasets.uniform_N(500, size, seed=seed),
    "uniform-1000": lambda size, seed: datasets.uniform_N(1000, size, seed=seed),
    "uniform-2000": lambda size, seed: datasets.uniform_N(2000, size, seed=seed),
    "uniform-4000": lambda size, seed: datasets.uniform_N(4000, size, seed=seed),
    "uniform-10000": lambda size, seed: datasets.uniform_N(10000, size, seed=seed),
    "uniform_N_0_1-12": lambda size, seed: datasets.uniform_N_0_1(12, size, seed=seed),
    "gaussian-1": lambda size, seed: datasets.gaussian(1, size, seed=seed),
    "gaussian-5": lambda size, seed: datasets.gaussian(5, size, seed=seed),
    "gaussian-10": lambda size, seed: datasets.gaussian(10, size, seed=seed),
    "gaussian-100": lambda size, seed: datasets.gaussian(100, size, seed=seed),
    "gaussian-500": lambda size, seed: datasets.gaussian(500, size, seed=seed),
    "gaussian-1000": lambda size, seed: datasets.gaussian(1000, size, seed=seed),
    "gaussian-2000": lambda size, seed: datasets.gaussian(2000, size, seed=seed),
    "gaussian-4000": lambda size, seed: datasets.gaussian(4000, size, seed=seed),
    "gaussian-10000": lambda size, seed: datasets.gaussian(10000, size, seed=seed),
    "sphere-7": lambda size, seed: datasets.sphere_7(size, seed=seed),
    "uniform-helix-r3": lambda size, seed: datasets.uniform_helix_r3(size, seed=seed),
    "swiss-roll-r3": lambda size, seed: datasets.swiss_roll_r3(size, seed=seed),
    "sin": lambda size, seed: datasets.sin(size, seed=seed),
    "sin-quant": lambda size, seed: datasets.sin_quant(size, seed=seed),
    "sin-dequant": lambda size, seed: datasets.sin_dequant(size, seed=seed),
    "gaussian-1-2": lambda size, seed: datasets.gaussian_N_2N(size, N=1, seed=seed),
    "gaussian-10-20": lambda size, seed: datasets.gaussian_N_2N(size, N=10, seed=seed),
    "gaussian-100-200": lambda size, seed: datasets.gaussian_N_2N(size, N=100, seed=seed),
    "gaussian-500-1000": lambda size, seed: datasets.gaussian_N_2N(size, N=500, seed=seed),
    "gaussian-1000-2000": lambda size, seed: datasets.gaussian_N_2N(size, N=1000, seed=seed),
    "gaussian-2000-4000": lambda size, seed: datasets.gaussian_N_2N(size, N=2000, seed=seed),
    "gaussian-10000-20000": lambda size, seed: datasets.gaussian_N_2N(size, N=10000, seed=seed),
    "lollipop": lambda size, seed: datasets.lollipop_dataset(size, seed=seed),
    "lollipop-0": lambda size, seed: datasets.lollipop_dataset_0(size, seed=seed),
    "lollipop-0-dense-head": lambda size, seed: datasets.lollipop_dataset_0_dense_head(size, seed=seed),
    "sin-01": lambda size, seed: datasets.sin_freq(size, freq=0.1, seed=seed),
    "sin-02": lambda size, seed: datasets.sin_freq(size, freq=0.2, seed=seed),
    "sin-05": lambda size, seed: datasets.sin_freq(size, freq=0.5, seed=seed),
    "sin-10": lambda size, seed: datasets.sin_freq(size, freq=1.0, seed=seed),
    "sin-20": lambda size, seed: datasets.sin_freq(size, freq=2.0, seed=seed),
    "sin-30": lambda size, seed: datasets.sin_freq(size, freq=3.0, seed=seed),
    "sin-50": lambda size, seed: datasets.sin_freq(size, freq=5.0, seed=seed),
    "sin-80": lambda size, seed: datasets.sin_freq(size, freq=8.0, seed=seed),
    "sin-160": lambda size, seed: datasets.sin_freq(size, freq=16.0, seed=seed),
    "sin-320": lambda size, seed: datasets.sin_freq(size, freq=16.0, seed=seed),
    "sin-dens-1": lambda size, seed: datasets.sin_dens(size, freq=1.0, seed=seed),
    "sin-dens-2": lambda size, seed: datasets.sin_dens(size, freq=2.0, seed=seed),
    "sin-dens-3": lambda size, seed: datasets.sin_dens(size, freq=3.0, seed=seed),
    "sin-dens-4": lambda size, seed: datasets.sin_dens(size, freq=4.0, seed=seed),
    "sin-dens-6": lambda size, seed: datasets.sin_dens(size, freq=6.0, seed=seed),
    "sin-dens-8": lambda size, seed: datasets.sin_dens(size, freq=8.0, seed=seed),
    "sin-dens-10": lambda size, seed: datasets.sin_dens(size, freq=10.0, seed=seed),
    "sin-dens-12": lambda size, seed: datasets.sin_dens(size, freq=12.0, seed=seed),
    "sin-dens-14": lambda size, seed: datasets.sin_dens(size, freq=14.0, seed=seed),
    "sin-dens-16": lambda size, seed: datasets.sin_dens(size, freq=16.0, seed=seed),
    "boston": lambda size, seed: datasets.csv_dataset(
        # path to boston_housing dataset here
        "~/datasets/boston_housing.txt"
    ),
    "protein": lambda size, seed: datasets.csv_dataset(
        # path to protein dataset here
        "~/datasets/protein.txt"
    ),
    "wine": lambda size, seed: datasets.csv_dataset(
        # path to wine dataset here
        "~/datasets/wine.txt"
    ),
    "power": lambda size, seed: datasets.csv_dataset(
        # path to power dataset here
        "~/datasets/power.txt"
    ),
    "yacht": lambda size, seed: datasets.csv_dataset(
        # path to yacht dataset here
        "~/datasets/yacht.txt"
    ),
    "concrete": lambda size, seed: datasets.csv_dataset(
        # path to concrete dataset here
        "~/datasets/concrete.txt"
    ),
    "energy": lambda size, seed: datasets.csv_dataset(
        # path to enery_heating_load dataset here
        "~/datasets/energy_heating_load.txt"
    ),
    "kin8nm": lambda size, seed: datasets.csv_dataset(
        # path to kin8nm dataset here
        "~/datasets/kin8nm.txt"
    ),
    "naval": lambda size, seed: datasets.csv_dataset(
        # path to naval_compressor_decay dataset here
        "~/datasets/naval_compressor_decay.txt"
    ),
    "year": lambda size, seed: datasets.csv_dataset(
        # path to year_prediction_msd dataset here
        "~/datasets/year_prediction_msd.txt"
    ),
}

skdim_algorithms = {
    'skdim_corrint': skdim.id.CorrInt,
    'skdim_danco': skdim.id.DANCo,
    'skdim_ess': skdim.id.ESS,
    'skdim_fishers': skdim.id.FisherS,
    'skdim_knn': skdim.id.KNN,
    'skdim_lpca': skdim.id.lPCA,
    'skdim_mada': skdim.id.MADA,
    'skdim_mind_ml': skdim.id.MiND_ML,
    'skdim_mle': skdim.id.MLE,
    'skdim_mom': skdim.id.MOM,
    'skdim_tle': skdim.id.TLE,
    'skdim_twonn': skdim.id.TwoNN,
}

parser = argparse.ArgumentParser(description="LIDL experiments")
parser.add_argument(
    "--algorithm",
    default="mle",
    type=str,
    choices=["mle", "mle-inv", "gm", "rqnsf", "maf",
             "corrdim"] + list(skdim_algorithms.keys()),
    help="name of the algorithm",
)
parser.add_argument(
    "--dataset",
    default="uniform-1",
    type=str,
    choices=list(inputs.keys()),
    help="dataset which id will be estimated",
)
parser.add_argument(
    "--covariance",
    default="diag",
    type=str,
    choices=['spherical', 'tied', 'diag', 'full'],
    help="covariance_type for GaussianMixture",
)
parser.add_argument(
    "--k",
    default="3",
    type=int,
    help="number of neighbours in mle and mle_inv (does nothing with other algorithms)",
)

parser.add_argument(
    "--delta",
    default=None,
    type=float,
    help="delta for density estimator models (does nothing with other algorithms)",
)

parser.add_argument(
    "--num_deltas",
    default=None,
    type=int,
    help="number of deltas for density estimator models (does nothing with other algorithms)",
)

parser.add_argument(
    "--gdim",
    default=False,
    type=bool,
    help="should skdim try to estimate global dim?",
)

parser.add_argument(
    "--neptune_name",
    default=None,
    type=str,
    help="name of the project you want to log to <YOUR_WORKSPACE>/<YOUR_PROJECT>",
)

parser.add_argument(
    "--neptune_token",
    default=None,
    type=str,
    help="token to your project",
)

parser.add_argument(
    "--deltas",
    required=False,
    default=None,
    type=str,
    help="all deltas for density estimator models separated by a comma (does nothing with other algorithms)",
)

parser.add_argument(
    "--ground_truth_const",
    required=False,
    default=None,
    type=int,
    help="if the dimension is constant in every item, you can estimate mse by adding this argument",
)

parser.add_argument(
    "--device",
    default="cpu",
    type=str,
    help="torch device to run the algorithm on (cpu/cuda) - works only for maf and rqnsf",
)

parser.add_argument(
    "--layers",
    default="4",
    type=int,
    help="number of layers in maf/reqnsf"
)

parser.add_argument(
    "--size",
    default="1000",
    type=int,
    help="number of samples in each dataset (number of rows)"
)

parser.add_argument(
    "--seed",
    default="0",
    type=int,
    help="seed for each dataset generator"
)

parser.add_argument(
    "--hidden",
    default="5",
    type=float,
    help="number of hidden features in maf"
)

parser.add_argument(
    "--lr",
    default=0.0001,
    type=float,
    help="learning rate"
)

parser.add_argument(
    "--epochs",
    default=10000,
    type=int,
    help="number of epochs"
)

parser.add_argument(
    "--bs",
    default=256,  # 256,
    type=int,
    help="batch_size"
)

parser.add_argument(
    "--blocks",
    default=5,
    type=int,
    help="number of blocks in rqnsf"
)

parser.add_argument(
    "--json_params",
    default=None,
    type=str,
    help="arguments to skdim"
)

parser.add_argument(
    "--gm_max_components",
    default=200,
    type=int,
    help="number of components in gaussian mixture"
)

args = parser.parse_args()

not_in_filename = [
    'covariance',
    'k',
    'device',
    'layers',
    'size',
    'seed',
    'hidden',
    'lr',
    'epochs',
    'bs',
    'blocks',
    'json_params',
    'gm_max_components',
    'neptune_token',
    'neptune_name',
    'ground_truth_const',
    'gdim']

argname = "_".join([f"{k}:{v}" for k, v in vars(
    args).items() if not k in not_in_filename])

output_dir = Path(f"results/{args.algorithm}/{args.dataset}/{args.deltas}")
output_dir.mkdir(parents=True, exist_ok=True)

report_filename = output_dir / "report.csv"
print(report_filename)

if args.deltas is not None:
    # print(args.deltas)
    ldeltas = args.deltas.split(';')
    deltas = list()
    assert len(ldeltas) >= 2
    for delta in ldeltas:
        fdelta = float(delta)
        assert fdelta > 0
        deltas.append(fdelta)
elif args.delta is not None:
    assert args.delta > 0, "delta must be greater than 0"
    if args.num_deltas is None:
        deltas = [
            args.delta / 2.0,
            args.delta / 1.41,
            args.delta,
            args.delta * 1.41,
            args.delta * 2.0,
        ]
    else:
        deltas = np.geomspace(args.delta/2, args.delta*2, args.num_deltas)
else:
    deltas = [
        0.010000,
        0.013895,
        0.019307,
        0.026827,
        0.037276,
        0.051795,
        0.071969,
        0.100000,
    ]


# data = normalize(data)
# print(args)

run = None
if not (args.neptune_name is None or args.neptune_token is None):
    run = neptune.init(
        project=args.neptune_name,
        api_token=args.neptune_token,
        source_files=['datasets.py', 'dim_estimators.py',
                      'likelihood_estimators.py', 'run_experiments.py', 's3.sh'],
    )
    for key, value in vars(args).items():
        run[key] = value
    starttime = time.time()


f = open(report_filename, "w")


def get_train_val_test(args):
    if args.dataset.startswith('e'):
        data = inputs[args.dataset](
            "/home/pt202342/projects/lid-benchmark-datasets/data/benchmarks_2024-12-09")
        train_dataset = data[0][0]
        val_dataset = data[1][0]
        test_dataset = data[2][0]
    else:
        raise Exception("Support for this dropped.")
        data = inputs[args.dataset](size=args.size, seed=args.seed)
        np.save(output_dir / "data", data)
        assert (args.size == 12000) or (args.size ==
                                        120000), "Other cases are unhandled now, sorry"
        multiplier = 10 if args.size == 120000 else 1
        train_start = 0 * multiplier
        train_end = 10000 * multiplier
        val_start = 10000 * multiplier
        val_end = 11000 * multiplier
        test_start = 11000 * multiplier
        test_end = 12000 * multiplier

        train_dataset = data[train_start:train_end]
        val_dataset = data[val_start:val_end]
        test_dataset = data[test_start:test_end]

    return train_dataset, val_dataset, test_dataset


if args.algorithm in skdim_algorithms:
    print(args.algorithm, file=f)
    if args.json_params is not None:
        with open(args.json_params) as f_skdim_args:
            params = json.load(f_skdim_args)
    else:
        params = dict()
    if not (args.neptune_name is None or args.neptune_token is None):
        run['skdim_params'] = params

    model = skdim_algorithms[args.algorithm](**params)
    ldims = model.fit_transform_pw(data)

    if args.gdim:
        model = skdim_algorithms[args.algorithm](**params)
        gdim = model.fit_transformw(data)

    results = ldims

elif args.algorithm == "gm":
    # TODO fix arguments (convariance)
    gm = LIDL(
        model_type="gm",
        runs=1,
        covariance_type="diag",
        max_components=args.gm_max_components)
    print(f"gm", file=f)
    results = gm(deltas=deltas, train_dataset=data, test=data)
    # gm.save(f"{args.dataset}")

elif args.algorithm == "corrdim":
    print("corrdim", file=f)
    results = corr_dim(data)

elif args.algorithm == "maf":

    train_dataset, val_dataset, test_dataset = get_train_val_test(args)

    maf = LIDL(
        model_type="maf",
        device=args.device,
        num_layers=args.layers,
        lr=args.lr,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.bs)
    print("maf", file=f)
    results = maf(
        deltas=deltas,
        train_dataset=train_dataset,
        val=val_dataset,
        test=test_dataset,
        # verbose=True,
        log_dir=output_dir / "tb",
    )
    # maf.save(f"{args.algorithm}_{args.dataset}")

elif args.algorithm == "rqnsf":
    train_dataset, val_dataset, test_dataset = get_train_val_test(args)

    rqnsf = LIDL(
        model_type="rqnsf",
        device=args.device,
        num_layers=args.layers,
        lr=args.lr,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.bs,
        num_blocks=args.blocks)
    results = rqnsf(
        deltas=deltas,
        train_dataset=train_dataset,
        val=val_dataset,
        test=test_dataset,
        # verbose=True,
        log_dir=output_dir / "tb",
    )
    print("rqnsf", file=f)
    # results = rqnsf.dims_on_deltas(deltas, epoch=best_epochs, total_dim=data.shape[1])
    # rqnsf.save(f"{args.algorithm}_{args.dataset}")

elif args.algorithm == "mle":
    print(f"mle:k={args.k}", file=f)
    # results = mle(data, k=args.k)
    results = mle_skl(data, k=args.k)

elif args.algorithm == "mle-inv":
    print(f"mle-inv:k={args.k}", file=f)
    results = mle_inv(data, k=args.k)


if not (args.neptune_name is None or args.neptune_token is None):
    for lid in results:
        run['lids'].log(lid)
    if args.ground_truth_const is not None:
        def mse(a, b):
            return ((a - b) ** 2).mean()
        mse_val = mse(np.array(results), np.full(
            len(results), args.ground_truth_const))
        run['mse'] = mse_val
    if args.algorithm in skdim_algorithms and args.gdim:
        run['gdim'] = gdim
    # End measurring time
    endtime = time.time()
    run['running_time'] = endtime - starttime
    run.stop()


print("\n".join(map(str, results)), file=f)
