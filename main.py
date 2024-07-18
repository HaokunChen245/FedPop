import argparse
import os
import time

import wandb
from tuners import *
from utils import setup_seed
from loaders import init_loaders
from loaders_complex import init_loaders_complex
from loaders_imagenet import init_loaders_tiny, init_loaders_full

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--debug", type=int, default=-1)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        choices=["cpu", "cuda"],
        help="run device (cpu | cuda)",
    )
    parser.add_argument("--algorithm", type=str, default="PBT")
    parser.add_argument(
        "--total_resources", type=int, default=4000, help="Training epochs for server"
    )
    parser.add_argument(
        "--max_resources", type=int, default=800, help="Training epochs for server"
    )
    parser.add_argument(
        "--num_hyp_setting", type=int, default=27, help="Training epochs for server"
    )
    parser.add_argument(
        "--quantile", type=float, default=0.3, help="Random seed used for training"
    )
    parser.add_argument(
        "--seed", nargs="+", type=int, default=-1, help="Random seed used for training"
    )
    parser.add_argument(
        "--discount_factor", type=float, default=0.9, help="Discount factor for computing the mean"
    )
    parser.add_argument(
        "--num_active_users", type=int, default=10, help="Number of Users per round"
    )
    parser.add_argument(
        "--niid",
        type=float,
        help="The strongness of hyperparameter perturbation",
    )
    parser.add_argument(
        "--perturb_eps",
        type=float,
        help="The strongness of hyperparameter perturbation",
    )
    parser.add_argument(
        "--apply_fix_perturb",
        action="store_true",
        help="Whether to apply fix number of perturb",
    )
    parser.add_argument(
        "--sim_user_init",
        action="store_true",
        help="Whether to apply similar user initializations",
    )
    parser.add_argument(
        "--freq_exploit_explore",
        type=int,
        default=5,
        help="Frequency of apply exploration",
    )
    parser.add_argument(
        "--eps_annealing",
        type=str,
        default='None'
    )
    parser.add_argument(
        "--apply_acc_for_selection",
        action="store_true",
    )
    parser.add_argument(
        "--max_rounds_local_perturb",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--update_base_hyp",
        action="store_true",
    )
    parser.add_argument(
        "--use_mp",
        action="store_true",
    )
    parser.add_argument(
        "--use_batch",
        action="store_true",
    )

    parser.add_argument(
        "--freq_exploit_explore_server",
        type=int,
        default=20,
        help="Frequency of apply exploration",
    )

    parser.add_argument(
        "--log_folder",
        type=str,
    )
    parser.add_argument(
        "--resample_prob",
        type=float,
        default=0.1,
        help="Frequency of apply exploration",
    )
    args = parser.parse_args()

    config = {}  # Following FedEx
    config['use_batch'] = args.use_batch
    config["device"] = args.device
    config["dataset"] = args.dataset
    config["algorithm"] = args.algorithm
    config["total_resources"] = args.total_resources
    config["resample_prob"] = args.resample_prob
    config["perturb_eps"] = args.perturb_eps
    config['discount_factor'] = args.discount_factor
    config['eps_annealing'] = args.eps_annealing
    config['apply_fix_perturb'] = args.apply_fix_perturb
    config['update_base_hyp'] = args.update_base_hyp
    config['apply_acc_for_selection'] = args.apply_acc_for_selection
    config['max_rounds_local_perturb'] = args.max_rounds_local_perturb
    config["seed"] = args.seed
    config['niid'] = args.niid
    config['debug'] = args.debug
    config["use_mp"] = args.use_mp
    config['log_folder'] = args.log_folder

    if not os.path.exists(f"/home/cc/fedhp/{config['log_folder']}"):
        os.system(f"mkdir -p /home/cc/fedhp/{config['log_folder']}")

    accs_glob = []
    accs_refine = []
    if args.seed == -1 or -1 in args.seed:
        seed_range = range(2, 5)
    else:
        seed_range = args.seed
    print(seed_range)

    for config["seed"] in seed_range:
        setup_seed(config["seed"])
        config["num_active_users"] = args.num_active_users
        config["sim_user_init"] = args.sim_user_init
        config["freq_exploit_explore"] = args.freq_exploit_explore
        config['freq_exploit_explore_server'] = args.freq_exploit_explore_server

        config["dataset_dir"] = f'/home/cc/datasets/{args.dataset}'
        config["quantile_top"] = args.quantile
        config["quantile_bottom"] = args.quantile
        config_net = {}
        config["num_hyp_setting"] = 27
        # for SHA, following FedEx
        if config["dataset"] in ['CIFAR10', 'shakespeare', 'CelebA', 'CIFAR10_scale']:
            config["max_resources"] = 800
            config["total_resources"] = 4000
            if config["dataset"]=='CIFAR10':
                config['num_users'] = 500
                config_net["backbone"] = "CNN_CIFAR10"
            if config['dataset']=='CIFAR10_scale':
                config["max_resources"] = 1500
                config["total_resources"] = 15000
                config['num_users'] = 500
                config_net["backbone"] = "CNN_CIFAR10"

            elif config["dataset"]=='shakespeare':
                config_net["backbone"] = "CharLSTM"
            elif config["dataset"]=='CelebA':
                config_net["backbone"] = "CNN_CelebA"

        elif config["dataset"]=='FEMNIST':
            config["max_resources"] = 200
            config["total_resources"] = 2000
            config_net["backbone"] = "CNN_FEMNIST"

        elif config["dataset"] in ['PACS', 'DomainNet', 'rxrx1', 'OfficeHome', 'OfficeCaltech']:
            config["num_hyp_setting"] = 9
            config["max_resources"] = 100
            config["total_resources"] = 500
            config_net["backbone"] = "resnet18"

        elif config['dataset'] == 'tinyimagenet':
            if 'default' in config['algorithm']:
                config["num_hyp_setting"] = 1
                config["max_resources"] = 2000
                config["total_resources"] = 2000
            else:
                config["num_hyp_setting"] = 5
                config["max_resources"] = 500
                config["total_resources"] = 2500

            config["num_classes"] = 200
            config["num_users"] = 50
            config_net["backbone"] = "resnet18"

        elif config['dataset'] == 'imagenet':
            config["num_hyp_setting"] = 5
            config["max_resources"] = 1000
            config["total_resources"] = 5000

            config_net["num_classes"] = 1000
            config["num_users"] = 100
            config_net["backbone"] = "resnet18"

        else:
            config_net["backbone"] = "ConvNet"
            config_net["num_classes"] = 10
            config_net["net_width"] = 64
            config_net["net_depth"] = 3
            config_net["net_act"] = "relu"
            config_net["net_norm"] = "instancenorm"
            config_net["net_pooling"] = "maxpooling"
            config_net["img_size"] = 32
            config_net["img_channel"] = 3

        if 'Central' in config['algorithm']:
            if config["dataset"]=='FEMNIST':
                config['total_resources'] = 100
                config["max_resources"] = 20
            elif config["dataset"]=='shakespeare':
                config['total_resources'] = 250
                config["max_resources"] = 50
            else:
                config['total_resources'] = 200
                config["max_resources"] = 40
            config["num_hyp_setting"] = 9

        if config['dataset'] in ['rxrx1', 'PACS', 'DomainNet', 'OfficeHome', 'OfficeCaltech']:
            loaders = init_loaders_complex(config)
            config["num_active_users"] = len(loaders)
            config_net["num_classes"] = config['num_classes']
        elif config['dataset'] in ['tinyimagenet']:
            loaders = init_loaders_tiny(config)
            config_net["num_classes"] = config['num_classes']
        elif config['dataset'] in ['imagenet']:
            loaders = init_loaders_full(config)
        else:
            loaders = init_loaders(config)


        config["net"] = config_net

        config["hps_list"] = [
            "algorithm",
            "dataset",
            "num_users",
            "freq_exploit_explore",
            "resample_prob",
            "perturb_eps",
            "freq_exploit_explore_server",
            "discount_factor",
            "eps_annealing",
            "apply_fix_perturb",
            "update_base_hyp",
            "apply_acc_for_selection",
            "max_rounds_local_perturb",
            "niid",
        ]

        if 'sca' in args.log_folder:
            config['total_resources'] = args.total_resources
            config['max_resources'] = args.max_resources
            config["num_hyp_setting"] = args.num_hyp_setting

        if "RS" in config["algorithm"]:
            config["num_hyp_setting"] = config["total_resources"] // config["max_resources"]

        algo = {
            "PBT": PBT,
            "SHA": SHA,
            "SHA_Single": SHA_Single,
            "SHA_Central": SHA_Central,
            "RS": RS,
            "RS_Single": RS_Single,
            "RS_Single_default": RS_Single,
            "RS_Central": RS_Central,
            "PBT_wSHA_Mix": PBT_wSHA_Mix,
            "PBT_wRS_Mix": PBT_wRS_Mix,
        }[config['algorithm']]
        tuner = algo(config, loaders)

        run_name = f"{config['dataset']}_{config['algorithm']}_" + time.strftime(
            "%Y%m%d_%H%M%S", time.localtime(time.time())
        )

        wandb.init(
            project="Hyperparameter Tuning for FL", group="PBT+SHA", name=run_name
        )

        tuner.train()
        acc_glob, acc_refine = tuner.test()
        acc_glob, acc_refine = 0., 0.

        accs_glob.append(round(acc_glob * 100, 4))
        accs_refine.append(round(acc_refine * 100, 4))

        del tuner

    if len(accs_glob)==1:
        accs_glob = accs_glob + accs_glob
        accs_refine = accs_refine + accs_refine

    import statistics
    print("global:", accs_glob, 'avg:', sum(accs_glob)/len(accs_glob), 'std:', statistics.stdev(accs_glob))
    print("refine:", accs_refine, 'avg:', sum(accs_refine)/len(accs_refine), 'std:', statistics.stdev(accs_refine))

    run_name = f"{config['dataset']}_{config['niid']}_{config['algorithm']}_" + time.strftime(
            "%Y%m%d_%H%M%S", time.localtime(time.time())
    )
    a = ' '.join(str(i) for i in ["global:", accs_glob, 'avg:', sum(accs_glob)/len(accs_glob), 'std:', statistics.stdev(accs_glob)])
    b = ' '.join(str(i) for i in ["refine:", accs_refine, 'avg:', sum(accs_refine)/len(accs_refine), 'std:', statistics.stdev(accs_refine)])
    c = '\n'.join(str(k)+': '+str(v) for k, v in config.items())
    with open(f"/home/cc/fedhp/{config['log_folder']}/{run_name}.txt", "w") as f:
        f.write(a + '\n' + b + '\n' + c)

