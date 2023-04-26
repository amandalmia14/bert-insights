import argparse


def get_config_parser():
    parser = argparse.ArgumentParser(description="Run an experiment to reproduced glue")

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--batch_size", type=int, default=32, help="batch size (default: %(default)s)."
    )
    data.add_argument(
        "--dataset", type=str,
        choices=["cola", "mrpc", "rte", "happy_db"],
        default="cola",
        help="name of the dataset to run (default: %(default)s).",
    )
    data.add_argument(
        "--max_len", type=int,
        choices=[32, 64, 96, 128, 256, 512],
        default=96,
        help="length of the token to run (default: %(default)s).",
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model",
        type=str,
        choices=["bert-base-uncased"],
        default="bert-base-uncased",
        help="name of the bert model to run (default: %(default)s).",
    )
    model.add_argument(
        "--model_config",
        type=str,
        default='config/bert_default.json',
        help="Path to model config json file"
    )

    model.add_argument(
        "--layer_config",
        type=str,
        choices=["config/layer_config1.json", "config/layer_config2.json", "config/layer_config3.json", "config/layer_config4.json"],
        default='config/layer_config1.json',
        help="Path to layers config json file"
    )

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of epochs for training (default: %(default)s).",
    )
    # optimization.add_argument(
    #     "--optimizer",
    #     type=str,
    #     default="adamw",
    #     choices=["sgd", "momentum", "adam", "adamw"],
    #     help="choice of optimizer (default: %(default)s).",
    # )
    # optimization.add_argument(
    #     "--lr",
    #     type=float,
    #     default=1e-3,
    #     help="learning rate for Adam optimizer (default: %(default)s).",
    # )
    # optimization.add_argument(
    #     "--momentum",
    #     type=float,
    #     default=0.9,
    #     help="momentum for SGD optimizer (default: %(default)s).",
    # )
    # optimization.add_argument(
    #     "--weight_decay",
    #     type=float,
    #     default=5e-4,
    #     help="weight decay (default: %(default)s).",
    # )

    exp = parser.add_argument_group("Experiment config")
    exp.add_argument(
        "--logdir",
        type=str,
        default='logdir',
        help="unique experiment identifier (default: %(default)s).",
    )
    exp.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for repeatability (default: %(default)s).",
    )

    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="device to store tensors on (default: %(default)s).",
    )
    misc.add_argument(
        "--print_every",
        type=int,
        default=80,
        help="number of minibatches after which to print loss (default: %(default)s).",
    )
    # misc.add_argument(
    #     "--visualize",
    #     action='store_true',
    #     help='A flag to visualize the filters or MLP layer at the end'
    # )
    return parser
