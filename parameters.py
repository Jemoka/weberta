import argparse

parser = argparse.ArgumentParser(prog='adventure')

# logistics
parser.add_argument("experiment", help="name for the experiment", type=str)
parser.add_argument('-v', '--verbose', action='count', default=0, help="log level")
parser.add_argument("--wandb", default=False, action="store_true", help="whether to use wandb")
parser.add_argument("--warm_start", default=None, type=str, help="recover trainer from this path")

# intervals
parser.add_argument("--report_interval", default=32, type=int, help="save to wandb every this many steps")
parser.add_argument("--checkpoint_interval", default=256, type=int, help="checkpoint every this many steps")
parser.add_argument("--validation_interval", default=256, type=int, help="validate every this many steps")

# dataset
parser.add_argument("--out_dir", help="directory to save checkpoints and outputs", type=str, default="output")
parser.add_argument("--dataset", help="what dataset", type=str, default="dlwh/wikitext_103_detokenized")
parser.add_argument("--mlm_probability", help="how much to mask", type=float, default=0.15)

# models
parser.add_argument("--base", help="what is the base model type we are training (has to be an MLM)", type=str, default="FacebookAI/roberta-base")

# hyperparameters
parser.add_argument("--lr", help="learning rate", type=float, default=6e-4)
parser.add_argument("--epochs", help="number of epochs to train", type=int, default=1000000)
parser.add_argument("--batch_size", help="batches per device", type=int, default=32)
parser.add_argument("--warmup_pct", help="how much to warm up", type=float, default=0.01)

