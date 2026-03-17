
import sys
import os
from sconf import Config

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from trainer.build import get_tokenizers, build_dataset
from utils import colorstr


def load_config(config_path):
    if config_path is not None:
        config = Config(config_path)
    else:
        config = Config(default="config/config.yaml")
    return config


def main():

    config = load_config(None)
    tokenizer = get_tokenizers(config)

    mode = "train"
    datasets = build_dataset(config, tokenizer, [mode])
    dataset = datasets[mode]

    #collate_fn=getattr(dataset, 'collate_fn', None)

    if len(dataset) > 0:
        item = dataset[0]


if __name__ == '__main__':
    main()

