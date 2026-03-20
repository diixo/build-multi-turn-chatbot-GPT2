import os
import sys
from sconf import Config
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from trainer import Trainer
from utils import LOGGER, colorstr
from utils.training_utils import choose_proper_resume_model



def env_setup():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config_dir(config_dir):
    if config_dir is not None:
        config_path = os.path.join(args.resume_model_dir, 'args.yaml')
    else:
        def_config = Config("config/project.yaml")
        config_dir = os.path.join(def_config.project, f"{def_config.name}-{def_config.epochs}-{def_config.batch_size}-{def_config.max_len}")
        config_path = os.path.join(config_dir, 'args.yaml')

        if not os.path.isfile(config_path):
            LOGGER.info(f'Extract reference config for: (project:{def_config.project}) (name:{def_config.name})')
            raise FileNotFoundError(f"Config file not found: {config_path}")

    config = Config(config_path)

    return config


def main(args):

    # init config
    config = load_config_dir(args.resume_model_dir)
    
    # init environment
    env_setup()

    # chatting 
    chatting(args, config)

    
def chatting(args, config):
    if config.device == 'mps':
        LOGGER.warning(colorstr('yellow', 'cpu is automatically selected because mps leads inaccurate validation.'))
        device = torch.device('cpu')
    else:
        device = torch.device('cpu') if config.device == 'cpu' else torch.device(f'cuda:{config.device[0]}')

    trainer = Trainer(
        config, 
        'validation', 
        device, 
        resume_path=choose_proper_resume_model(args.resume_model_dir, args.load_model_type) if args.resume_model_dir else None
    )
    
    LOGGER.info(colorstr('Chatbot starts...\n'))
    query_done = False
    is_first_query = True
    while 1:
        query = input('Q: ')
        if query == 'exit':
            break
        query, answer, query_done, is_first_query = trainer.chatting(query, is_first_query)
        LOGGER.info('A: ' + answer)

    LOGGER.info(colorstr('Chatbot ends...\n'))



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', '--resume_model_dir', type=str, required=False)
    parser.add_argument('-l', '--load_model_type', type=str, default='metric', required=False, choices=['loss', 'last', 'metric'])
    args = parser.parse_args()

    main(args)
