import os
import torch
import pickle
import time
import argparse

from MeLU import MeLU
from options import config
from model_training import training, evaluate_test
from data_generation import generate
from evidence_candidate import selection
from utils import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--load', action='store_true', help='load', default=False)
parser.add_argument('--name', type=str, help='name', default="")
parser.add_argument('--test_state', type=str, help='test_state', default="warm_state")
parser.add_argument('--mname', type=str, help='mname', default="")
parser.add_argument('--num_grad_steps_inner', type=int, help='num_grad_steps_inner', default=5)
args = parser.parse_args()

if __name__ == "__main__":
    master_path= "./ml"
    if not os.path.exists("{}/".format(master_path)):
        os.mkdir("{}/".format(master_path))
        # preparing dataset. It needs about 22GB of your hard disk space.
        generate(master_path)

    # training model.
    melu = MeLU(config)
    model_filename = "./save/{}.pkl".format(args.mname)
    if (not os.path.exists(model_filename)) and args.load == 0:
        # Load training dataset.
        total_dataset = load_dataset("warm_state")
        training(args, melu, total_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=True, model_filename=model_filename)
    else:
        trained_state_dict = torch.load(model_filename)
        melu.load_state_dict(trained_state_dict)
        melu.store_parameters()

    # testing model.
    evaluate_test(args, melu, load_dataset(args.test_state))
    # selecting evidence candidates.
    print("Selecting evidence candidates...")
    evidence_candidate_list = selection(melu, master_path, config['num_candidate'])
    for movie, score in evidence_candidate_list:
        print(movie, score)
