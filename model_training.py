import os
import torch
import pickle
import random
import time
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_generation import Metamovie
from utils import load_dataset

from MeLU import MeLU
from options import config, states

from tqdm import tqdm
from evidence_candidate import selection

def trans_time(ti):
    return time.strftime("%d-%H-%M", time.localtime(ti))

def evaluate_test(args, model, test_dataset):
    model.eval()
    testing_set_size = len(test_dataset)
    random.shuffle(test_dataset)
    a,b,c,d = zip(*test_dataset)
    assert len(a) == testing_set_size
    loss_all = []
    for i in range(testing_set_size):
        sset_x = a[i].unsqueeze(0)
        sset_y = b[i].unsqueeze(0)
        qset_x = c[i].unsqueeze(0)
        qset_y = d[i].unsqueeze(0)
        loss_list = model.global_update(sset_x, sset_y, qset_x, qset_y, config['inner'], False)
        loss_all.append(loss_list[-1])
    loss_all = np.array(loss_all)
    print('{}+/-{}'.format(np.mean(loss_all), 1.96*np.std(loss_all,0)/np.sqrt(len(loss_all))))
    
    
def training(args, melu, total_dataset, batch_size, num_epoch, model_save=True, model_filename=None):
    if config['use_cuda']:
        melu.cuda()

    training_set_size = len(total_dataset)
    melu.train()
    
    num_epoch = 30
    
    for epoch in tqdm(range(num_epoch)):
        begin_time = time.time()
        random.shuffle(total_dataset)
        num_batch = int(training_set_size / batch_size)
        a,b,c,d = zip(*total_dataset)
        for i in range(num_batch):
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])
            except IndexError:
                continue
            melu.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'])

    if model_save:
        torch.save(melu.state_dict(), f"./save/model_{args.name}_{trans_time(time.time())}.pkl")
        
    test_dataset = load_dataset(args.test_state)
    evaluate_test(args, melu, test_dataset)
    # begin_time = time.time()
    # evidence_candidate_list = selection(melu, "./ml", config['num_candidate'])
    # end_time = time.time()
    # print("Time taken for evaluating at epoch {}: {:.2f} seconds".format(_, end_time - begin_time))
    # for movie, score in evidence_candidate_list:
    #     print(movie, score)
                
    
