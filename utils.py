import time
import pickle
import os
master_path = "./ml"

def load_dataset(state):
    training_set_size = int(len(os.listdir("{}/{}".format(master_path, state))) / 4)
    supp_xs_s = []
    supp_ys_s = []
    query_xs_s = []
    query_ys_s = []
    
    begin_time = time.time()
    for idx in range(training_set_size):
        supp_xs_s.append(pickle.load(open("{}/{}/supp_x_{}.pkl".format(master_path, state, idx), "rb")))
        supp_ys_s.append(pickle.load(open("{}/{}/supp_y_{}.pkl".format(master_path, state, idx), "rb")))
        query_xs_s.append(pickle.load(open("{}/{}/query_x_{}.pkl".format(master_path, state, idx), "rb")))
        query_ys_s.append(pickle.load(open("{}/{}/query_y_{}.pkl".format(master_path, state, idx), "rb")))
    total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
    del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)
    end_time = time.time()
    print("Time taken to load training dataset: {} seconds".format(end_time - begin_time))
    return total_dataset