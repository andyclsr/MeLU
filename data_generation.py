import re
import os
import json
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm

from torch.utils.data import Dataset
from options import states
from dataset import movielens_1m


def item_converting(row, rate_list, genre_list, director_list, actor_list):
    rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()
    genre_idx = torch.zeros(1, 25).long()
    for genre in str(row['genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1
    director_idx = torch.zeros(1, 2186).long()
    for director in str(row['director']).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, idx] = 1
    actor_idx = torch.zeros(1, 8030).long()
    for actor in str(row['actors']).split(", "):
        idx = actor_list.index(actor)
        actor_idx[0, idx] = 1
    return torch.cat((rate_idx, genre_idx, director_idx, actor_idx), 1)


def user_converting(row, gender_list, age_list, occupation_list, zipcode_list): 
    gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
    zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()
    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)


def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_


def generate(master_path):
    dataset_path = "movielens/ml-1m"
    rate_list = load_list("{}/m_rate.txt".format(dataset_path))
    genre_list = load_list("{}/m_genre.txt".format(dataset_path))
    actor_list = load_list("{}/m_actor.txt".format(dataset_path))
    director_list = load_list("{}/m_director.txt".format(dataset_path))
    gender_list = load_list("{}/m_gender.txt".format(dataset_path))
    age_list = load_list("{}/m_age.txt".format(dataset_path))
    occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
    zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))

    if not os.path.exists("{}/warm_state/".format(master_path)):
        for state in states:
            os.mkdir("{}/{}/".format(master_path, state))
    if not os.path.exists("{}/log/".format(master_path)):
        os.mkdir("{}/log/".format(master_path))

    dataset = movielens_1m()

    # hashmap for item information
    if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
        movie_dict = {}
        for idx, row in dataset.item_data.iterrows():
            m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
            movie_dict[row['movie_id']] = m_info
        pickle.dump(movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
    else:
        movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))
    # hashmap for user profile
    if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
        user_dict = {}
        for idx, row in dataset.user_data.iterrows():
            u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
            user_dict[row['user_id']] = u_info
        pickle.dump(user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
    else:
        user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))

    for state in states:
        idx = 0
        if not os.path.exists("{}/{}/{}".format(master_path, "log", state)):
            os.mkdir("{}/{}/{}".format(master_path, "log", state))
        with open("{}/{}.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset_y = json.loads(f.read())
        for _, user_id in tqdm(enumerate(dataset.keys())):
            u_id = int(user_id)
            seen_movie_len = len(dataset[str(u_id)])
            indices = list(range(seen_movie_len))

            if seen_movie_len < 13 or seen_movie_len > 100:
                continue

            random.shuffle(indices)
            tmp_x = np.array(dataset[str(u_id)])
            tmp_y = np.array(dataset_y[str(u_id)])

            support_x_app = None
            for m_id in tmp_x[indices[:-10]]:
                m_id = int(m_id)
                tmp_x_converted = torch.cat((movie_dict[m_id], user_dict[u_id]), 1)
                try:
                    support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                except:
                    support_x_app = tmp_x_converted

            query_x_app = None
            for m_id in tmp_x[indices[-10:]]:
                m_id = int(m_id)
                u_id = int(user_id)
                tmp_x_converted = torch.cat((movie_dict[m_id], user_dict[u_id]), 1)
                try:
                    query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
                except:
                    query_x_app = tmp_x_converted
            support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])
            query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])

            pickle.dump(support_x_app, open("{}/{}/supp_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(support_y_app, open("{}/{}/supp_y_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(master_path, state, idx), "wb"))
            with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[:-10]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            with open("{}/log/{}/query_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[-10:]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            idx += 1


class Metamovie(Dataset):
    def __init__(self, args, partition='train', test_way=None, path=None):
        super(Metamovie, self).__init__()
        #self.dataset_path = args.data_root
        self.partition = partition
        #self.pretrain = pretrain
        
        self.dataset_path = args.data_root
        dataset_path = self.dataset_path
        rate_list = load_list("{}/m_rate.txt".format(dataset_path))
        genre_list = load_list("{}/m_genre.txt".format(dataset_path))
        actor_list = load_list("{}/m_actor.txt".format(dataset_path))
        director_list = load_list("{}/m_director.txt".format(dataset_path))
        gender_list = load_list("{}/m_gender.txt".format(dataset_path))
        age_list = load_list("{}/m_age.txt".format(dataset_path))
        occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
        zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))

        self.dataset = movielens_1m()
        
        master_path = self.dataset_path
        if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
            self.movie_dict = {}
            for idx, row in self.dataset.item_data.iterrows():
                m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
                self.movie_dict[row['movie_id']] = m_info
            pickle.dump(self.movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
        else:
            self.movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))
        # hashmap for user profile
        if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
            self.user_dict = {}
            for idx, row in self.dataset.user_data.iterrows():
                u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
                self.user_dict[row['user_id']] = u_info
            pickle.dump(self.user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
        else:
            self.user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))
        if partition == 'train' or partition == 'valid':
            self.state = 'warm_state'
        else:
            if test_way is not None:
                if test_way == 'old':
                    self.state = 'warm_state'
                elif test_way == 'new_user':
                    self.state = 'user_cold_state'
                elif test_way == 'new_item':
                    self.state = 'item_cold_state'
                else:
                    self.state = 'user_and_item_cold_state'
        print(self.state)
        with open("{}/{}.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split_y = json.loads(f.read())            
        length = len(self.dataset_split.keys())
        self.final_index = []
        for _, user_id in tqdm(enumerate(list(self.dataset_split.keys()))):
            u_id = int(user_id)
            seen_movie_len = len(self.dataset_split[str(u_id)])

            if seen_movie_len < 13 or seen_movie_len > 100:
                continue
            else:
                self.final_index.append(user_id)
         

    def __getitem__(self, item):
        user_id = self.final_index[item]
        u_id = int(user_id)
        seen_movie_len = len(self.dataset_split[str(u_id)])
        indices = list(range(seen_movie_len))
        random.shuffle(indices)
        tmp_x = np.array(self.dataset_split[str(u_id)])
        tmp_y = np.array(self.dataset_split_y[str(u_id)])
        
        support_x_app = None
        for m_id in tmp_x[indices[:-10]]:
            m_id = int(m_id)
            tmp_x_converted = torch.cat((self.movie_dict[m_id], self.user_dict[u_id]), 1)
            try:
                support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
            except:
                support_x_app = tmp_x_converted
        query_x_app = None
        for m_id in tmp_x[indices[-10:]]:
            m_id = int(m_id)
            u_id = int(user_id)
            tmp_x_converted = torch.cat((self.movie_dict[m_id], self.user_dict[u_id]), 1)
            try:
                query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
            except:
                query_x_app = tmp_x_converted
        support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])
        query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])
        return support_x_app, support_y_app.view(-1,1), query_x_app, query_y_app.view(-1,1)
        
    def __len__(self):
        return len(self.final_index)