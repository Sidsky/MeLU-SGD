import datetime
import re
import os
import json
import torch
import random
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd


def load():
    path = "data_generation/raw_data/dat_files"

    userPath = "{}/users.dat".format(path)
    scorePath = "{}/ratings.dat".format(path)
    itemPath = "{}/movies_extrainfos.dat".format(path)

    profile_data = pd.read_csv(
        userPath,
        names=['user_id', 'gender', 'age', 'occupation_code', 'zip'],
        sep="::",
        engine='python'
    )
    item_data = pd.read_csv(
        itemPath,
        names=['movie_id', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot',
               'poster'],
        sep="::",
        engine='python',
        encoding="utf-8"
    )
    score_data = pd.read_csv(
        scorePath,
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        sep="::",
        engine='python'
    )

    score_data['time'] = score_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
    score_data = score_data.drop(["timestamp"], axis=1)

    return profile_data, item_data, score_data


class movieLens(object):
    def __init__(self):
        self.users, self.items, self.score = load()


def user_converting(row, gender_list, age_list, occupation_list, zipcode_list):
    gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
    zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()
    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)


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


def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_


def getData():
    raw_data_path = "data_generation/raw_data"
    rate_list = load_list("{}/rate.txt".format(raw_data_path))
    genre_list = load_list("{}/genre.txt".format(raw_data_path))
    actor_list = load_list("{}/actor.txt".format(raw_data_path))
    director_list = load_list("{}/director.txt".format(raw_data_path))
    gender_list = load_list("{}/gender.txt".format(raw_data_path))
    age_list = load_list("{}/age.txt".format(raw_data_path))
    occupation_list = load_list("{}/occupation.txt".format(raw_data_path))
    zipcode_list = load_list("{}/zipcode.txt".format(raw_data_path))

    data_path = "data_generation/movie_lens_dataset"
    states = ["warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"]

    if not os.path.exists(data_path):
        os.mkdir(data_path)

        movie_lens = movieLens()
        pickle.dump(movie_lens, open("{}/movie_lens.pkl".format(data_path), "wb"))

        os.mkdir("{}/log/".format(data_path))

        for state in states:
            os.mkdir("{}/{}/".format(data_path, state))

        movies = {}
        users = {}

        for idx, row in movie_lens.items.iterrows():
            m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
            movies[row['movie_id']] = m_info

        for idx, row in movie_lens.users.iterrows():
            u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
            users[row['user_id']] = u_info

        pickle.dump(movies, open("{}/movie_dict.pkl".format(data_path), "wb"))
        pickle.dump(users, open("{}/user_dict.pkl".format(data_path), "wb"))

        for state in states:
            i = 0
            os.mkdir("{}/{}/{}".format(data_path, "log", state))

            with open("{}/json_files/{}.json".format(raw_data_path, state), encoding="utf-8") as f:
                dataset = json.loads(f.read())
            with open("{}/json_files/{}_y.json".format(raw_data_path, state), encoding="utf-8") as f:
                dataset_y = json.loads(f.read())

            for _, user_id in tqdm(enumerate(dataset.keys()), colour="magenta"):
                u_id = int(user_id)
                num_movies_seen = len(dataset[str(u_id)])
                indices = list(range(num_movies_seen))

                if num_movies_seen < 13 or num_movies_seen > 100:
                    continue

                random.shuffle(indices)
                tmp_x = np.array(dataset[str(u_id)])
                tmp_y = np.array(dataset_y[str(u_id)])

                support_x_app = None
                for m_id in tmp_x[indices[:-10]]:
                    m_id = int(m_id)
                    tmp_x_converted = torch.cat((movies[m_id], users[u_id]), 1)
                    try:
                        support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                    except:
                        support_x_app = tmp_x_converted

                query_x_app = None

                for m_id in tmp_x[indices[-10:]]:
                    m_id = int(m_id)
                    u_id = int(user_id)
                    tmp_x_converted = torch.cat((movies[m_id], users[u_id]), 1)
                    try:
                        query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
                    except:
                        query_x_app = tmp_x_converted

                support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])
                query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])

                pickle.dump(support_x_app, open("{}/{}/supp_x_{}.pkl".format(data_path, state, i), "wb"))
                pickle.dump(support_y_app, open("{}/{}/supp_y_{}.pkl".format(data_path, state, i), "wb"))
                pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(data_path, state, i), "wb"))
                pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(data_path, state, i), "wb"))

                with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(data_path, state, i), "w") as f:
                    for m_id in tmp_x[indices[:-10]]:
                        f.write("{}\t{}\n".format(u_id, m_id))

                with open("{}/log/{}/query_x_{}_u_m_ids.txt".format(data_path, state, i), "w") as f:
                    for m_id in tmp_x[indices[-10:]]:
                        f.write("{}\t{}\n".format(u_id, m_id))

                i += 1

    movie_data = pickle.load(open("{}/movie_lens.pkl".format(data_path), "rb"))
    return movie_data
