import os.path
import random
import torch
import pickle
import cpuinfo
from tqdm import tqdm
import numpy as np
from recommender_system.config import config
from recommender_system.MeLU_SGD import MeLU
from recommender_system.log import log


def training(melu, total_dataset, batch_size, num_epoch, model_save=True, model_filename=None):

    print("Training now...")

    if torch.cuda.is_available():
        melu.cuda()
        print("Model running on {}".format(torch.cuda.get_device_name(0)))
    else:
        print("Model running on {}".format(cpuinfo.get_cpu_info()['brand_raw']))

    if config['use_cuda']:
        melu.cuda()

    training_set_size = len(total_dataset)
    melu.train()
    logger = log()

    for e in range(num_epoch):
        random.shuffle(total_dataset)
        num_batch = int(training_set_size / batch_size)
        a, b, c, d = zip(*total_dataset)

        training_loss_per_epoch = []
        validation_loss_per_epoch = []
        print('---------------------------------')

        for i in tqdm(range(num_batch), colour="green", leave=True, position=0, desc="Epoch [{}/{}]:".format(e + 1, num_epoch)):  # The training is done on a batch basis.
            try:
                supp_xs = list(a[batch_size * i:batch_size * (i + 1)])
                supp_ys = list(b[batch_size * i:batch_size * (i + 1)])
                query_xs = list(c[batch_size * i:batch_size * (i + 1)])
                query_ys = list(d[batch_size * i:batch_size * (i + 1)])
            except IndexError:
                continue
            train_loss_batch, valid_loss_batch = melu.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'])  # Once
            # a batch is formed, it is then passed for the global update
            training_loss_per_epoch.append(train_loss_batch)
            validation_loss_per_epoch.append(valid_loss_batch)

        mean_training_loss = np.mean(training_loss_per_epoch)
        mean_validation_loss = np.mean(validation_loss_per_epoch)
        logger.training_loss.append(mean_training_loss)
        logger.validation_loss.append(mean_validation_loss)

        print("Train Loss: {} Validation Loss: {}".format(mean_training_loss, mean_validation_loss))
        print('---------------------------------')

    # logger.get_training_loss_curve(config['num_epoch'])
    logger.store_on_disk()
    logger.get_length()
    if model_save:
        torch.save(melu.state_dict(), model_filename)


def generate_and_train(dataset_path):

    # training model.
    melu = MeLU(config)
    # print(melu)
    model_filename = "{}/debugModels.pkl".format(dataset_path)
    if not os.path.exists(model_filename):
        # Load training dataset.
        training_set_size = int(len(os.listdir("{}/warm_state".format(dataset_path))) / 4)

        # Training set size is different for each state. If there are 100 users in the warm state category then the
        # warm state folder will have 400 items (100*4), because each user's movie-rating data will be split into 4
        # lists- support X, support Y, query X, Query Y.

        sup_x = []
        sup_y = []
        query_x = []
        query_y = []

        # The support set for each user is different because it depends on the number of movies the user has watched,
        # but the query size is of fixed size 10 for each user.
        print("\nLoading training data..")
        for idx in tqdm(range(training_set_size), colour="blue", position=0):
            sup_x.append(pickle.load(open("{}/warm_state/supp_x_{}.pkl".format(dataset_path, idx), "rb")))
            # supp_xs_s is a matrix of size N x 10242, where N is number of movies the user has watched - 10.
            sup_y.append(pickle.load(open("{}/warm_state/supp_y_{}.pkl".format(dataset_path, idx), "rb")))
            # supp_ys_s is a column matrix of size N-10, where each value is a rating given by the user for a movie
            # in supp_xs_s.
            query_x.append(pickle.load(open("{}/warm_state/query_x_{}.pkl".format(dataset_path, idx), "rb")))
            query_y.append(pickle.load(open("{}/warm_state/query_y_{}.pkl".format(dataset_path, idx), "rb")))
            # the query lists work in the same fashion as support x but the size is a fixed size 10.
        total_dataset = list(zip(sup_x, sup_y, query_x, query_y))

        # The support and query data is collected and zipped together which is later going to be unpacked by the
        # model during training.

        del (sup_x, sup_y, query_x, query_y)
        training(melu, total_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=True,
                 model_filename=model_filename)
    else:
        trained_state_dict = torch.load(model_filename)
        melu.load_state_dict(trained_state_dict)

    # # selecting evidence candidates.
    # evidence_candidate_list = selection(melu, dataset_path, config['num_candidate'])
    # for movie, score in evidence_candidate_list:
    #     print(movie, score)




