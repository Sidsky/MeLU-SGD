import os.path
import numpy as np
import torch
import pickle
from tqdm import tqdm
from recommender_system.config import config
from recommender_system.MeLU_SGD import MeLU
from torch.nn import functional as F


def testing(MeLU_SGD, dataset, dataset_len):
    print("\nTesting now...")
    MeLU_SGD.cuda()
    MeLU_SGD.eval()

    s_x, s_y, q_x, q_y = zip(*dataset)
    loss_all = []

    for i in tqdm(range(len(s_x)), colour="magenta"):

        sup_x = s_x[i].cuda()  # The loss is evaluated on a task basis
        sup_y = s_y[i].cuda()
        query_x = q_x[i].cuda()
        query_y = q_y[i].cuda()

        for idx in range(5):  # Number of local updates
            if idx <= 0:
                pass
            else:
                MeLU_SGD.model.load_state_dict(MeLU_SGD.fast_weights)

            weight_for_local_update = list(MeLU_SGD.model.state_dict().values())

            support_set_y_pred = MeLU_SGD.model(sup_x)
            loss = F.mse_loss(support_set_y_pred, sup_y.view(-1, 1))

            MeLU_SGD.model.zero_grad()
            grad = torch.autograd.grad(loss, MeLU_SGD.model.parameters(), create_graph=True)

            # local update
            for ii in range(MeLU_SGD.weight_len):
                if MeLU_SGD.weight_name[ii] in MeLU_SGD.local_update_target_weight_name:
                    MeLU_SGD.fast_weights[MeLU_SGD.weight_name[ii]] = weight_for_local_update[ii] - MeLU_SGD.local_lr * grad[ii]
                else:
                    MeLU_SGD.fast_weights[MeLU_SGD.weight_name[ii]] = weight_for_local_update[ii]

        query_set_y_pred = MeLU_SGD.model(query_x)
        loss_all.append(F.l1_loss(query_y.view(-1, 1), query_set_y_pred).item())

    return np.mean(loss_all)


def test(dataset_path, state):
    melu_sgd = MeLU(config)
    model_filename = "{}/debugModels.pkl".format(dataset_path)
    trained_state_dict = torch.load(model_filename)
    melu_sgd.load_state_dict(trained_state_dict)

    testing_set_size = int(len(os.listdir("{}/{}".format(dataset_path, state))) / 4)

    supp_x = []
    supp_y = []
    query_x = []
    query_y = []

    print("Loading testing data for {}...".format(state))

    for i in tqdm(range(testing_set_size), colour="green"):
        s_x = pickle.load(open("{}/{}/supp_x_{}.pkl".format(dataset_path, state, i), "rb"))
        s_y = pickle.load(open("{}/{}/supp_y_{}.pkl".format(dataset_path, state, i), "rb"))
        q_x = pickle.load(open("{}/{}/query_x_{}.pkl".format(dataset_path, state, i), "rb"))
        q_y = pickle.load(open("{}/{}/query_y_{}.pkl".format(dataset_path, state, i), "rb"))

        supp_x.append(s_x)
        supp_y.append(s_y)
        query_x.append(q_x)
        query_y.append(q_y)

    test_dataset = list(zip(supp_x, supp_y, query_x, query_y))
    del (supp_x, supp_y, query_x, query_y)

    return testing(melu_sgd, test_dataset, len(test_dataset))
