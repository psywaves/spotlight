# -*- coding: utf-8 -*-
import time

import numpy as np
from tqdm.auto import tqdm

from spotlight.distance.implicit import DistanceBasedModel
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.evaluation import precision_recall_score


def data_load(url, skiprows=2):
    with open(url) as f:
        lines = f.readlines()
    item_ids, user_ids, count = [], [], []
    for line in tqdm(lines[skiprows:]):
        iid, uid, c = map(float, line.split())
        item_ids.append(iid-1)
        user_ids.append(uid-1)
        count.append(c)
    return np.array([item_ids, user_ids, count], dtype=np.int32)


def filter_data(data, min_user_interactions=0, min_item_interactions=0, min_rating=0):
    # we filter by the following order: ratings -> items -> users
    data = data.T if type(data) is np.ndarray else np.vstack(data).T

    # filter ratings(count) under minimum value
    if min_rating > 0:
        data = data[data[:, 2] >= min_rating]

    # filter items with less than minimum interactions
    if min_item_interactions > 0:
        iids, count = np.unique(data.T[1], return_counts=True)
        item_set = iids[count >= min_item_interactions]
        data = data[np.isin(data[:, 1], item_set)]

    # filter users with less than minimum interactions
    if min_user_interactions > 0:
        uids, count = np.unique(data.T[0], return_counts=True)
        user_set = uids[count >= min_user_interactions]
        data = data[np.isin(data[:, 0], user_set)]

    data = data.T

    # remap ids
    uid_set = np.unique(data[0])
    iid_set = np.unique(data[1])

    uid_map = {uid: i for i, uid in enumerate(uid_set)}
    iid_map = {iid: i for i, iid in enumerate(iid_set)}

    data[0] = map(lambda x: uid_map[x], data[0])
    data[1] = map(lambda x: iid_map[x], data[1])

    # print stats
    sparsity = (1 - float(data.T.shape[0])/(uid_set.shape[0]*iid_set.shape[0])) * 100
    print "NNZ: {:,} / N_users: {:,} / N_items: {:,}\n(Sparsity: {:.4f}%)".format(
        data.T.shape[0], uid_set.shape[0], iid_set.shape[0], sparsity)

    return data


def distance(train, test, out_dir=None, data_name="empty", repeats=1, verbose=False, **kwargs):
    """
    Run experiment for distance based models (Distance Module)
    """

    precisions, recalls, losses, norms = [], [], [], []

    st = time.time()

    for _ in tqdm(range(repeats)):
        model = DistanceBasedModel(**kwargs)
        embedding_loss, cov_norm = model.fit(train, verbose=verbose, return_loss=True)

        test_precision, test_recall = precision_recall_score(model, test, train, k=50)

        precisions.append(np.mean(test_precision))
        recalls.append(np.mean(test_recall))
        losses.append(embedding_loss)
        norms.append(cov_norm)

    ts = time.time()

    print "*="*40
    print "data: {} with {} repeats".format(data_name, repeats)
    print "Distance Based Model\n", kwargs
    print "Average training time: {:.4f}".format((ts-st)/repeats)
    print("Embedding Loss: {:.4f}, Covariance Norm: {:.4f}\n"
          "Test Precision@50 {:.4f}, Test Recall@50 {:.4f}".format(
              np.mean(losses), np.mean(norms),
              np.mean(precisions), np.mean(recalls)))

    if out_dir is not None:
        with open(out_dir, "a") as f:
            f.write("*="*40 + "\n")
            f.write("data: {} with {} repeats".format(data_name, repeats) + "\n")
            f.write("Distance Based Model\n" + str(kwargs) + "\n")
            f.write("Average training time: {:.4f}".format((ts-st)/repeats) + "\n")
            f.write("Embedding Loss: {:.4f}, Covariance Norm: {:.4f}\n"
                    "Test Precision@50 {:.4f}, Test Recall@50 {:.4f}".format(
                        np.mean(losses), np.mean(norms),
                        np.mean(precisions), np.mean(recalls)) + "\n")


def factorization(train, test, out_dir=None, data_name="empty", repeats=1, verbose=False, **kwargs):
    """
    Run experiment for dot product based models (Factorization Module)
    """

    precisions, recalls = [], []

    st = time.time()

    for _ in tqdm(range(repeats)):
        model = ImplicitFactorizationModel(**kwargs)
        model.fit(train, verbose=verbose)

        test_precision, test_recall = precision_recall_score(model, test, train, k=50)

        precisions.append(np.mean(test_precision))
        recalls.append(np.mean(test_recall))

    ts = time.time()

    print "*="*40
    print "data: {} with {} repeats".format(data_name, repeats)
    print "Dot Product Model\n", kwargs
    print "Average training time: {:.4f}".format((ts-st)/repeats)
    print 'Test Precision@50 {:.4f}, Test Recall@50 {:.4f}'.format(np.mean(precisions), np.mean(recalls))

    if out_dir is not None:
        with open(out_dir, "a") as f:
            f.write("*="*40 + "\n")
            f.write("data: {} with {} repeats".format(data_name, repeats) + "\n")
            f.write("Dot Product Model\n" + str(kwargs) + "\n")
            f.write("Average training time: {:.4f}".format((ts-st)/repeats) + "\n")
            f.write('Test Precision@50 {:.4f}, Test Recall@50 {:.4f}'.format(
                np.mean(precisions), np.mean(recalls)) + "\n")
