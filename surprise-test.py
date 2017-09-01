import sys
from collections import defaultdict
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset, evaluate, GridSearch, dump
from surprise.dataset import Reader
import pandas as pd
import os
import pickle
from surprise import accuracy

# df = pd.read_csv('~/Documents/machine_learning_data_2017_05_25/votes.csv', engine='python')


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append([iid, est])

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        # top_n[uid] = user_ratings[:n]
        top_n[uid] = user_ratings

    return top_n


def get_best_params(data):
    param_grid = {
        'n_epochs': [5, 10],
        'lr_all': [0.002, 0.005],
        'reg_all': [0.4, 0.6]
    }
    grid_search = GridSearch(SVD, param_grid=param_grid)
    data.split(n_folds=3)
    grid_search.evaluate(data)
    best_param = grid_search.best_params['RMSE']

    return best_param

def get_predictions(data):
    trainset = data.build_full_trainset()
    best_params = get_best_params(data)
    algo = SVD(reg_all=best_params['reg_all'], lr_all=best_params['lr_all'], n_epochs=best_params['n_epochs'])
    algo.train(trainset)

    # Than predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)

    return predictions

def print_my_recommends(top_n):
    for uid, user_ratings in top_n.items():
        if uid == '9274':
            lists = [iid for (iid, _) in user_ratings]
            print(uid, lists[:10])
            print(user_ratings[:10])

def get_data_from_file(path):
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
    data = Dataset.load_from_file(path, reader=reader)

    return data

def get_duplicated_posts(top_1, top_2):
    clip_posts = get_recommend_post_in_top_n(top_1)
    voted_posts = get_recommend_post_in_top_n(top_2)

    return [x for x in clip_posts if x in voted_posts]

def get_recommend_post_in_top_n(top_n):
    for uid, user_ratings in top_n.items():
        if uid == '9274':
            return [iid for (iid, _) in user_ratings]

# First train an SVD algorithm on the movielens dataset.0
# data = Dataset.load_builtin('ml-100k')

# path to dataset file
# file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
def recommend(n, path, predictions_file):
    try:
        predictions = dump.load(file_name=predictions_file)[0]
    except IOError:
        print("Error: File does not appear to exist.")
        data = get_data_from_file(path)
        predictions = get_predictions(data)
        # dump.dump(file_name=predictions_file, predictions=predictions)

    top_n = get_top_n(predictions, n=n)
    # Print the recommended items for each user
    # print_my_recommends(top_n)

    return top_n

def get_top_new_posts(top_n_clip, top_n_vote):
    for uid, item_ratings_based_on_clip in top_n_clip.items():
        for index, value in enumerate(item_ratings_based_on_clip):
            iid = value[0]
            if top_n_vote[uid]:
                item_ratings_based_on_vote = top_n_vote[uid]
                for item_id, item_rating_based_on_vote in item_ratings_based_on_vote:
                    if item_id == iid and item_rating_based_on_vote > 1:
                        print('item_id', item_id)
                        print('item_rating_based_on_vote', item_rating_based_on_vote)
                        top_n_clip[uid][index][1] += 0.5 * item_rating_based_on_vote
                        print('top_n_clip[uid][index][1]', top_n_clip[uid][index][1])

    return top_n_vote

def save_top_posts_to_file(data, file_path):
    with open(file_path, "w") as f:
        for s in data:
            f.write(str(s) + "\n")

def read_top_posts_from_file(file_path):
    data = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                data.append(int(line.strip()))
    except IOError:
        print("Error: File does not appear to exist.")

    return data

# evaluate(algo, data)
def execute():
    file_path = 'top_n_all.dat'
    top_new_posts = read_top_posts_from_file(file_path)
    if top_new_posts:
        pass
    else:
        top_amount = 10
        top_n_clip = recommend(n=top_amount, path='~/Documents/machine_learning_data_2017_05_25/clips.csv', predictions_file='top_n_clip')
        top_n_vote = recommend(n=top_amount, path='~/Documents/machine_learning_data_2017_05_25/votes.csv', predictions_file='top_n_vote')

        # duplicated_posts = get_duplicated_posts(top_n_clip, top_n_vote)
        print('type of top_n_vote: ', type(top_n_vote))
        top_new_posts = get_top_new_posts(top_n_clip, top_n_vote)
        # save_top_posts_to_file(top_new_posts, file_path)

    print('top n vote after combine with clip points: ', print_my_recommends(top_new_posts))
    # print('Posts exist in 2 list: ', duplicated_posts)

execute()