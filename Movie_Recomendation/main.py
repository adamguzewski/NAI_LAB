"""
***********************************************
MACHINE LEARNING - Movie_Recomendation
***********************************************
Author: Adam Gużewski

My algorithm is used to recommend the top five movies to watch.
Additionally, it also indicates five movies that should not be watched.

To run program You should type in terminal e.g:
py main.py --user 'Adam Guzewski'

The dataset is stored in movies.json and other_movies_list.json
"""
import argparse
import json
import math

import numpy as np


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find recommended movies for specific user')
    parser.add_argument('--user', dest='user', required=True,
                        help='Input user')
    return parser


def euclidean_distance(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError(user1 + " doesn\'t exist in dataset!")
    if user2 not in dataset:
        raise TypeError(user2 + " doesn\'t exist in dataset!")
    similarity = {}
    for item in dataset[user1]:
        # print(item)
        if item in dataset[user2]:
            # print(item)
            similarity[item] = 1

    # print(similarity)
    # print(len(similarity))

    if len(similarity) == 0:
        return 0
    # print([dataset[user1][item] for item in dataset[user1] if item in dataset[user2]])
    # print([dataset[user2][item] for item in dataset[user2] if item in dataset[user1]])

    sum_euclidean = sum([math.pow(dataset[user1][item] - dataset[user2][item], 2)
                         for item in dataset[user1] if item in dataset[user2]])
    return 1 / (1 + math.sqrt(sum_euclidean))


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    # movies_ratings = 'other_movies_list.json'
    movies_ratings = 'movies.json'

    with open(movies_ratings, 'r') as movies:
        movies_data = json.loads(movies.read())


#
def get_similar_users(dataset, user):
    similarity = [(euclidean_distance(dataset, user, other), other) for other in dataset if user != other]
    similarity.sort()
    similarity.reverse()
    return similarity


# def get_top_similar_users(dataset, user, max_similar_users):
#     similarity = [(euclidean_distance(dataset, user, other), other) for other in dataset if user != other]
#     similarity.sort()
#     similarity.reverse()
#     top_users = similarity[:max_similar_users]
#     return top_users


all_users = get_similar_users(movies_data, user)

not_fitted_list = all_users[7:]
fitted_list = all_users[:7]
# print('not fitted:')
# print(not_fitted_list)
print('Best users to compare and give recommendations:')
print('************************************************')
print(fitted_list)

# Making the list of users with good match
new_list = []

print(new_list)

for el in range(len(not_fitted_list)):
    element = ((not_fitted_list[el])[1])
    new_list.append(element)


def get_user_recommendation(dataset, user):
    totals = {}
    sum_similarity = {}
    for other_user in dataset:
        if other_user == user:
            continue

        similarity = euclidean_distance(dataset, user, other_user)

        if similarity == 0:
            continue

        for item in dataset[other_user]:
            if item not in dataset[user]:
                totals.setdefault(item, 0)
                totals[item] += dataset[other_user][item] * similarity

                sum_similarity.setdefault(item, 0)
                sum_similarity[item] += similarity

    movie_recommendations = [(total / sum_similarity[item], item) for item, total in totals.items()]
    movie_recommendations.sort()
    movie_recommendations.reverse()
    return movie_recommendations


# Removing users with bad match
for item in new_list:
    if item in movies_data:
        del movies_data[item]

# print(movies_data)

all_movies = get_user_recommendation(movies_data, user)

print('Five Movies recommended to watch based on others users grades:')
print(all_movies[:5])
print('Five Movies recommended NOT! to watch based on others users grades:')
print(all_movies[-5:])
