"""
***********************************************
MACHINE LEARNING - Movie_Recomendation
***********************************************
Author: Adam Gużewski

My algorithm is used to recommend the top five movies to watch.
Additionally, it also indicates five movies that should not be watched.

At the end of script program shows short information about recommended movies

To run program You should type in terminal e.g:
py main.py --user 'Adam Guzewski'

The dataset is stored in movies.json and other_movies_list.json
"""
import argparse
import json
import math
import requests


def build_arg_parser():
    """Function to pass parameters
    Parameters:
    user (string): name of user from movies.json
    """
    parser = argparse.ArgumentParser(description='Find recommended movies for specific user')
    parser.add_argument('--user', dest='user', required=True,
                        help='Input user')
    return parser


def euclidean_distance(dataset, user1, user2):
    """
    Function measures the Euclidean distance between user1 and user2

    :param dataset: (dictionary) File which contains grades of movies given from users
    :param user1: (string) name of first user to compare
    :param user2: (string) name of second user to compare

    :return: Value (float) of Euclidean distance between user1 and user2
    """
    # checking if the user is in dataset
    if user1 not in dataset:
        raise TypeError(user1 + " doesn\'t exist in dataset!")
    if user2 not in dataset:
        raise TypeError(user2 + " doesn\'t exist in dataset!")
    # movies rated by both users
    similarity = {}
    for item in dataset[user1]:
        # print(item)
        if item in dataset[user2]:
            # print(item)
            similarity[item] = 1

    # if there are no same movies between users then the similarity is 0
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

    movies_ratings = 'other_movies_list.json'
    # movies_ratings = 'movies.json'

    with open(movies_ratings, 'r') as movies:
        movies_data = json.loads(movies.read())


#
def get_similar_users(dataset, user):
    """
    Function looking for users with similar tastes
    :param dataset: (dictionary) File which contains grades of movies given from users
    :param user: (string) The name of user I am looking for similar users for
    :return: (list) Ordered list of similar users - descending
    """
    similarity = [(euclidean_distance(dataset, user, other), other) for other in dataset if user != other]
    similarity.sort()
    similarity.reverse()
    return similarity


# list of similar users
all_users = get_similar_users(movies_data, user)

# bad users to compare
not_fitted_list = all_users[6:]
# good users to compare
fitted_list = all_users[:6]

print('Best users to compare and give recommendations:')
print(fitted_list)
print('**************************************************************************************************************')

# Making the list of users with good match
users_with_good_match = []

for el in range(len(not_fitted_list)):
    element = ((not_fitted_list[el])[1])
    users_with_good_match.append(element)


def get_user_recommendation(dataset, user):
    """
    Function to find recommended movies based on similar users

    :param dataset: (dictionary) File which contains grades of movies given from users
    :param user: (string) The name of user for who I am looking for recommended movies
    :return: list of tuples which contains recommended movies
    """
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


# Removing from movies_data users with bad match
for item in users_with_good_match:
    if item in movies_data:
        del movies_data[item]

# print(movies_data)

all_movies = get_user_recommendation(movies_data, user)

print('Five Movies recommended to watch based on others users grades:')
print(all_movies[:5])
print('**************************************************************************************************************')
print('Five Movies recommended NOT! to watch based on others users grades:')
print(all_movies[-5:])

# Reading data from imdb service
# Original name, actors and year of production
for el in all_movies[:5]:
    url = "https://imdb8.p.rapidapi.com/auto-complete"

    querystring = {"q": el[1]}

    headers = {
        'x-rapidapi-host': "imdb8.p.rapidapi.com",
        'x-rapidapi-key': "92af6a8ee6msh5c2bd00e6e7e890p1925cdjsn7d763ae36ca6"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    json_data = json.loads(response.text)
    # print(json_data)

    # print(type(json_data))
    original_name = json_data["d"][0]["l"]
    cast = json_data["d"][0]["s"]
    year_of_production = json_data["d"][0]["y"]
    print('********************************************************************')
    print('Information of recommended movie ', el[1], ':')
    print('Original name of the movie: ', original_name)
    print('Short description: ', cast)
    print('Year of production: ', year_of_production)
