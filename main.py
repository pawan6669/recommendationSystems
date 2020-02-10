### This file is used to import some items in a dataframe and genarate the vectors for the items.

import pandas as pd
import logging
import time
import numpy as np
from sklearn.cluster import KMeans

#Libraries to show data plot
import matplotlib.pyplot as plt
import seaborn as sns


##Replace this file with any file to parse from
#ITEMS_FILENAME='file://localhost/Users/pawan/PycharmProjects/recommendationsystems/test1.csv'
ITEMS_FILENAME='file://localhost/Users/pawan/PycharmProjects/recommendationsystems/items_genre_new.csv'
#ITEMS_VECTOR_CSV='file://localhost/Users/pawan/PycharmProjects/recommendationsystems/items_vector.csv'
ITEMS_VECTOR_CSV='//Users/pawan/PycharmProjects/recommendationsystems/items_vector_norm_v2.csv'
select_colmns = ['ITEM_ID','GENRE']

#Below set is received from below query on SearchRiverDB in mongoDB
# ﻿db.titles.distinct("Genre.genreName")
item_genre_list= frozenset([
    "Crime",
    "Soap",
    "Thriller",
    "Anthology",
    "Drama",
    "Action",
    "Adventure",
    "Crime drama",
    "Romance",
    "Historical drama",
    "History",
    "Miniseries",
    "War",
    "Mystery",
    "News",
    "Entertainment",
    "Children",
    "Comedy",
    "Anime",
    "Fantasy",
    "Football",
    "Sports talk",
    "Documentary",
    "Adults only",
    "Erotic",
    "Gaming",
    "Religious",
    "Martial arts",
    "Docudrama",
    "Fashion",
    "Western",
    "Musical comedy",
    "Horror",
    "Animated",
    "Comedy drama",
    "Science fiction",
    "Variety",
    "Sitcom",
    "Politics",
    "Public affairs",
    "Community",
    "Animals",
    "Art",
    "Travel",
    "Shopping",
    "Auction",
    "Motorcycle",
    "Bobsled",
    "Gymnastics",
    "Musical",
    "Tennis",
    "Holiday",
    "Game show",
    "Romantic comedy",
    "Dark comedy",
    "Music",
    "Biography",
    "Reality",
    "Paranormal",
    "Law",
    "Medical",
    "Educational",
    "Military",
    "Interview",
    "Talk",
    "Wrestling",
    "Nature",
    "Cooking",
    "Dance",
    "Fishing",
    "Outdoors",
    "Science",
    "Technology",
    "Gay/lesbian",
    "Collectibles",
    "Health",
    "Auto racing",
    "Action sports",
    "Horse",
    "Newsmagazine",
    "Soccer",
    "Aviation",
    "Auto",
    "Home improvement",
    "Arts/crafts",
    "House/garden",
    "How-to",
    "Standup",
    "Mixed martial arts",
    "Environment",
    "Pro wrestling",
    "Bus./financial",
    "Self improvement",
    "Bullfighting",
    "Weather",
    "Parenting",
    "Playoff sports",
    "Bodybuilding",
    "Exercise",
    "Golf",
    "Debate",
    "Archery",
    "Hunting",
    "Boxing",
    "Hockey",
    "Special",
    "Figure skating",
    "Basketball",
    "Lacrosse",
    "Horse racing",
    "Agriculture",
    "Sumo wrestling",
    "Consumer",
    "Skiing",
    "Baseball",
    "Skeleton",
    "Biathlon",
    "Rugby union",
    "Motorsports",
    "Cycling",
    "Bull riding",
    "Cricket",
    "Equestrian",
    "Handball",
    "Card games",
    "Poker",
    "Freestyle Skiing",
    "Luge",
    "Cross-country skiing",
    "Ski Jumping",
    "Snowboarding",
    "Alpine skiing",
    "Track/field",
    "Motorcycle racing",
    "Rugby",
    "Surfing",
    "Diving",
    "Event",
    "Bowling",
    "Snowmobile",
    "Olympics",
    "Rodeo",
    "Cheerleading",
    "Intl soccer",
    "Opera",
    "Performing arts",
    "Theater",
    "Suspense",
    "Skateboarding",
    "Computers",
    "Running",
    "Pool",
    "Swimming",
    "Beach soccer",
    "Weightlifting",
    "Curling",
    "Volleyball",
    "Speed skating",
    "Sailing",
    "Parade",
    "Dog show",
    "Dog sled",
    "Boat racing",
    "Triathlon",
    "Rugby league",
    "Yacht racing",
    "Drag racing",
    "Watersports",
    "Arm wrestling",
    "Awards",
    "Australian rules football",
    "Water polo",
    "Bicycle",
    "Bicycle racing",
    "Field hockey",
    "Futsal",
    "Boat",
    "Mountain biking",
    "Softball",
    "Judo",
    "Racquet",
    "Squash",
    "Skating",
    "Marathon"
])


#Below set is received from below query on SearchRiverDB in mongoDB
# ﻿db.series.distinct("Genre.genreName")
series_genre_list=frozenset([
    "News",
    "Entertainment",
    "Bobsled",
    "Documentary",
    "Special",
    "Gymnastics",
    "Horse racing",
    "Community",
    "Skateboarding",
    "Mystery",
    "Shopping",
    "History",
    "War",
    "Fashion",
    "Wrestling",
    "Tennis",
    "Auction",
    "Motorcycle",
    "Art",
    "Religious",
    "Variety",
    "Drama",
    "Fundraiser",
    "Animals",
    "Nature",
    "Football",
    "Soccer",
    "House/garden",
    "Pro wrestling",
    "Auto",
    "Adventure",
    "Reality",
    "Music",
    "Fishing",
    "Outdoors",
    "Olympics",
    "Sitcom",
    "Newsmagazine",
    "Anthology",
    "Interview",
    "Awards",
    "Event",
    "Politics",
    "Public affairs",
    "Biography",
    "Travel",
    "Adults only",
    "Erotic",
    "Golf",
    "Bus./financial",
    "Soap",
    "Historical drama",
    "Crime",
    "Weather",
    "Dance",
    "Musical comedy",
    "Miniseries",
    "Romance",
    "Crime drama",
    "Western",
    "Talk",
    "Action",
    "Exercise",
    "Educational",
    "Running",
    "Cooking",
    "Sports talk",
    "Comedy",
    "Musical",
    "Animated",
    "Children",
    "Comedy drama",
    "Fantasy",
    "Boxing",
    "Thriller",
    "Game show",
    "Paranormal",
    "Action sports",
    "Rugby",
    "Surfing",
    "Watersports",
    "Snowboarding",
    "Martial arts",
    "Science fiction",
    "Health",
    "Poker",
    "Hockey",
    "Science",
    "Basketball",
    "Medical",
    "Performing arts",
    "Computers",
    "Law",
    "Docudrama",
    "Skiing",
    "Home improvement",
    "How-to",
    "Consumer",
    "Holiday",
    "Hunting",
    "Equestrian",
    "Baseball",
    "Horror",
    "Agriculture",
    "Arts/crafts",
    "Collectibles",
    "Bowling",
    "Cycling",
    "Military",
    "Anime",
    "Ballet",
    "Rodeo",
    "Figure skating",
    "Aviation",
    "Track/field",
    "Australian rules football",
    "Lacrosse",
    "Triathlon",
    "Swimming",
    "Motorcycle racing",
    "Standup",
    "Horse",
    "Volleyball",
    "Curling",
    "Speed skating",
    "Self improvement",
    "Arm wrestling",
    "Water polo",
    "Romantic comedy",
    "Auto racing",
    "Technology",
    "Theater",
    "Environment",
    "Luge",
    "Biathlon",
    "Boat",
    "Bullfighting",
    "Debate",
    "Ski Jumping",
    "Gay/lesbian",
    "Parenting",
    "Field hockey",
    "Suspense",
    "Card games",
    "Mixed martial arts",
    "Motorsports",
    "Bull riding",
    "Opera",
    "Boat racing",
    "Sumo wrestling",
    "Diving",
    "Rugby league",
    "Skeleton",
    "Snowmobile",
    "Rugby union",
    "Cricket",
    "Cheerleading",
    "Yacht racing",
    "Polo",
    "Intl soccer",
    "Drag racing",
    "Beach soccer",
    "Dark comedy",
    "Shooting",
    "Gaming",
    "Handball",
    "Freestyle Skiing",
    "Cross-country skiing",
    "Alpine skiing",
    "Aerobics",
    "Sailing",
    "Archery",
    "Parade",
    "Weightlifting",
    "Dog show",
    "Bicycle",
    "Mountain biking",
    "Bodybuilding",
    "Pool",
    "Dog sled",
    "Skating",
    "Blackjack",
    "Futsal",
    "Marathon",
    "Racquet",
    "Squash",
    "Softball"
])

total_genre_list = item_genre_list.union(series_genre_list)



fios_items={}

#TODO: Function takes a string, splits the data based on the sep argument and sends back the list
def extract_data_from_cols(genreString):
    print(genreString)
    tmplist = genreString.split('|')
    return tmplist
    #print("The Genres extracted are:")
    # for tmpgenre in tmplist:
    #     print(tmpgenre)


#Module to read CSV file to a dataframe
#TODO: Improve the reading of the CSV file by chunking it so that we are working on part of the dataframe.
def readcsvfile(filename):
    print("File Name: " + filename)
    data = None
    try:
        data = pd.read_csv(filename, error_bad_lines=False)
    except IOError:
        logging.exception('')
    return data

def processItemGenre(tmpGenreList):
    tmpdata = {}
    for tmpgenre in tmpGenreList:
        # print(tmpgenre)
        tmpdata[tmpgenre] = 1
    return tmpdata

def processItemGenre_list(item_df):
    tmp_fios_items = {}

    #For each row in the data frame select the column and form the vector for the object
    for tmpRow in item_df.itertuples():
        #print(tmpRow)
        #extract_data_from_cols(tmpRow.GENRE)
        if (isinstance(tmpRow.GENRE, str) and (tmpRow.GENRE)):
            tmp_genre_list = tmpRow.GENRE.split('|')
            # print("The Genres extracted are:")
            item_genres = processItemGenre(tmp_genre_list)
            tmp_fios_items[tmpRow.ITEM_ID]=item_genres

    return tmp_fios_items


def create_fios_dataframe(item_df):
    total_genre_num = frozenset(["total_item_genre"])
    total_cols = total_genre_list.union(total_genre_num)
    fios_genre_df = pd.DataFrame(columns=total_cols)
    for tmpRow in item_df.itertuples():


        if(isinstance(tmpRow.GENRE, str) and (tmpRow.GENRE)):
            tmp_genre_list = tmpRow.GENRE.split('|')
            tmp_genre_num = len(tmp_genre_list)
            fios_genre_df.at[tmpRow.ITEM_ID, "total_item_genre"] = tmp_genre_num
            for tmpgenre in tmp_genre_list:
                # print(tmpgenre)
                fios_genre_df.at[tmpRow.ITEM_ID, tmpgenre] = 1
        else:
            fios_genre_df.at[tmpRow.ITEM_ID, "total_item_genre"] = 0
    return fios_genre_df



def create_fios_dataframe_norm(item_df):
    #total_genre_num = frozenset(["total_item_genre"])
    #total_cols = total_genre_list.union(total_genre_num)
    fios_genre_df = pd.DataFrame(columns=total_genre_list)
    count = 0
    for tmpRow in item_df.itertuples():
        count = count + 1
        if ((count % 1000) == 0):
            print("Processing item: {}".format(count))
        if(isinstance(tmpRow.GENRE, str) and (tmpRow.GENRE)):
            tmp_genre_list = tmpRow.GENRE.split('|')
            tmp_genre_num = len(tmp_genre_list)
            for tmpgenre in tmp_genre_list:
                # print(tmpgenre)
                if tmpgenre in total_genre_list:
                    fios_genre_df.at[tmpRow.ITEM_ID, tmpgenre] = (1 / tmp_genre_num )
                else:
                    print("The genre {} not present in original Genre list".format(tmpgenre))
        else:
            pass
            #fios_genre_df.at[tmpRow.ITEM_ID, "total_item_genre"] = 0
    return fios_genre_df


### This is the main module to read the data frame and form a item vector
def main():

    print('Reading the CSV file into a dataframe')
    #Getting the item data from the csv file
    #item_data = pd.read_csv(ITEMS_FILENAME)
    item_data = readcsvfile(ITEMS_FILENAME)

    #print the top lines in the data with selected columns
    print(item_data.loc[:, select_colmns])

    #calculate the start time
    startTime = time.process_time()
    #fios_df = create_fios_dataframe(item_data)
    fios_df = create_fios_dataframe_norm(item_data)
    endTime = time.process_time()
    elapsed_time = endTime - startTime
    print("Time taken to do the job is: {}".format(elapsed_time))
    #fios_items = processItemGenre_list(item_data)

    #Write the data frame to a CSV file
    itemvCsvFile = open(ITEMS_VECTOR_CSV, 'w')
    fios_df.to_csv(itemvCsvFile, na_rep='0',mode='w')
    itemvCsvFile.close()

    print("End of processing")

main()