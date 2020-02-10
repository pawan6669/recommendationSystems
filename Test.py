'''
    This file is used to validate the personalization models.
    Input:It takes the user Interaction data.
    Method:
    1.  Randomizes, divides into training and test data.
    2.  Calls training of the models
    3.  Calls test matrix
    4.  Computes the percentage of hit and miss for each user
    5.  Computes the percentage of hits for all users.


    Output:
    1. Percentage of hits for each user.
    2. Overall percentage of hits for all users.

'''

import pandas as pd
import logging
import time
import random
import numpy as np
from sklearn.cluster import KMeans

#Libraries to show data plot
import matplotlib.pyplot as plt
import seaborn as sns

#USER_INTERACTION_DATA="file://localhost/Users/pawan/PycharmProjects/recommendationsystems/user_genre_interactions_new.csv"
#USER_INTERACTION_FILE="file://localhost/Users/pawan/PycharmProjects/recommendationsystems/user_musical_1_interactions.csv"
#+USER_INTERACTION_FILE="file://localhost/Users/pawan/PycharmProjects/recommendationsystems/User_Beachvolleyball_50_interactions.csv"
#USER_INTERACTION_FILE="file://localhost/Users/pawan/PycharmProjects/recommendationsystems/User_Crime_12_interactions.csv"
#USER_INTERACTION_FILE="file://localhost/Users/pawan/PycharmProjects/recommendationsystems/User_Religious_8_interactions.csv"
#USER_INTERACTION_FILE="file://localhost/Users/pawan/PycharmProjects/recommendationsystems/User_Documentary_15_interactions.csv"
#USER_INTERACTION_FILE="file://localhost/Users/pawan/PycharmProjects/recommendationsystems/User_Action_1.csv"
#USER_INTERACTION_FILE="file://localhost/Users/pawan/PycharmProjects/recommendationsystems/User_Animated_50.csv"
USER_INTERACTION_FILE="file://localhost/Users/pawan/PycharmProjects/recommendationsystems/User_War_50.csv"
BASE_DIR=""
ITEMS_FILE="file://localhost/Users/pawan/PycharmProjects/recommendationsystems/item_cluster__center_vector_space160.csv"
CLUSTER_CENTER_FILE="file://localhost/Users/pawan/PycharmProjects/recommendationsystems/item_cluster__centers_160.csv"
RAW_ITEMS_FILENAME='file://localhost/Users/pawan/PycharmProjects/recommendationsystems/items_genre_new.csv'
items_df = pd.DataFrame()
USER_TRAIN_PERCENT=0.6
USER_TEST_FACTOR_PERCENT=0.05
USER_ITEM_CLUSTER_DF=pd.DataFrame()
CLUSTER_CENTERS_IDS = []
user_data_df=pd.DataFrame()
NUM_OF_ITEMS_TO_BE_SELECTED = 100
select_colmns = ['ITEM_ID','GENRE']

def readcsvfile(filename):
    print("File Name: " + filename)
    data = None
    try:
        data = pd.read_csv(filename, error_bad_lines=False)
    except IOError:
        logging.exception('')
    return data


def load_data() -> object:
    global items_df
    global CLUSTER_CENTERS_IDS
    global user_data_df
    #read items into a data frame and rename the columns
    items_df = readcsvfile(ITEMS_FILE)
    items_df.columns = ["ITEM_ID", "CLUSTER_NUM"]
    items_df.index = items_df.loc[:,"ITEM_ID"]

    #read the complete user data
    user_interaction_df = readcsvfile(USER_INTERACTION_FILE)
    user_select_columns = ["USER_ID","ITEM_ID"]
    user_data_df = user_interaction_df.loc[:,user_select_columns]

    #Get the value of the cluster_center_IDS
    cluster_center_df = readcsvfile(CLUSTER_CENTER_FILE)
    CLUSTER_CENTERS_IDS = cluster_center_df.index.values

def test_user(user_df):
    random.seed()
    #Randomly select the items from the items viewed.
    user_train_data = user_df.sample(frac=0.6, replace=False, random_state = random.seed())
    user_test_data = user_df[~user_df.isin(user_train_data)].dropna()
    tmpdata = {"traindata": user_train_data, "test_data": user_test_data}
    return tmpdata



def user_train(user_train_df):
    global USER_ITEM_CLUSTER_DF
    
    #Make sure the user data frame has columns right
    USER_ITEM_CLUSTER_DF = pd.DataFrame(columns=CLUSTER_CENTERS_IDS)
    present = 0
    notpresent = 0

    try:
        for tmpRow in user_train_df.itertuples():
            bool = False
            if (items_df.index == tmpRow.ITEM_ID).any():
                #TODO: COLUMN is hard coded. Can we change this?
                tmpClusterCol = items_df.loc[tmpRow.ITEM_ID, 'CLUSTER_NUM']
                bool = True
                #Check if the item is already present
                if(USER_ITEM_CLUSTER_DF.index == tmpRow.USER_ID).any():
                    USER_ITEM_CLUSTER_DF.loc[tmpRow.USER_ID, tmpClusterCol] += 1
                else:
                    USER_ITEM_CLUSTER_DF.at[tmpRow.USER_ID, tmpClusterCol] = 1
                present += 1
            else:
                notpresent += 1
            #tmpCluster = 1
            #print("UserId: {}, Item:  {}, Present: {}".format(tmpRow.USER_ID, tmpRow.ITEM_ID, bool))

        #Fill all the sparse matrix NANs with 0
        USER_ITEM_CLUSTER_DF.fillna(0, inplace=True)
    except Exception as exp:
        logging.exception('')


    print("Items Present {}, Not present: {}".format(present, notpresent) )
    # Write the user train data to a file
    # userTrainFile = open(USER_TRAIN_CSV_FILE, 'w')
    # user_cluster_df.to_csv(userTrainFile, na_rep='0', mode='w')
    # userTrainFile.close()


'''
    This function generated the test items that needs to be presented to the user. 
    Input arguments: 
        User test content: Contents which are watched by user and being used as test content
        Items not viewed by user:  Contents which the items have not used.
        Total Number of items to be presented to user:
        
    Output:
        A list of items that need to be sent to the sorting module to sort the items to be presented to the user.
'''

def select_test_data(user_test_items = None, items_not_viewed = None, total_items = 100):
    random.seed()
    num_user_test_items = int(min(total_items * USER_TEST_FACTOR_PERCENT, len(user_test_items)))
    random_test_user_items = random.sample(set(user_test_items), num_user_test_items)
    random_test_items_not_viewed = random.sample(set(items_not_viewed), total_items - num_user_test_items)

    ## Shuffles the generated list.
    test_data = random_test_items_not_viewed + random_test_user_items
    #test_data = random_test_items_not_viewed
    #shuffled_test_data = random.shuffle(test_data)
    random.shuffle(test_data)
    return test_data



def display_data(ipItems):
    print("Displaying the data")
    original_items_df = readcsvfile(RAW_ITEMS_FILENAME)
    original_items_df = original_items_df[original_items_df.ITEM_ID.isin(ipItems)]
    original_items_df = original_items_df.set_index('ITEM_ID')
    #print(original_items_df.loc[ipItems,select_colmns])
    print(original_items_df.loc[ipItems, 'GENRE'])

def prioritize_clusters(userId = None):
    print('Prioritizing item clusters to be displayed for User:  {}'.format(userId))
    opClusters = []
    tmpdata_df = pd.DataFrame()
    #Get the userId row from the dataframe
    if userId is None:
        print("ERROR: UserId is not set. Exit with NO data")

    else:
        #Check if the user data is present.
        if(USER_ITEM_CLUSTER_DF.index == userId).any():
            print("User data exists")
            tmpclusterdata = {'AFFINITY':USER_ITEM_CLUSTER_DF.loc[userId, :]}

            #Transformed user data
            # tmpdata_df = tmpdata_df.transpose()
            # tmpdata_df = tmpdata_df.sort_values(userId, inplace=True, ascending=False)
            index_values = USER_ITEM_CLUSTER_DF.columns.values

            tmpdata_df = pd.DataFrame(tmpclusterdata, index = USER_ITEM_CLUSTER_DF.columns.values)
            tmpdata_df.sort_values('AFFINITY', inplace=True, ascending=False)
            print(tmpdata_df.loc[:,'AFFINITY'])
            opClusters = tmpdata_df.index.values

        else:
            print("User data doesn't exist")

    #return opClusters
    return tmpdata_df


def prioritize_items(inpItems = None, userId = None):
    priority_cluster_df = prioritize_clusters(userId)
    total_item_num = len(inpItems)
    count = 0;
    opItems = []
    filtered_item_df = items_df[items_df.ITEM_ID.isin(inpItems)]
    for tmpClusterId in priority_cluster_df.index.values:
        if (priority_cluster_df.at[tmpClusterId, "AFFINITY"] != 0):
            print("Retriewing items with cluster Id: " + str(tmpClusterId))
            tmpNumItems = 0
            tmpNewItems = []
            tmpclusterIdItems_df = filtered_item_df[filtered_item_df["CLUSTER_NUM"] == tmpClusterId]
            tmpNumItems = len(tmpclusterIdItems_df.index)
            print(" Number of items with this cluster Id: ".format(tmpNumItems))
            tmpNewItems = list(tmpclusterIdItems_df.index.values)
            print(tmpNewItems)
            opItems.extend(tmpNewItems)
            count = count + tmpNumItems
        else:
            print("There are no more valid clusters")
            break

    if(count < total_item_num):
        remainingItems = [x for x in inpItems if x not in opItems]
        opItems.extend(remainingItems)
    return opItems

def main():
    # calculate the start time
    startTime = time.process_time()
    load_data()
    #items_df = tmpdata["item_data"]
    #user_data_df = tmpdata["user_data"]
    endTime = time.process_time()
    loading_time = endTime - startTime
    print("Time taken to do the job is: {}".format(loading_time))

    #Get the items which are not viewed by the user at all
    #TODO: The Column name is hardcoded in here. Try to remove this dependency.
    tmpitems_notviewed = items_df[~items_df.ITEM_ID.isin(user_data_df.ITEM_ID)].dropna()

    tmpitems_notviewed.columns = ["ITEM_ID","CLUSTER_NUM"]
    tmpitems_notviewed_list = tmpitems_notviewed.loc[:,"ITEM_ID"]

    #test the user
    tmpuserdata = test_user(user_data_df)

    user_train_df = tmpuserdata["traindata"]
    user_test_df = tmpuserdata["test_data"]
    user_test_item_ids = user_test_df.loc[:,"ITEM_ID"]

    #Get the user Id
    tmpUserId = user_train_df.loc[:,"USER_ID"].values[0]
    print("The rows in train data: {}".format(len(user_test_item_ids)))

    # Train the user data
    print("Start of User training")
    user_train(user_train_df)
    print("End of user training")

    test_data_selected = select_test_data(user_test_items=user_test_item_ids, items_not_viewed=tmpitems_notviewed_list, total_items=NUM_OF_ITEMS_TO_BE_SELECTED)

    #Display the data in the order it was sent to the sorting module
    print("Data to be sorted: ")
    display_data(test_data_selected)

    user_presented_data = prioritize_items(test_data_selected, userId=tmpUserId)

    #Now we display the data in the order presented to the user.
    print("Sorted Items for the user: ")
    display_data(user_presented_data)

main()
