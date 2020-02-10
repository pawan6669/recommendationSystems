"""
This file gets the user item interactions and trains the user item vector cluster
"""


import pandas as pd
import logging
import time
import numpy as np
from sklearn.cluster import KMeans

#Libraries to show data plot
import matplotlib.pyplot as plt
import seaborn as sns

# USER_TRAIN_VECTOR_FILE="file://localhost/Users/pawan/PycharmProjects/recommendationsystems/user_item_interations.csv"
# FIOS_ITEM_CLUSTER_FILE="file://localhost/Users/pawan/PycharmProjects/recommendationsystems/item_cluster__center_vector_space4000.csv"
# CLUSTER_CENTER_FILE="file://localhost/Users/pawan/PycharmProjects/recommendationsystems/item_cluster__centers_4000.csv"
# USER_TRAIN_CSV_FILE="//Users/pawan/PycharmProjects/recommendationsystems/user_train_data.csv"
# NUMBER_OF_CLUSTERS=4000

def readcsvfile(filename, indexCol = 99999):
    print("File Name: " + filename)
    data = None
    try:
        if (indexCol == 99999):
            data = pd.read_csv(filename, error_bad_lines=False)
        else:
            data = pd.read_csv(filename, error_bad_lines=False, index_col = indexCol)
    except IOError:
        logging.exception('')
    return data


def user_train():
    ### read the user interaction data into a dataframe.
    user_train_df = readcsvfile(USER_TRAIN_VECTOR_FILE)

    ### read the fios_id mapping
    fios_id_df = readcsvfile(FIOS_ITEM_CLUSTER_FILE, indexCol = 0)

    #read the number of cluster centers
    cluster_center_df = readcsvfile(CLUSTER_CENTER_FILE, indexCol = 0)

    #Get the cluster numbers from the cluster center df rowIds
    cluster_centers = cluster_center_df.index.values

    user_cluster_df = pd.DataFrame(columns=cluster_centers)
    present = 0
    notpresent = 0

    try:
        for tmpRow in user_train_df.itertuples():
            #find the cluster for the fiosId
            #tmpCluster = fios_id_df.loc[tmpRow.ITEM_ID, 0]
            #tmpRowId = str(tmpRow.ITEM_ID)
            bool = False
            if (fios_id_df.index == tmpRow.ITEM_ID).any():
                tmpCluster = fios_id_df.loc[tmpRow.ITEM_ID, '0']
                bool = True
                #Check if the item is already present
                if(user_cluster_df.index == tmpRow.USER_ID).any():
                    user_cluster_df.loc[tmpRow.USER_ID, tmpCluster] += 1
                else:
                    user_cluster_df.at[tmpRow.USER_ID, tmpCluster] = 1
                present += 1
            else:
                notpresent += 1
            #tmpCluster = 1
            print("UserId: {}, Item:  {}, Present: {}".format(tmpRow.USER_ID, tmpRow.ITEM_ID, bool))
    except Exception as exp:
        logging.exception('')


    print("Items Present {}, Not present: {}".format(present, notpresent) )
    # Write the user train data to a file
    userTrainFile = open(USER_TRAIN_CSV_FILE, 'w')
    user_cluster_df.to_csv(userTrainFile, na_rep='0', mode='w')
    userTrainFile.close()


user_train()