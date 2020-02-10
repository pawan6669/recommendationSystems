import pandas as pd
import logging
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

#Libraries to show data plot
import matplotlib.pyplot as plt
import seaborn as sns


#ITEMS_VECTOR_CSV='file://localhost/Users/pawan/PycharmProjects/recommendationsystems/items_vector_norm_v2.csv'
ITEMS_VECTOR_CSV='//Users/pawan/PycharmProjects/recommendationsystems/items_vector_norm_v2.csv'

ITEMS_CLUSTER_NAME="item_cluster_"

def main():
    #item_vector_df = pd.read_csv(ITEMS_VECTOR_CSV, error_bad_lines=False, nrows=50000)

    #item_vector_df = pd.read_csv(ITEMS_VECTOR_CSV, error_bad_lines=False, index_col= 0, usecols=range(0,192,1))
    #TODO: The data in the column 0 was storing rowIds.  Change this if it is corrected in the previous module.
    item_vector_df = pd.read_csv(ITEMS_VECTOR_CSV, error_bad_lines=False, index_col=0)
    inpRows, inpCols = item_vector_df.shape
    print("The dimensions of the dataframe: rows: {}, columns: {}".format(inpRows, inpCols))
    fiosIds = item_vector_df.index.values
    # #kmeans_sample = KMeans(n_clusters=3).fit(item_vector_df)
    #
    Sum_of_squared_distances = []
    silhoutte_values = []
    K = range(20, 200, 20)
    for k in K:
        print("Processing k value : {} ".format(k))
        #km = KMeans(n_clusters=k)
        km = MiniBatchKMeans(n_clusters=k, batch_size=9500)
        km = km.fit(item_vector_df)
        Sum_of_squared_distances.append(km.inertia_)
        silhoutte_values.append(silhouette_score(item_vector_df, km.labels_))

        #form the output file
        tmp_cluster_center_name = ITEMS_CLUSTER_NAME + "_centers_"+ str(k) + ".csv"
        print("Storing cluster center in file {}".format(tmp_cluster_center_name))
        tmp_cluster_centers_df = pd.DataFrame(km.cluster_centers_)

        itemvClusterCenterFile = open(tmp_cluster_center_name, 'w')
        tmp_cluster_centers_df.to_csv(itemvClusterCenterFile, na_rep='0', mode='w')
        itemvClusterCenterFile.close()

        tmp_item_vec_space_df = pd.DataFrame(km.labels_)
        tmpitemRows, tmpitemCols = tmp_item_vec_space_df.shape
        print("The dimensions of the processed item dataframe: rows: {}, columns: {}".format(tmpitemRows, tmpitemCols))

        if (inpRows == tmpitemRows):
            tmp_item_vec_space_df.index = fiosIds
        else:
            print("WARN: Number of rows are not equal!!!! WARN")
        tmp_items_assigned_clusters = ITEMS_CLUSTER_NAME + "_center_vector_space"+ str(k) + ".csv"
        itemVectorspaceFile = open(tmp_items_assigned_clusters, 'w')
        tmp_item_vec_space_df.to_csv(itemVectorspaceFile, na_rep='0', mode='w')
        itemVectorspaceFile.close()

        print("End of processing for K {}".format(k))

    #Dump the values of Sum of squared distances
    #Sum_of_squared_distances_df = pd.DataFrame(data= {'KValue': K, 'Sum_of_squares': Sum_of_squared_distances} ,columns = ["Value of K","Sum_of_squared_distance"])
    Sum_of_squared_distances_df = pd.DataFrame(data={'KValue': K, 'Sum_of_squares': Sum_of_squared_distances})
    #Sum_of_squared_distances_df["Value of K"] = K
    #Sum_of_squared_distances_df["Sum_of_squared_distance"] = Sum_of_squared_distances
    squaredDistFile = open("sum_of_squared_distance", 'w')
    Sum_of_squared_distances_df.to_csv(squaredDistFile, na_rep='0', mode='w')
    squaredDistFile.close()

    #Initialize the plot
    fig = plt.figure(figsize=(10, 4))

    #plot the sum of squared distances
    plt.subplot(1,2, 1)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    #plt.show()


    #Dump the values of silhoutte values
    silhouette_score_df = pd.DataFrame(data={'KValue': K, 'Silhoutte_score': silhoutte_values})
    silhouetteScoreFile = open("Silhoutte_score_file", 'w')
    silhouette_score_df.to_csv(silhouetteScoreFile, na_rep='0', mode='w')
    silhouetteScoreFile.close()


    #Plot the Silhouette Values
    plt.subplot(1, 2, 2)
    plt.plot(K, silhoutte_values, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Value')
    plt.title('Silhouette Values for K')

    plt.show()

main()