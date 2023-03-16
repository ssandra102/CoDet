
####
# This code is already modified and included in main.py as required.
# Check this code for exclusive implementation of KNN algorithm...
####

from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import streamlit as st

#-----To test run-----#
#image = cv2.imread('04.jpg')
#print("Type  :{}".format(type(image)))
#print("Shape :{}".format(image.shape))
#plt.imshow(image)
#image = cv2.resize(image, (450,450), interpolation = cv2.INTER_AREA)
#plt.imshow(image)
#---------------------#


#Resizing and converting to RGB(from BGR) all the images in the folder
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    return modified_image


#-----To test run-----#
#plt.imshow(get_image('images/01.jpg'))
#---------------------#

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))





#K means algorithm used to group(cluster) colors found in a pic

def get_colors(modified_image, number_of_colors, show_chart):
    
    #number to clusters to be formed
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    #print("counts : {}".format(counts))

    center_colors = clf.cluster_centers_
    #print("cluster centers :{}".format(center_colors))

    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    #....
    st.write("ordered colors:{}".format(ordered_colors))
    #print("hex colors    :".format(hex_colors))
    #print("rgb colors    :".format(rgb_colors))

    #displaying using a pie chart
    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    
    return rgb_colors




#-----To test run-----#
#(ignore)
#pic = cv2.imread('images/04.jpg')
#modified_image = cv2.resize(pic, (600, 400), interpolation = cv2.INTER_AREA)
#modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
#clf = KMeans(n_clusters = 2)
#labels = clf.fit_predict(modified_image)
#print(labels)
#---------------------#
