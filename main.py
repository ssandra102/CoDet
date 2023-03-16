
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import streamlit as st


st.title('Streamlit web app')

st.subheader("""
Explore. Classifiers and Datasets.
KNN Classifier \n
K-Nearest Neighbors is one of the most basic yet essential classification algorithms in Machine Learning. 
It belongs to the supervised learning domain and finds intense application in pattern recognition, data mining
and intrusion detection.\n
Intuition \n
If we plot some points on a graph, we may be able to locate some clusters or groups. Now, given an 
unclassified point, we can assign it to a group by observing what group its nearest neighbors belong to. 
This means a point close to a cluster of points classified as, say ‘Red’ has a higher probability of getting 
classified as ‘Red’.\n
Find out the composition of an image.
""")
st.write("read more here:[link](https://www.geeksforgeeks.org/k-nearest-neighbours/) ")
st.write("-------------------------------------------------- ")

selected_box = st.sidebar.selectbox(
    'Choose an image',
    ('1','2','3','4','5')
    )
x = st.sidebar.slider('Number of colors',min_value = 3,max_value = 10)


if selected_box == '1':
    path = '01.jpg'
if selected_box == '2':
    path = '02.jpg'
if selected_box == '3':
    path = '03.jpg'
if selected_box == '4':
    path = '04.jpg'
if selected_box == '5':
    path = '05.jpg'



st.image(path,width=300)

#Resizing and converting to RGB(from BGR) 
pic = cv2.imread(path)
pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
modified_image = cv2.resize(pic, (600, 400), interpolation = cv2.INTER_AREA)
modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

#K means algorithm used to group(cluster) colors found in a pic
#number to clusters to be formed
clf = KMeans(n_clusters = x)
labels = clf.fit_predict(modified_image)
counts = Counter(labels)
# sort to ensure correct color percentage
counts = dict(sorted(counts.items()))
center_colors = clf.cluster_centers_

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


ordered_colors = [center_colors[i] for i in counts.keys()]
rgb_colors = [ordered_colors[i] for i in counts.keys()]
hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
st.write(" ")
st.write("-------------------------------------------------- ")
st.write("COLOR COMPOSITIONS")
#st.write("1. ordered colors:{}".format(ordered_colors))
st.write(" hex colors    :{}".format(hex_colors))
#st.write("3. rgb colors    :{}".format(rgb_colors))



fig1, ax1 = plt.subplots()
ax1.pie(counts.values(), labels=hex_colors, colors = hex_colors ,autopct='%1.1f%%',
    startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)

if (st.button("display code")):
    with st.echo(code_location='below'):
        ####
        #Resizing and converting to RGB(from BGR) 
        pic = cv2.imread(path)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        modified_image = cv2.resize(pic, (600, 400), interpolation = cv2.INTER_AREA)
        modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
        
        #K means algorithm used to group(cluster) colors found in a pic
        #number to clusters to be formed
        clf = KMeans(n_clusters = x)
        labels = clf.fit_predict(modified_image)
        counts = Counter(labels)
        # sort to ensure correct color percentage
        counts = dict(sorted(counts.items()))
        center_colors = clf.cluster_centers_
        ####
