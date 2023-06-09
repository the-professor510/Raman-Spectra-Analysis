# This python file will use scikitlearns libraries to analyse a set of data 
# by PCA and then read in a separate file full another set of data that we want to fit to the PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv
import os



#create a list of integers missing a single value between the start and end value
def createList(missingValue, startValue, endValue):
    array = []
    for i in range(startValue,missingValue):
        array.append(i)
    for i in range(missingValue+1,endValue+1):
        array.append(i)
    return array

#calculate the distance between two vectors(!D arrays/lists) of the same dimension
def distance(start, end):
    
    distance = 0
    
    for i in range(len(start)):
        distance += ((start[i]-end[i])**2)

    return np.sqrt(distance)



#Import a CSV with formating of 
#         |ind1|ind2|ind3|ind4| ...
#spectra1a| 6.0| 0.4| 0.5| 0.3| ...
#spectra1b| 6.1| 0.2| 0.5| 0.2| ...
#spectra1c| 6.3| 0.3| 0.3| 0.4| ...
#spectra2a| 6.2| 0.5| 0.3| 0.2| ...
#spectra2b| 6.2| 0.5| 0.3| 0.2| ...
#spectra2c| 6.2| 0.5| 0.3| 0.2| ...
#spectra3a| 6.2| 0.5| 0.3| 0.2| ...

#file names formatted as shown above
#file = r"C:\Users\edwar\OneDrive\Documents\University\Summer_2023_Internship\Data\RecycledBottles\DifferentWhiskies\PCA of unopened Bottles\UnopenedBottles.csv"
#file = r"C:\Users\edwar\OneDrive\Documents\University\Summer_2023_Internship\Data\RecycledBottles\DifferentWhiskies\PCA of unopened Bottles\UnopenedBottlesLimitedRamanShift.csv"
#file = r"C:\Users\edwar\OneDrive\Documents\University\Summer_2023_Internship\Data\RecycledBottles\DifferentWhiskies\PCA of unopened Bottles\UnopenedBottlesNormalised.csv"
file = r"C:\Users\edwar\OneDrive\Documents\University\Summer_2023_Internship\Data\RecycledBottles\DifferentWhiskies\PCA of unopened Bottles\UnopenedBottlesNormalisedLimitedRamanShift.csv"

#file2 names formatted as shown above
#file2 = r"C:\Users\edwar\OneDrive\Documents\University\Summer_2023_Internship\Data\RecycledBottles\DifferentWhiskies\PCA of unopened Bottles\OpenedBottles.csv"
#file2 = r"C:\Users\edwar\OneDrive\Documents\University\Summer_2023_Internship\Data\RecycledBottles\DifferentWhiskies\PCA of unopened Bottles\OpenedBottlesLimitedRamanShift.csv"
#file2 = r"C:\Users\edwar\OneDrive\Documents\University\Summer_2023_Internship\Data\RecycledBottles\DifferentWhiskies\PCA of unopened Bottles\OpenedBottlesNormalised.csv"
file2 = r"C:\Users\edwar\OneDrive\Documents\University\Summer_2023_Internship\Data\RecycledBottles\DifferentWhiskies\PCA of unopened Bottles\OpenedBottlesNormalisedLimitedRamanShift.csv"


data = pd.read_csv(file, header = 0, index_col =0)

#Get the data
X_colnames = data.columns
All_Samples = data.iloc[0:len(data),:]
X = All_Samples[X_colnames].values


#Get a list of all the names of the samples
sampleNames = [data.axes[0][0]]
numberSamples = [0]
for i in range(len(data.axes[0])):
    notContained = True
    for j in range(len(sampleNames)):
        if not (notContained and sampleNames[j] != data.axes[0][i]):
            notContained = False

    if notContained:
        sampleNames.append(data.axes[0][i])
        numberSamples.append(0)




#create a folder to store the Eigenvalues and loocv
folderPathway = os.path.dirname(file)
fileName = os.path.basename(file)

directory = "PCAtoPredict" + fileName
  
# Path
path = os.path.join(folderPathway, directory)

#attempts to create a new directory
try:
    os.mkdir(path)
except:
  print("File already exists")

# store an image of the explained variance to allow the user to choose the right number of components
plt.plot(np.cumsum((PCA(n_components=min(len(data), len(X_colnames))).fit(X)).explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Explained variance")
plt.savefig(path + "\PCA Explained variance.png")

#get the number of components from the user
numComponents= int(input("Enter the number of Prinicpal Components you would like to use: "))


#get a new PCA missing the ith spectra in the given data

pca = PCA(n_components=numComponents).fit(X)

X_Transformed = pca.transform(X)

dfPCA = pd.DataFrame(X_Transformed,index = data.axes[0])
dfPCA.to_csv(os.path.join(path,("PCAofTrainingData.csv")))

#find the average position of each smaple in the PC space
average = []
for j in range(len(sampleNames)):
    #find index of all scores relating to first name
    index = []

    for x in range (len(data)):
        if sampleNames[j] == data.axes[0][x]:
            index.append(x)

    total = []
    subtraction = 0


    total = X_Transformed[index[0]]
    
    for x in index[1:]:
        for y in range(numComponents):
            total[y] += X_Transformed[x,y]

    for y in range(numComponents):
        total[y] = total[y]/len(index)

    average.append(total.copy())



data2 = pd.read_csv(file2, header = 0, index_col =0)

#Get the data
X_colnames2 = data2.columns
All_Samples2 = data2.iloc[0:len(data2),:]
X2 = All_Samples2[X_colnames2].values

distances = []
projectedPCA = []

for i in range(len(data2)):
    #fit the missing spectra to the PC space
    TransformNew = pca.transform(X2[i,:].reshape(1, -1))
    #print(TransformNew)
    projectedPCA.append(TransformNew[0])


    #find the minimum distance between the missing spectra and the average position of
    # samples in the PC space and add it to the score dataframe
    minDistance = distance(average[0], TransformNew[0])
    index = 0
    tempDistance = [distance(average[0], TransformNew[0])]

    for j in range(1,len(average)):
        tempDistance.append(distance(average[j],TransformNew[0]))
        #print(tempDistance)
        if(tempDistance[j]<minDistance):
            minDistance = tempDistance[j]
            index = j
        
    tempDistance.append(sampleNames[index])
    distances.append(tempDistance)

# create a data frame to store the loocv
sampleNames.append("Prediction")
df = pd.DataFrame(distances, columns = sampleNames,index = data2.axes[0])
#print(df)
dfPCA = pd.DataFrame(projectedPCA,index = data2.axes[0])
dfPCA.to_csv(os.path.join(path,("PCAofFittedData.csv")))

#store the eigenvectors for the pca
fileName = "eigenVectors.csv"
filePath = os.path.join(path,fileName)
try:
    os.remove(filePath)
except:
    #Do Notihng
    doNothing = 0

#save the eigenvectors
with open(filePath, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_NONE)
    header = ["Raman Shift"] + list(X_colnames)
    writer.writerow(header)
    for counter in range(numComponents):
        row = [("PC"+ str(counter+1))] + list(pca.components_[counter])
        writer.writerow(row)


#print the socres and save them to the same file as the eigenvectors
print(df)
df.to_csv(os.path.join(path,("Predictions.csv")))