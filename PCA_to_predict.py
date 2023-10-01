# This python file will use scikitlearns libraries to analyse a set of data 
# by PCA and then read in a separate file full another set of data that we want to fit to the PCA

print("Importing Libraries")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv
import os
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV

#create a list of integers missing a single value between the start and end value
def createList(missingValue, startValue, endValue):
    array = []
    for i in range(startValue,missingValue):
        array.append(i)
    for i in range(missingValue+1,endValue+1):
        array.append(i)
    return array

#calculate the distance between two vectors(!D arrays/lists) of the same dimension
def score(start, end, weightings):
    
    distance = 0
    
    for i in range(len(start)):
        distance += (((start[i]-end[i])*weightings[i])**2)

    return np.sqrt(distance)


def PCAPredict():
    #Import a CSV with formating of 
    #         |ind1|ind2|ind3|ind4| ...
    #spectra1a| 6.0| 0.4| 0.5| 0.3| ...
    #spectra1b| 6.1| 0.2| 0.5| 0.2| ...
    #spectra1c| 6.3| 0.3| 0.3| 0.4| ...
    #spectra2a| 6.2| 0.5| 0.3| 0.2| ...
    #spectra2b| 6.2| 0.5| 0.3| 0.2| ...
    #spectra2c| 6.2| 0.5| 0.3| 0.2| ...
    #spectra3a| 6.2| 0.5| 0.3| 0.2| ...

    #gets the file the user would like to train the PCA on
    print("\nPlese enter the full path to a file containing the data you would like to train the PCA on")
    print("An exmaple of a format is C:\\Users\\Documents\\FolderName\\FileContainingTrainingData.csv")
    file = input("Enter here: ")

    #gets the file the user would like to fit to the trained PCA
    print("\nPlese enter the full path to a file containing the data you would like to fit")
    print("An exmaple of a format is C:\\Users\\Documents\\FolderName\\FileContainingFittingData.csv")
    file2 = input("Enter here: ")

    #reads the training data into a panda data frame
    try:
        data = pd.read_csv(file, header = 0, index_col =0)
    except:
        print("Please ensure that you entered the file path for the training data properly")
        return

    #Get the data from the data frame
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




    #create a folder to store the eigenvectors and predictions
    folderPathway = os.path.dirname(file)
    fileName = os.path.basename(file)

    print("\nPlese enter the name of the directory you would like to store the LOOCV in")
    print("An exmaple of a is \"PCA\"")
    directory = input("Enter here: ")
    
    # Path
    path = os.path.join(folderPathway, directory)

    #attempts to create a new directory
    try:
        os.mkdir(path)
    except:
        print("File already exists, do you want to override it?")
        choice = input("Please enter 'Y' or 'N': ")

        if(choice.upper() != "Y"):
            return

    # store an image of the explained variance to allow the user to choose the right number of components
    weightings=((PCA(n_components=min(len(data), len(X_colnames))).fit(X)).explained_variance_ratio_)
    plt.plot(np.cumsum(weightings))
    plt.xlabel("Number of components")
    plt.ylabel("Explained variance")
    plt.savefig(path + "\PCA Explained variance.png")
    

    #get the number of components from the user
    numComponents= int(input("Enter the number of Prinicpal Components you would like to use: "))


    #create pca from taining data
    pca = PCA(n_components=numComponents).fit(X)

    #transform the training data into the PCA space
    X_Transformed = pca.transform(X)

    #store PCA of the training data in a csv
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

    #read in the second file of the data to be fit into a panda data frame
    try:
        data2 = pd.read_csv(file2, header = 0, index_col =0)
    except:
        print("Please ensure that you entered the file path properly")
        return

    #Get the data
    X_colnames2 = data2.columns
    All_Samples2 = data2.iloc[0:len(data2),:]
    X2 = All_Samples2[X_colnames2].values

    distances = []
    distancesForProb = []
    minDistPred = []
    projectedPCA = []

    #fit the fitting data to the PCA, and find the minimum distance between 
    # the average position of the training data and the fitted data
    for i in range(len(data2)):
        #fit the missing spectra to the PC space
        TransformNew = pca.transform(X2[i,:].reshape(1, -1))
        projectedPCA.append(TransformNew[0])


        #find the minimum distance between the missing spectra and the average position of
        # samples in the PC space and add it to the score dataframe
        minDistance = score(average[0], TransformNew[0], weightings)
        index = 0
        
        tempDistance = [score(average[0], TransformNew[0], weightings)]

        for j in range(1,len(average)):
            tempDistance.append(score(average[j],TransformNew[0], weightings))
            if(tempDistance[j]<minDistance):
                minDistance = tempDistance[j]
                index = j

        distancesForProb.append(tempDistance.copy())    
        minDistPred.append(sampleNames[index])

        tempDistance.append(sampleNames[index])
        distances.append(tempDistance)
        

    
    print("Atempting to fit the classifier to the training set")
    param_grid = {
        "C": loguniform(1e3, 1e5),
        "gamma": loguniform(1e-4, 1e-1),
    }
    clf = RandomizedSearchCV(
        SVC(kernel="rbf", class_weight="balanced", decision_function_shape='ovr'), param_grid, n_iter=10
    )
    try:
        clf = clf.fit(X_Transformed, data.axes[0])
    
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)

        y_pred = clf.predict(projectedPCA)

        probabilities = clf.decision_function(projectedPCA)

        prob = []
        for i in probabilities:
            tempProb = []
            summation = 0
            for j in i:
                summation += np.exp(j)
            
            for j in i:
                tempProb.append(np.exp(j)/summation)
        
            prob.append(tempProb)
    
        #print(prob)
        print("Storing Predictions from SVM")
        dfProb = pd.DataFrame(prob, columns = sampleNames, index = data2.axes[0])
        dfProb['Prediction'] = (y_pred)
        dfProb.to_csv(os.path.join(path,("PredictionsFromSVMSoftMaxProb.csv")))


        #print(probabilities)
        dfSVC = pd.DataFrame(probabilities, columns = sampleNames, index = data2.axes[0])
        dfSVC['Prediction'] = (y_pred)
        #print(dfSVC)
        dfSVC.to_csv(os.path.join(path,("PredictionsFromSVM.csv")))
    except:
        print("Failed to use a SVM to categorise the data, will only use minimum distance")

    #stores the fitted data in a csv
    dfPCA = pd.DataFrame(projectedPCA,index = data2.axes[0])
    dfPCA.to_csv(os.path.join(path,("PCAofFittedData.csv")))

    #stores the predictions from minimum distance to a dataframe
    minDistanceProb = []
    minDistanceProb2 = []
    for i in distancesForProb:
        
        
        tempProb2 = []
        summation = 0
        arccsch = []

        tempProb = []
        tempTanh = []
        tempNormTanh = []
        tanhSummation = 0

        #calculated argmax values by first converting using arccsch
        for j in i:
            summation += j

        average = summation/len(i)

        for j in i:
            arccsch.append(np.log((average/j)+np.sqrt((average/j)*(average/j)+1)))

        summation = 0
        for j in arccsch:
            summation += np.exp(j)

        for j in arccsch:
            tempProb2.append(np.exp(j)/summation)

        minDistanceProb2.append(tempProb2)


        #calculated argmax values by first converting using tanh(1/x)
        for j in i:
            tempTanh.append(np.tanh(1/j))

        maxValue = max(tempTanh)

        for j in tempTanh:
            tempNormTanh.append(j/maxValue)

        for j in tempNormTanh:
            tanhSummation += np.exp(j)

        for j in tempNormTanh:
            tempProb.append(np.exp(j)/tanhSummation)

        minDistanceProb.append(tempProb)

    #store softmax values from tanh
    dfMin = pd.DataFrame(minDistanceProb, columns = sampleNames, index = data2.axes[0])
    dfMin['Prediction'] = (minDistPred)
    dfMin.to_csv(os.path.join(path,("PredictionsFromMinDistsoftMaxTanh.csv")))

    #store softmax values from arccsch
    dfMin2 = pd.DataFrame(minDistanceProb2, columns = sampleNames, index = data2.axes[0])
    dfMin2['Prediction'] = (minDistPred)
    dfMin2.to_csv(os.path.join(path,("PredictionsFromMinDistsoftMaxArccsch.csv")))


    sampleNames.append("Prediction")
    df = pd.DataFrame(distances, columns = sampleNames,index = data2.axes[0])
    
    #store the eigenvectors for the pca
    print("Storing eigenVectors for PCA")
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
    print("Storing minimum distance predictions")
    df.to_csv(os.path.join(path,("PredictionsFromMinDistance.csv")))
    return


def Main():
    print("\nThis program will perform PCA on a set of spectra and fit another set of spectra to the same space"
         +"\nThe program will take two 'csv's please ensure that they are formatted as shown below")
    print("\n       |ind1|ind2|ind3|ind4| ..."
        + "\nsample1| 6.0| 0.4| 0.5| 0.3| ..."
        + "\nsample1| 6.1| 0.3| 0.5| 0.2| ..."
        + "\nsample2| 6.3| 0.5| 0.3| 0.4| ..."
        + "\nsample2| 6.2| 0.6| 0.3| 0.2| ..."
        + "\nsample3| 7.2| 0.6| 0.3| 1.2| ..."
        + "\nsample3| 7.1| 0.4| 0.3| 1.3| ..."
        + "\nsample4| 6.2| 2.6| 4.3| 3.1| ..."
        + "\nsample4| 6.3| 2.5| 4.5| 3.2| ...")
    
    PCAPredict()

    while(True):
        print("\nAre you finished performing PCAs?")
        stop = input("Please enter 'Y' or 'N': ")

        if(stop.upper() == "Y"):
            return
        #end if

        PCAPredict()
    #end while
#end Main

Main()