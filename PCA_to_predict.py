# This python file will use scikitlearns libraries to analyse a set of data 
# by PCA and then read in a separate file full another set of data that we want to fit to the PCA

print("Importing Libraries")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

#create a list of integers missing a single value between the start and end value
def createList(missingValue, startValue, endValue):
    array = []
    for i in range(startValue,missingValue):
        array.append(i)
    for i in range(missingValue+1,endValue+1):
        array.append(i)
    return array

#returns an evaluation score when considering just one principal component
#is passed only one value for average, std, and fitted
def scoreOnePC(average, std, fitted):
    
    score = ((fitted-average)**2)
    score = (-1)*score/(2*(std**2))
    score = np.exp(score)

    return score


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

    PredChoices = sampleNames.copy()
    PredChoices.append("Unknown")

    columnName = sampleNames.copy()
    columnName.append("Prediction")



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
    plt.savefig(path + r"\PCA Explained variance.png")
    

    #get the number of components from the user
    numComponents= int(input("Enter the number of Prinicpal Components you would like to use: "))

    #get the number of components from the user
    UncertaintyScore = float(input("Enter the minimum score you would like to have to not be predicted as unknown: "))

    #create pca from taining data
    pca = PCA(n_components=numComponents).fit(X)

    #transform the training data into the PCA space
    X_Transformed = pca.transform(X)

    #store PCA of the training data in a csv
    dfPCA = pd.DataFrame(X_Transformed,index = data.axes[0])
    dfPCA.to_csv(os.path.join(path,("PCAofTrainingData.csv")))

    #find the average position and standard deviation of each smaple in the PC space
    average = []
    std = []
    for j in range(len(sampleNames)):
        #find index of all scores relating to first name
        index = []

        for x in range (len(data)):
            if sampleNames[j] == data.axes[0][x]:
                index.append(x)

        total = []


        total = X_Transformed[index[0]]

        #get list to find std of
        listOfPC = X_Transformed[index[0]]
        
        for x in index[1:]:

            listOfPC = np.column_stack((listOfPC, X_Transformed[x]))

            for y in range(numComponents):
                total[y] += X_Transformed[x,y]

        #calculate average
        for y in range(numComponents):
            total[y] = total[y]/len(index)

        average.append(total.copy())

        #calculate std
        std.append(np.std(listOfPC, axis =1))

    #store average PC values and standard deviations
    dfAverages = pd.DataFrame(average,index = sampleNames)
    dfAverages.to_csv(os.path.join(path,("AvgOfPCs.csv")))

    dfSTD = pd.DataFrame(std, index = sampleNames)
    dfSTD.to_csv(os.path.join(path,("STDofPCs.csv")))

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


    projectedPCA = []
    predictionsMinScore = []
    predictionAvgScore = []

    #fit the fitting data to the PCA, and find the minimum distance between 
    # the average position of the training data and the fitted data
    for i in range(len(data2)):
        #fit the missing spectra to the PC space
        TransformNew = pca.transform(X2[i,:].reshape(1, -1))
        projectedPCA.append(TransformNew[0])

        scorePC = np.array(scoreOnePC(average[0][0], std[0][0], TransformNew[0][0]))
        for k in range(1,len(average)):
            scorePC = np.append(scorePC, scoreOnePC(average[k][0], std[k][0], TransformNew[0][0]))

        
        scores = np.array(scorePC)
        scoreColNames = [("PC" + str(0))]

        #find the probability scores when considering each axis individually
        for j in range(1,numComponents):
            #calculate the socres due to the jth principle component
            scorePC = np.array(scoreOnePC(average[0][j], std[0][j], TransformNew[0][j]))
            for k in range(1,len(average)):
                scorePC = np.append(scorePC, scoreOnePC(average[k][j], std[k][j], TransformNew[0][j]))

            scores = np.column_stack((scores, scorePC))
            scoreColNames += [("PC" + str(j))]

        minScores = np.min(scores, axis=1)
        averageScore = np.average(scores, axis=1)

        scores = np.column_stack((scores, minScores))
        scoreColNames += [("Minimum Confidence score")]
        scores = np.column_stack((scores,averageScore))
        scoreColNames += [("Average Confidence Score")]

        index = 0
        maxMinScore = minScores[0]
        for j in range(1,len(minScores)):
            if maxMinScore < minScores[j]:
                maxMinScore = minScores[j]
                index = j

        if maxMinScore < UncertaintyScore:
            index = len(sampleNames)

        #append the scores and predictions to a list of all the minscores and predictions
        predictionsMinScore.append((np.concatenate((minScores,np.array([PredChoices[index]])))).tolist())

        index = 0
        maxAvgScore = averageScore[0]
        for j in range(1,len(averageScore)):
            if maxAvgScore < averageScore[j]:
                maxAvgScore = averageScore[j]
                index = j

        if maxAvgScore < UncertaintyScore:
            index = len(sampleNames)

        predictionAvgScore.append((np.concatenate((averageScore,np.array([PredChoices[index]])))).tolist())


        #store the scores for each pc in a different 
        dfscores = pd.DataFrame(scores, columns = scoreColNames, index = sampleNames)
        scoresPath = os.path.join(path, "ScoresOfPCs")
        try:
            os.mkdir(scoresPath)
        except:
            #Do Notihng
            doNothing = 0
        #store the scores of the principal components of the training data
        fileName = "Scores for " + str(i+1) + "th spectra in fitting file.csv"
        filePath = os.path.join(scoresPath,fileName)
        dfscores.to_csv(filePath)


    #store the predictions
    df = pd.DataFrame(predictionsMinScore, columns = columnName, index = data2.axes[0])
    df.to_csv(os.path.join(path,("PredictionsMaxMin.csv")))
    print(df)

    #store the predictions
    df = pd.DataFrame(predictionAvgScore, columns = columnName, index = data2.axes[0])
    df.to_csv(os.path.join(path,("PredictionsMaxAVG.csv")))
    print(df)

    #stores the fitted data in a csv
    dfPCA = pd.DataFrame(projectedPCA,index = data2.axes[0])
    dfPCA.to_csv(os.path.join(path,("PCAofFittedData.csv")))


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