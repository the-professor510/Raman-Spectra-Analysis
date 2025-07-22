# This python file will use scikitlearns libraries to analyse a set of data 
# by PCA and performing Leave One Out Cross Validation

print("Importing Libraries")
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

#returns an evaluation score when considering just one principal component
#is passed only one value for average, std, and fitted
def scoreOnePC(average, std, fitted):
    
    score = ((fitted-average)**2)
    score = (-1)*score/(2*(std**2))
    score = np.exp(score)

    return score

#gets a file from the user on which PAC with LOOCV will be performed
def PCAWithLOOCV():
    #Import a CSV with formating of 
    #         |ind1|ind2|ind3|ind4| ...
    #spectra1a| 6.0| 0.4| 0.5| 0.3| ...
    #spectra1b| 6.1| 0.2| 0.5| 0.2| ...
    #spectra1c| 6.3| 0.3| 0.3| 0.4| ...
    #spectra2a| 6.2| 0.5| 0.3| 0.2| ...
    #spectra2b| 6.2| 0.5| 0.3| 0.2| ...
    #spectra2c| 6.2| 0.5| 0.3| 0.2| ...
    #spectra3a| 6.2| 0.5| 0.3| 0.2| ...


    print("\nPlese enter the full path to a file containing the data you would like to perform Leave One Out Cross Validation (LOOCV) on")
    print("An exmaple of a format is C:\\Users\\Documents\\FolderName\\FileContainingData.csv")
    file = input("Enter here: ")

    #reads in the file to a panda data frame
    try:
        data = pd.read_csv(file, header = 0, index_col =0)
    except:
        print("Please ensure that you entered the file path properly")
        return

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

    columnName = sampleNames.copy()
    columnName.append("Unknown")

    # create a data frame to store the loocv
    df = pd.DataFrame(columns = columnName,
                    index = sampleNames)
    df = df.fillna(0)


    #create a folder to store the Eigenvalues and loocv
    folderPathway = os.path.dirname(file)
    fileName = os.path.basename(file)


    print("\nPlese enter the name of the directory you would like to store the LOOCV in")
    print("An exmaple of a is \"PCAwithLOOCV\"")
    directory = input("Enter here: ")

    
    # Path
    path = os.path.join(folderPathway, directory)

    #attempts to create a new directory with name specfied by the user
    try:
        os.mkdir(path)
    except:
        print("File already exists, do you want to override it?")
        choice = input("Please enter 'Y' or 'N': ")

        if(choice.upper() != "Y"):
            return

    # store an image of the explained variance to allow the user to choose the right number of components
    plt.plot(np.cumsum((PCA(n_components=min(len(data), len(X_colnames))).fit(X)).explained_variance_ratio_))
    plt.xlabel("Number of components")
    plt.ylabel("Explained variance")
    plt.savefig(path + r"\PCA Explained variance.png")

    #get the number of components from the user
    numComponents= int(input("Enter the number of Prinicpal Components you would like to use: "))

    #get the number of components from the user
    UncertaintyScore = float(input("Enter the minimum score you would like to have to not be predicted as unknown: "))

    #loops through all spectra to perfrom PCA with LOOCV
    for i in range(len(data)):

        #figures out which sample we are missing
        y = -1
        for k in range(len(sampleNames)):
            if sampleNames[k] == data.axes[0][i]:
                y = k
                #numberSamples[k] += 1
                break
        
        #get a new PCA missing the ith spectra in the given data
        createdList =createList(i,0,len(data)-1)
        X_For_PCA = X[createdList,:]
        
        #create the PCA
        pca = PCA(n_components=numComponents).fit(X_For_PCA)

        #get the Principal components for the training data
        X_Transformed = pca.transform(X_For_PCA)
        dtPCA = pd.DataFrame(X_Transformed,index = data.axes[0][createdList])

        #store Prinicple Components of the training data in a csv
        trainingPath = os.path.join(path, "TrainingPCs")
        try:
            os.mkdir(trainingPath)
        except:
            #Do Notihng
            doNothing = 0
        fileName = "TrainingPrincipalComponentsMissing" + str(sampleNames[y]) + "(" + str(numberSamples[y]) + ").csv"
        filePath = os.path.join(trainingPath,fileName)
        dtPCA.to_csv(filePath)

        #find the average position and standard deviation of each smaple in the PC space
        average = []
        std = []
        for j in range(len(sampleNames)):
            #find index of all scores relating to first name
            index = []

            for k in createdList:
                if sampleNames[j] == data.axes[0][k]:
                    index.append(k)

            total = []
            subtraction = 0

            if index[0] >= i:
                subtraction = 1

            total = X_Transformed[index[0] - subtraction]

            #get list to find std of
            listOfPC = X_Transformed[index[0] - subtraction]
            
            for k in index[1:]:
                if (k >= i):
                    subtraction = 1

                listOfPC = np.column_stack((listOfPC, X_Transformed[k-subtraction]))

                for l in range(numComponents):
                    total[l] += X_Transformed[k-subtraction,l]


            #calculate average
            for k in range(numComponents):
                total[k] = total[k]/len(index)

            average.append(total.copy())

            #calculate std
            std.append(np.std(listOfPC, axis =1))
        
        #store average PC values and standard deviations
        dfAverages = pd.DataFrame(average,index = sampleNames)
        dfSTD = pd.DataFrame(std, index = sampleNames)

        averagePath = os.path.join(path, "AvgOfPCs")
        stdPath = os.path.join(path, "STDofPCs")

        try:
            os.mkdir(averagePath)
        except:
            #Do Notihng
            doNothing = 0
        try:
            os.mkdir(stdPath)
        except:
            #Do Notihng
            doNothing = 0

        #store the averages of the Principle components of the training data
        fileName = "AveragePCMissing" + str(sampleNames[y]) + "(" + str(numberSamples[y]) +").csv"
        filePath = os.path.join(averagePath,fileName)
        dfAverages.to_csv(filePath)
        #store the standard deviations of the principal components of the training data
        fileName = "STDPCMissing" + str(sampleNames[y]) + "(" + str(numberSamples[y]) +").csv"
        filePath = os.path.join(stdPath,fileName)
        dfSTD.to_csv(filePath)



        #fit the missing spectra to the PC space
        TransformMissing = pca.transform(X[i,:].reshape(1, -1))
        dfPCA = pd.DataFrame(TransformMissing,index = [sampleNames[y]])

        fittingPath = os.path.join(path, "FittingPCs")
        try:
            os.mkdir(fittingPath)
        except:
            #Do Notihng
            doNothing = 0

        #store Prinicple Components of the fitted data in a csv
        fileName = "FittedPrincipalComponentsMissing" + str(sampleNames[y]) + "(" + str(numberSamples[y]) + ").csv"
        filePath = os.path.join(fittingPath,fileName)
        dfPCA.to_csv(filePath)
        
        scorePC = np.array(scoreOnePC(average[0][0], std[0][0], TransformMissing[0][0]))
        for k in range(1,len(average)):
            scorePC = np.append(scorePC, scoreOnePC(average[k][0], std[k][0], TransformMissing[0][0]))

        scores = np.array(scorePC)
        scoreColNames = [("PC" + str(0))]

        #find the probability scores when considering each axis individually
        for j in range(1,numComponents):
            #calculate the socres due to the jth principle component
            scorePC = np.array(scoreOnePC(average[0][j], std[0][j], TransformMissing[0][j]))
            for k in range(1,len(average)):
                scorePC = np.append(scorePC, scoreOnePC(average[k][j], std[k][j], TransformMissing[0][j]))

            scores = np.column_stack((scores, scorePC))
            scoreColNames += [("PC" + str(j))]

        minScores = np.min(scores, axis=1)
        scores = np.column_stack((scores, minScores))
        scoreColNames += [("Minimum Confidence score")]

        index = 0
        maxMinScore = minScores[0]
        for j in range(1,len(minScores)):
            if maxMinScore < minScores[j]:
                maxMinScore = minScores[j]
                index = j

        if maxMinScore < UncertaintyScore:
            index = len(sampleNames)


        dfscores = pd.DataFrame(scores, columns = scoreColNames, index = sampleNames)
        scoresPath = os.path.join(path, "ScoresOfPCs")
        try:
            os.mkdir(scoresPath)
        except:
            #Do Notihng
            doNothing = 0
        #store the scores of the principal components of the training data
        fileName = "ScoresMissing" + str(sampleNames[y]) + "(" + str(numberSamples[y]) +").csv"
        filePath = os.path.join(scoresPath,fileName)
        dfscores.to_csv(filePath)

    

        #get the correct column and row to add a point to the LOOCV table
        y = -1
        for k in range(len(sampleNames)):
            if sampleNames[k] == data.axes[0][i]:
                y = k
                numberSamples[k] += 1
                break
        x = index

        df.iat[y,x] += 1
        #[row,column]

        #save the data
        # Create folders to better organise the data being saved
        eigenPath = os.path.join(path, "EigenVectors")
    
        #attempts to create a new directory with name specfied by the user
        try:
            os.mkdir(eigenPath)
        except:
            #Do Notihng
            doNothing = 0
    


        #store the eigenvectors for the pca
        fileName = "eigenVectorsMissing" + str(sampleNames[y]) + "(" + str(numberSamples[y]) +").csv"
        filePath = os.path.join(eigenPath,fileName)
        
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
    df.to_csv(os.path.join(path,("LOOCVTable.csv")))
    return()



def Main():
    print("\nThis program will perform PCA and LOOCV on a set of data in a csv"
        + "\nPlease ensure that the format is formatted as shown below")
    
    print("\n       |ind1|ind2|ind3|ind4| ..."
        + "\nsample1| 6.0| 0.4| 0.5| 0.3| ..."
        + "\nsample1| 6.1| 0.3| 0.5| 0.2| ..."
        + "\nsample2| 6.3| 0.5| 0.3| 0.4| ..."
        + "\nsample2| 6.2| 0.6| 0.3| 0.2| ..."
        + "\nsample3| 7.2| 0.6| 0.3| 1.2| ..."
        + "\nsample3| 7.1| 0.4| 0.3| 1.3| ..."
        + "\nsample4| 6.2| 2.6| 4.3| 3.1| ..."
        + "\nsample4| 6.3| 2.5| 4.5| 3.2| ...")
    
    PCAWithLOOCV()

    while(True):
        print("\nAre you finished performing PCAs?")
        stop = input("Please enter 'Y' or 'N': ")

        if(stop.upper() == "Y"):
            return
        #end if

        PCAWithLOOCV()
    #end while
#end Main

Main()