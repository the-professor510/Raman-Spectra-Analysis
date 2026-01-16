# This python file will use scikitlearns libraries to analyse a set of data 
# by PCA and performing Leave One Out Cross Validation

print("Importing Libraries")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats
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


#gets a file from the user on which PAC with LOOCV will be performed
def PearsonWithLOOCV():
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
    dfPCALOOCVTable = pd.DataFrame(columns = columnName,
                    index = sampleNames)
    dfPCALOOCVTable = dfPCALOOCVTable.fillna(0)
    
    # create a data frame to store the loocv
    dfPearsonLOOCVTable = pd.DataFrame(columns = columnName,
                    index = sampleNames)
    dfPearsonLOOCVTable = dfPearsonLOOCVTable.fillna(0)


    #create a folder to store the Eigenvalues and loocv
    folderPathway = os.path.dirname(file)
    fileName = os.path.basename(file)


    print("\nPlese enter the name of the directory you would like to store the LOOCV in")
    print("An exmaple of a is \"PearsonWithLOOCV\"")
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

    #get the number of components from the user
    numComponents= int(input("Enter the number of Prinicpal Components you would like to use: "))
    while numComponents > len(sampleNames):
        print(f"Require that number of components is less than or equal to the number of different samples({len(sampleNames)})")
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
                numberSamples[k] += 1
                break
        
        #get a new data set with the missing the ith spectra in the given data
        createdList =createList(y,0,len(data)-1)
        X_For_Pearson = np.array(X[createdList,:])
        Labels_For_Pearson = np.array(data.index[createdList])
        
        ##################################
        ### FIND AVERGE OF EACH SAMPLE ###
        ##################################
        
        AverageSpectra = []
        StdSpectra = []
        for k in range(len(sampleNames)):
            array = []
            for l in range(len(createdList)):
                if Labels_For_Pearson[l] == sampleNames[k]:
                    array.append(X_For_Pearson[l])
            
            average = np.average(array, axis=0)
            std = np.std(array, axis=0)
            
            AverageSpectra.append(average)
            StdSpectra.append(std)
        
        
        
        
        
        ###########################################
        ### FIND THE AVERAGE STD OF EACH SAMPLE ###
        ###########################################
        
        fittedAvg = []
        fittedStd = []
        fittedPearsonStd = []
        #perform PCA on the averages
        pca = PCA(n_components=numComponents).fit(AverageSpectra)       
        
        # find the average and std of the cluster of each sample
        for index, sample in enumerate(sampleNames):    
            tempListForAvgandStd = []
            tempListForPearsonStd = []
            
            for label, spectra in zip(Labels_For_Pearson, X_For_Pearson):
                if label == sample:

                    transformed = pca.transform([spectra])
                    
                    pearson = stats.pearsonr(spectra, AverageSpectra[index])
                    pearsonScore = pearson.statistic
                    
                    #append to temporary list for average and standard deviation
                    if len(tempListForAvgandStd) != 0:
                        tempListForAvgandStd = np.append(tempListForAvgandStd, transformed, axis = 0)
                        tempListForPearsonStd = np.append(tempListForPearsonStd, pearsonScore)

                    else:
                        tempListForAvgandStd = transformed
                        tempListForPearsonStd = [pearsonScore]
            #end for
            
            #calculate the new average and std for each sample
            avg = [np.average(tempListForAvgandStd, axis = 0)]
            std = [np.std(tempListForAvgandStd, axis = 0)]
                        
            pearsonStd = np.sqrt(np.sum((np.array(tempListForPearsonStd)-1.0)**2))/len(tempListForPearsonStd)            
            
            #append to the list of average and standard deviation for the fitted samples
            if len(fittedAvg) != 0:
                fittedAvg = np.append(fittedAvg, avg, axis = 0)
                fittedStd = np.append(fittedStd, std, axis = 0)
                
                fittedPearsonStd = np.append(fittedPearsonStd, pearsonStd)
            else:
                fittedAvg = avg
                fittedStd = std
                
                fittedPearsonStd = pearsonStd
        #end for
                
        #store the mean of the training data pre-fitting
        meanPreFitting = np.array([pca.mean_])
        #store the new eigenvectors
        eigenvectors = pca.components_    
        
        
        
        
        ####################
        ### PCA ANALYSIS ###
        ####################
        
        #transform the unfitted sample
        transformed = np.array(X[i,:])
        transformed = transformed - meanPreFitting[0]
        transformed = np.matmul(eigenvectors, transformed)
        
        minScores = np.array([])
        avgScores = np.array([])
        pcaScores = np.array([])
        np.seterr(divide = "raise") # set numpy to raise divide by zero errors
        for average, std in zip(fittedAvg, fittedStd):
            
            
            score = (transformed-average)**2
            try:
                score = (-1)*score/(2*(std**2))
            except FloatingPointError:
                score = (-1)*score/1e-4
            
                
            score = np.exp(score)
            
            pcaScores = np.append(pcaScores, score)
            minScores = np.append(minScores, min(score))
            avgScores = np.append(avgScores, np.average(score))
        #end for
        np.seterr(divide="warn") #set numpy to only warn divide by zero errors
        pcaScores = pcaScores.reshape((len(sampleNames), numComponents))
        
        indexOfBestPCA = np.argmax(minScores)
        
        
        
        
        
        ####################################
        ### PEARSON CORRELATION ANALYSIS ###
        ####################################
        
        pearsonCorr = np.array([])
        confidenceScore = np.array([])
        np.seterr(divide = "raise") # set numpy to raise divide by zero errors
        for average, std in zip(AverageSpectra, fittedPearsonStd):
            
            #calculate the pearson correlation between y_data spectrum and the average            
            pearson = stats.pearsonr(np.array(X[i,:]), average)
            pearsonCorr = np.append(pearsonCorr, pearson.statistic)
            
                        
            score = (pearsonCorr[-1] - 1.0)**2
            try:
                score = (-1)*score/(2*(std**2))
            except FloatingPointError:
                score = (-1)*score/1e-4
            
            score = np.exp(score)
            confidenceScore = np.append(confidenceScore, score)
            
        #end for
        np.seterr(divide="warn") #set numpy to only warn divide by zero errors

        indexOfBestCorr = np.argmax(pearsonCorr)
        
        
        #store the predictions in the LOOCV tables
        if np.max(minScores) < UncertaintyScore:
            indexOfBestPCA = len(sampleNames)
        if confidenceScore[indexOfBestCorr] < UncertaintyScore:
            indexOfBestCorr = len(sampleNames)
        dfPCALOOCVTable.iat[y,indexOfBestPCA] += 1
        dfPearsonLOOCVTable.iat[y,indexOfBestCorr] += 1
        #[row,column]
        
        
        
        ###################################
        ### STORE ALL THE RELEVANT DATA ###
        ###################################
        
        dtPCA = pd.DataFrame(AverageSpectra ,index = sampleNames)

        #store the training data in a csv
        trainingPath = os.path.join(path, "AvgTrainingData")
        try:
            os.mkdir(trainingPath)
        except:
            #Do Nothing
            doNothing = 0
        fileName = "ATDataMissing" + str(sampleNames[y]) + "(" + str(numberSamples[y]) + ").csv"
        filePath = os.path.join(trainingPath,fileName)
        dtPCA.to_csv(filePath)


        avgTransformed = pca.transform(AverageSpectra)
        dtPCA = pd.DataFrame(avgTransformed,index = sampleNames)
        #store the transformed training data in a csv
        trainingPath = os.path.join(path, "TransformedTrainingData")
        try:
            os.mkdir(trainingPath)
        except:
            #Do Nothing
            doNothing = 0
        fileName = "TTDataMissing" + str(sampleNames[y]) + "(" + str(numberSamples[y]) + ").csv"
        filePath = os.path.join(trainingPath,fileName)
        dtPCA.to_csv(filePath)

        
        #store average PC values and standard deviations
        dfAverages = pd.DataFrame(fittedAvg,index = sampleNames)
        dfSTD = pd.DataFrame(fittedStd, index = sampleNames)

        averagePath = os.path.join(path, "AvgOfPCs")
        stdPath = os.path.join(path, "STDofPCs")

        try:
            os.mkdir(averagePath)
        except:
            #Do Nothing
            doNothing = 0
        try:
            os.mkdir(stdPath)
        except:
            #Do Nothing
            doNothing = 0

        #store the averages of the Principle components of the training data
        fileName = "AveragePCMissing" + str(sampleNames[y]) + "(" + str(numberSamples[y]) +").csv"
        filePath = os.path.join(averagePath,fileName)
        dfAverages.to_csv(filePath)
        #store the standard deviations of the principal components of the training data
        fileName = "STDPCMissing" + str(sampleNames[y]) + "(" + str(numberSamples[y]) +").csv"
        filePath = os.path.join(stdPath,fileName)
        dfSTD.to_csv(filePath)



        #store the transformed fitted data in a csv
        dfPCA = pd.DataFrame(transformed.reshape(1,-1), index = [sampleNames[y]])
        fittingPath = os.path.join(path, "FittingPCs")
        try:
            os.mkdir(fittingPath)
        except:
            #Do Notihng
            doNothing = 0

        fileName = "FittedPrincipalComponentsMissing" + str(sampleNames[y]) + "(" + str(numberSamples[y]) + ").csv"
        filePath = os.path.join(fittingPath,fileName)
        dfPCA.to_csv(filePath)
        
        
        

        #score the confidence values for the pca data
        scoreColNames = [("PC" + str(0))]
        for j in range(1,numComponents):
            scoreColNames += [("PC" + str(j))]
        scoreColNames += [("Minimum Confidence score")]
        pcaScores = np.column_stack((pcaScores,minScores))
        dfscores = pd.DataFrame(pcaScores, columns = scoreColNames, index = sampleNames)
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

    
        
        #store the pearson std, correlation scores and confidence values
        scoreColNames = ["stdPearsonFitted", "PearsonCorrelationScore", "PearsonCorrelationConfidence"]
        pearsonData = np.array([fittedPearsonStd, pearsonCorr, confidenceScore]).T
        dfscores = pd.DataFrame(pearsonData, columns = scoreColNames, index = sampleNames)
        scoresPath = os.path.join(path, "PearsonData")
        try:
            os.mkdir(scoresPath)
        except:
            #Do Notihng
            doNothing = 0
        #store the scores of the principal components of the training data
        fileName = "ScoresMissing" + str(sampleNames[y]) + "(" + str(numberSamples[y]) +").csv"
        filePath = os.path.join(scoresPath,fileName)
        dfscores.to_csv(filePath)
    

        #save the data
        #Create folders to better organise the data being saved
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
    print("PCA LOOCV Table")
    print(dfPCALOOCVTable)
    dfPCALOOCVTable.to_csv(os.path.join(path,("PCALOOCVTable.csv")))
    print("\n\nPearson LOOCV Table")
    print(dfPearsonLOOCVTable)
    dfPearsonLOOCVTable.to_csv(os.path.join(path,("PearsonLOOCVTable.csv")))
    return()



def Main():
    print("\nThis program will perform LOOCV on a set in a csv of data using PCA and Pearson Correlation analysis LOOCV"
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
    
    PearsonWithLOOCV()

    while(True):
        print("\nAre you finished performing PCA and Pearson Correlation?")
        stop = input("Please enter 'Y' or 'N': ")

        if(stop.upper() == "Y"):
            return
        #end if

        PearsonWithLOOCV()
    #end while
#end Main

Main()