print("Importing Packages")

import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

#Import a CSV with formating of 
    #         |ind1|ind2|ind3|ind4| ...|depen1|depen2|...
    #spectra1a| 6.0| 0.4| 0.5| 0.3| ...|    12|    23|...
    #spectra1b| 6.1| 0.2| 0.5| 0.2| ...|    62|    23|...
    #spectra1c| 6.3| 0.3| 0.3| 0.4| ...|    52|    22|...
    #spectra2a| 6.2| 0.5| 0.3| 0.2| ...|    32|    25|...
    #spectra2b| 6.2| 0.5| 0.3| 0.2| ...|    12|    26|...
    #spectra2c| 6.2| 0.5| 0.3| 0.2| ...|    32|    27|...
    #spectra3a| 6.2| 0.5| 0.3| 0.2| ...|    22|    19|...

def PLSR():
    print("\nPlese enter the full path to a file containing the data you would like to train the PCA on")
    print("An exmaple of a format is C:\\Users\\Documents\\FolderName\\FileContainingTrainingData.csv")
    file = input("Enter here: ")

    print("Importing Data")
    try:
            data = pd.read_csv(file, header = 0, index_col =0)
    except:
        print("Please ensure that you entered the file path properly")
        return
        #return

    #randomises the order of the data
    data = data.reindex(np.random.permutation(data.index))

    while(True):
        #Get fromm user, ensure that they add up to 100
        print("\nYou need to split your data into Training, Validation, and test data")
        print("Plese enter the a value between 0 and 100 representing how much of your data you would like to be Training data")
        percentage_train = int(input("Enter here: "))
        while(percentage_train>=100 or percentage_train<=0):
            print("\nPlease ensure that you have entered a value between 0 and 100")
            percentage_train = int(input("Enter here: "))

        print("\nPlese enter the a value between 0 and "+ str(100-percentage_train)+ " representing how much of your data you would like to be Validation data")
        percentage_validation = int(input("Enter here: "))
        while(percentage_validation>100-percentage_train or percentage_validation<=0):
            print("\nPlease ensure that you have entered a value between 0 and "+ str(100-percentage_train))
            percentage_validation = int(input("Enter here: "))

        print("\nPlese enter the a value between 0 and "+ str(100-percentage_train- percentage_validation)+ " representing how much of your data you would like to be Test data")
        percentage_test = int(input("Enter here: "))
        while(percentage_test>100-percentage_train-percentage_validation or percentage_test<0):
            print("\nPlease ensure that you have entered a value between 0 and "+ str(100-percentage_train- percentage_validation))
            percentage_test = int(input("Enter here: "))

        print("\nPlease eneter how many dependent variables are at the end of the file")
        num_depen_var = int(input("Enter here: "))

        while(num_depen_var<=0 or num_depen_var>= len(data.axes[0])):
            print("Please ensure that the number of dependent variables is not greater than ")
            num_depen_var = int(input("Enter here: "))


        train_cut_off = (int)(len(data) * percentage_train/100)
        val_cut_off = (int)(len(data) * (percentage_train+percentage_validation)/100)


        #Split into training, validation, and testing data
        train = data.iloc[0:train_cut_off,:]
        val = data.iloc[train_cut_off:val_cut_off,:]
        test = data.iloc[val_cut_off:len(data),:]


        # Split the columns in X and Y
        X_colnames = data.columns[:-num_depen_var]
        Y_colnames = data.columns[-num_depen_var:]

        # Split each train, val and test into two arrays
        X_train = train[X_colnames].values
        Y_train = train[Y_colnames].values

        X_val = val[X_colnames].values
        Y_val = val[Y_colnames].values

        X_test = test[X_colnames].values
        Y_test = test[Y_colnames].values

        #Find the best number of components
        alphaValues = np.arange(0.1,1000,0.1)
        
        my_ridge = Ridge(alpha=alphaValues[0])
        my_ridge.fit(X_train, Y_train)
        preds = my_ridge.predict(X_val)

        best_r2 = r2_score(preds, Y_val)
        best_alpha = alphaValues[0]
        
        for i in alphaValues:
            my_ridge = Ridge(alpha = i)
            my_ridge.fit(X_train,Y_train)
            preds = my_ridge.predict(X_val)

            r2 = r2_score(preds, Y_val)
            if r2 > best_r2:
                best_r2 = r2
                best_alpha = i

        print("The Best r2 score was: " + str(best_r2))
        print("The alpha value for best r2 score was: " + str(best_alpha))

        #Build a PLSR model with the best number of components
        best_model = Ridge(alpha=best_alpha)
        best_model.fit(X_train, Y_train)

        #Predict the value of the test data off the best model
        if(len(X_test) != 0):
            preds = best_model.predict(X_test)

            #store the predicted model in a data frame
            df = pd.DataFrame(Y_test,columns = data.axes[1][-num_depen_var:], index=data.axes[0][val_cut_off:len(data)])
            df = df.add_prefix('Actual_')
            dfPred = pd.DataFrame(preds, columns=data.axes[1][-num_depen_var:], index=data.axes[0][val_cut_off:len(data)])
            dfPred = dfPred.add_prefix('Predited_')

            concatanation = pd.concat([df,dfPred], axis = 1)

            folderPathway = os.path.dirname(file)
            fileName = os.path.basename(file)
            concatanation.to_csv(os.path.join(folderPathway,("Ridge_Predictions_"+fileName)))
            print(concatanation)
        else:
            print("\nThere is no testing data")

        print("\nDo you want to change the ratio of training:validation:testing or retrain the model")
        choice = input("Please enter 'Y' or 'N': ")
        while not(choice.upper() == "Y" or choice.upper() == "N"):
            choice = input("Please enter 'Y' or 'N': ")

        if(choice.upper() == "N"):
            break


    print("\nUsing this PLS model do you want to choose some data to predict?")

    choice = input("Please enter 'Y' or 'N': ")
    while not(choice.upper() == "Y" or choice.upper() == "N"):
        choice = input("Please enter 'Y' or 'N': ")

    if(choice.upper() == "N"):
        return

    #predicts the values of data from the PLS model
    while(True):
        print("\nPlese enter the full path to a file containing the data you would like to fit")
        print("Please ensure that the file is formatted in the following manner:")
        print("\n       |ind1|ind2|ind3|ind4| ..."
            + "\nsample1| 6.0| 0.4| 0.5| 0.3| ..."
            + "\nsample2| 6.1| 0.3| 0.5| 0.2| ..."
            + "\nsample3| 6.3| 0.5| 0.3| 0.4| ..."
            + "\nsample4| 6.2| 0.6| 0.3| 0.2| ..."
            + "\nsample5| 7.2| 0.6| 0.3| 1.2| ..."
            + "\nsample6| 7.1| 0.4| 0.3| 1.3| ..."
            + "\nsample7| 6.2| 2.6| 4.3| 3.1| ..."
            + "\nsample8| 6.3| 2.5| 4.5| 3.2| ...")
        print("An exmaple of a format is C:\\Users\\Documents\\FolderName\\FileContainingFittingData.csv")
        file2 = input("Enter here: ")

        #Import the new data
        try:
            data2 = pd.read_csv(file2, header = 0, index_col =0)
        except:
            print("Please ensure that you entered the file path properly")
            return

        #Ensure that the training data and prediction data have the same number of independent variables
        if (len(data.axes[1])-num_depen_var != len(data2.axes[1])):
            print("Please ensure that the number of independent variables in the training and predicting file are the same")
            return

        #Get the data
        X_colnames2 = data2.columns
        All_Samples = data2.iloc[0:len(data2),:]
        X = All_Samples[X_colnames2].values

        #predict the value of the dependent variables using the model
        Y = best_model.predict(X)

        #store the predictions in a panda data frame
        dfPred2 = pd.DataFrame(Y, columns=data.axes[1][-num_depen_var:], index=data2.axes[0])
        dfPred2 = dfPred2.add_prefix('Predited_')

        #store the predictions in a csv
        folderPathway = os.path.dirname(file)
        fileName2 = os.path.basename(file2)
        dfPred2.to_csv(os.path.join(folderPathway,("PLSR_Predictions_"+fileName2)))

        print("Do you want to predict more data with this model")

        choice = input("Please enter 'Y' or 'N': ")
        while (choice.upper() != "Y" or choice.upper() != "N"):
            choice = input("Please enter 'Y' or 'N': ")

        if(choice.upper() == "N"):
            return

def Main():
    print("\nThis program will perform PCA on a set of spectra and fit another set of spectra to the same space"
         +"\nThe program will take two 'csv's please ensure that they are formatted as shown below")
    print("\n       |ind1|ind2|ind3|ind4| ... |depen1|..."
        + "\nsample1| 6.0| 0.4| 0.5| 0.3| ... |   0.5|..."
        + "\nsample2| 6.1| 0.3| 0.5| 0.2| ... |   0.7|..."
        + "\nsample3| 6.3| 0.5| 0.3| 0.4| ... |   0.4|..."
        + "\nsample4| 6.2| 0.6| 0.3| 0.2| ... |   0.2|..."
        + "\nsample5| 7.2| 0.6| 0.3| 1.2| ... |   0.2|..."
        + "\nsample6| 7.1| 0.4| 0.3| 1.3| ... |   0.3|..."
        + "\nsample7| 6.2| 2.6| 4.3| 3.1| ... |   0.4|..."
        + "\nsample8| 6.3| 2.5| 4.5| 3.2| ... |   0.7|...")
    
    PLSR()

    while(True):
        print("\nAre you finished performing PLSRs?")
        stop = input("Please enter 'Y' or 'N': ")

        if(stop.upper() == "Y"):
            return
        #end if

        PLSR()
    #end while
#end Main

Main()