# Raman Spectra Analysis
 Programs I initially developed in my summer internship in 2023 to help me analyse Raman spectra, all are usable outside Raman spectra analysis. These tools have seen further additions and adaptions during my PhD.

## PCA with inbuilt LOOCV.py: 
Used to perform a PCA with LOOCV on a file of data

Inputs: A single file with multiple sets of data from multiple samples, data from the same sample should have the same name

Outputs: A folder with the eigenvectors for each PCA created, an image of the PCA explained variance to allow the user to choose the ideal number of PCs, and a csv storing the scores of the LOOCV

The input file must be a CSV, formatted such that the independent variable is the first row, then each sample is a row beneath. The first column should be the sample names. To perform the LOOCV we require at least two sets of data for each sample, with the same sample name in the CSV. The output score file is formatted such that the columns are the samples and the rows the predictions. A perfect LOOCV would give a diagonal matrix. 

## PCA and Pearson with inbuilt LOOCV.py: 
Used to perform a PCA and Pearson correlation classification with LOOCV on a file of data

Inputs: A single file with multiple sets of data from multiple samples, data from the same sample should have the same name

Outputs: A folder with all data used to make the classifications, and two CSVs storing the scores of the LOOCV using each method

The input file must be a CSV, formatted such that the independent variable is the first row, then each sample is a row beneath. The first column should be the sample names. To perform the LOOCV we require at least two sets of data for each sample, with the same sample name in the CSV. The output score file is formatted such that the columns are the samples and the rows the predictions. A perfect LOOCV would give a diagonal matrix. 

## PCA_to_predict.py:
Used to build a PCA on a set of training data and then fit more data to the PCA and predict what the fitted data is when compared to the training data

Inputs: A single CSV file with multiple sets of data from multiple samples, data from the same sample should have the same name

Outputs: A folder with the eigenvectors for the PCA, an image of the PA explained variance to allow the user to choose the ideal number of PCs, the trained and fitted data transformed to the PC space in separate CSVs, and multiple CSVs storing the predictions for the fitted data. Scoring for the predictions is made using my maximum minimum method.
 

## Partial Least Squares.py: 
Used to build a PLSR model and predict the values of a given dependent variable

Inputs: Model building CSV file, and an arbitrary number of CSVs to predict the values of dependent variables using the model

Outputs: CSVs with the predictions from the testing of the model building data, and a CSV for each file that the user wishes to use the model to predict dependent variables

The initial model building input file must be a CSV, formatted such that the independent variable is the first row, then each sample is a row beneath. The dependent variables must be at the end of the file and stated for each sample. Each sample must have a different name. The order of the samples does not matter as the program will randomise it. The program will ask the user to split the initial model building data into training, validation, and testing data by asking for percentages of the initial data to be each. The program will build the model and ask the user if they wish to use the model or retrain it. This allows the user to change the ratio of training, validation, and testing or rerun the model with the same ratio, but a different random ordering of the data.
The program will then ask the user if they wish to use this model to predict the values of dependent variables from new data. This new data must be CSV and must be formatted in the same manner as the model building data, except it must not have the dependent variables. The user can give as many files as they like, it must be noted that once they exit this part the model cannot be retained and must instead be built from scratch.
