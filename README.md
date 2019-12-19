# Comparison of learners' profermance on semeion handwritten digit
This project is to study the prefermance of different learners (KNN, SVM, and CNN) on predicting
handwritten digits.

## Data reference
Semeion Research Center of Sciences of Communication, via Sersale 117, 
00128 Rome, Italy
Tattile Via Gaetano Donizetti, 1-3-5,25030 Mairano (Brescia), Italy. 

## Lab notebook

Dec 18, 2019 -- Amberley & Fiona
1. Complete slides for presentation.

Dec 17, 2019 -- Amberley
1. Add more test data of our own.
2. Draw confusion matrix for validation data and our own data.
3. Analyze the reasons for different performance of three models.

Dec 17, 2019 -- Fiona
1. Created methods to plot confusion matrix in forms of heatmap for visualization.
2. Work on slides for presentation.

Dec 16, 2019 -- Amberley
1. Test the best epoch for cnn.
2. See the relationship of train and validation accuracy in the long run.
3. Test cnn model with our own data.

Dec 16, 2019 -- Fiona
1. Created methods to reading csv file.

Dec 14, 2019 -- Amberley
1. Put our daraset in CNN model, see how well it performs.
2. Get test data ready, create a csv file to store test dataset.

Dec 14, 2019 -- Fiona:
1. Met with Amberley, and set up the CNN model togother.

Dec 13, 2019 -- Fiona:
1. Created methods for pulling out wrong perdictions and transformed those into images for visualization.
2. Collected own handwritten digits, and pre-processed those images into the input type we need.

Dec 9, 2019 -- Fiona:
1. Set up a random seed (42) for numpy shuffle in util in order to generate reproducable 
training and testing datasets.
2. Drew a graph for SVM's performance with different hyperparmeters (C and gamma), 
and set C=100 and gamma=0.01 for further study.
3. Built confusion matrices for KNN and SVM, found SVM performed better than KNN generally,
but making the same type of mistakes (e.g. they were both relatively weak on predicting digit 9 with 4).

Dec 5, 2019 -- Fiona:
1. Set up data-preprocessing for KNN, SVM and CNN methods.
2. Drew a graph for KNN's performance with different hyperparmeters (number of neighbors and weights),
and chose 5 as the number of neighbors, and distance as weight for further study.