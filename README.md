# Comparison of learners' profermance on semeion handwritten digit
## Lab notebook
Dec 9, 2019 -- Fiona:

1. Set up a random seed (42) for numpy shuffle in util in order to generate reproducable 
training and testing datasets.
2. Drew a graph for SVM's performance with different hyperparmeters (C and gamma), 
and set C=100 and gamma=0.01 for further study.
3. Built confusion matrices for KNN and SVM, found SVM performed better than KNN generally,
but making the same type of mistakes (e.g. they were both relatively weak when predicting digit 9).

Dec 5, 2019 -- Fiona:

1. Set up data-preprocessing for KNN, SVM and CNN methods.
2. Drew a graph for KNN's performance with different hyperparmeters (number of neighbors and weights),
and chose 5 as the number of neighbors, and distance as weight for further study.