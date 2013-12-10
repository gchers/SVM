SVM
===

This code was part of a coursework. The function SVM trains a Support Vector Machine on the training set (*XTrain*, *YTrain*), using a Vapnik's polynomial kernel of degree *degree*, and tests it against the test set.
The fact that the script had to both train and test the SVM was a requirement of the assignment.
*C* and *threshold* are the constraints of the "alphas", that have to be such that
    threshold < alphas < C - threshold
