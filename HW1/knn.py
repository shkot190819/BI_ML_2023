import numpy as np
from scipy.spatial.distance import cityblock
import statistics as st

class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=2):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        test_train_matrix = np.empty(shape=(len(X), len(self.train_X)))
        for i in range(len(X)):
            for j in range(len(self.train_X)):
                test_train_matrix[i][j] = cityblock(X[i], self.train_X[j])
        return test_train_matrix
        
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        """
        YOUR CODE IS HERE
        """
        


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        """
        YOUR CODE IS HERE
        """
        pass


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        """
        YOUR CODE IS HERE
        """
        pass


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.empty(n_test, dtype=str)
        for z in range(n_test):
            np.argpartition(distances[z],self.k)[:self.k]
            test_y_probable_class = 0
            for i in np.argpartition(distances[z],self.k)[:self.k]:
                test_y_probable_class += int(self.train_y[i])
            if test_y_probable_class/self.k > 0.5:
                prediction[z] = '1'
            else: 
                prediction[z] = '0'
        return prediction



    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, int)
        for z in range(n_test):
            np.argpartition(distances[z],self.k)[:self.k]
            test_y_probable_class = []
            for i in np.argpartition(distances[z],self.k)[:self.k]:
                test_y_probable_class += [int(self.train_y[i])]
                prediction[z] = st.mode(test_y_probable_class)     
        return prediction
        

