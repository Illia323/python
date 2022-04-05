import os    
os.environ['MPLCONFIGDIR'] = "matplot-temp"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def loadAndProcess():
    print('Loading...')
    # load the data...

    data = np.loadtxt('./DeepLearning/handwriting.csv',skiprows=1, unpack=False, delimiter=',')

    y = data[:, 0]
    X = data[:, 1:]
    
    return X, y
    
def buildTrainAndTest(X, y):
    print('Building train and test sets...')
    # create the train and test sets for X and y
    # traning has 67% of the rows and test has 33% of the rows...
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 12)

    ########################################################
    # Add scaling here ....
        # No Scaling:
        # Training...
        # Score on train data ???
        
        # Testing...
        # Score on test data ???

        # Scaling:
        # Training...
        # Score on train data ???
        
        # Testing...
        # Score on test data ???
    ########################################################
   
    return X_train, X_test, y_train, y_test
    
def train(X_train, y_train):
    # train the algorithm...
    print('Training...')

    # lbfgs faster on small datasets, Adam for largers datasets
    #mlp = MLPClassifier(solver = 'lbfgs' , random_state = 42, hidden_layer_sizes = [], activation = 'tanh') #tanh
    mlp = MLPClassifier(solver = 'adam' , random_state = 42, hidden_layer_sizes = [ 100 ], activation = 'tanh') #tanh
    mlp.fit( X_train , y_train )
    score = mlp.score(X_train, y_train)
    
    return mlp, score    
    
def test(mlp, X_test , y_test):
    # test the algorithm...
    print('Testing...')
    return mlp.score(X_test , y_test)

def predict(mlp, sample):
    print('\nPredicting...')
    
    y_classification = mlp.predict([sample])
    return y_classification[0]

def main():
    print("Running Main...")
    X, y = loadAndProcess()

    # summarize data...
    print('X shape: %s' % str(X.shape))
    print('y shape: %s' % str(y.shape))
    print('x min/max= %.2f/%.2f' % (X.min(), X.max()))
    print('y min/max= %.2f/%.2f' % (y.min(), y.max()))
    print('first five rows of X = \n%s' % X[0:6, :])
    print('first 150 rows of y = \n%s' % y[0:150])    

    # plot the fourth and fifth examples
    plt.matshow(X[3].reshape(28,28))
    plt.savefig(r"digit4.png",bbox_inches='tight')    

    plt.matshow(X[4].reshape(28,28))
    plt.savefig(r"digit5.png",bbox_inches='tight')    
    

    X_train, X_test, y_train, y_test = buildTrainAndTest(X, y)
    print("X_train = %s\n" % X_train)
    print("X_test = %s\n" % X_test)
    print("y_train = %s\n" % y_train)
    print("y_test = %s\n" % y_test)
  
    mlp, score = train(X_train, y_train)
    print("Score on train data %s\n" % score)

    score = test(mlp, X_test , y_test)
    print("Score on test data %s\n" % score)

    plt.matshow(X_test[7].reshape(28,28))
    plt.savefig(r"sampleToBePredicted.png",bbox_inches='tight')    
    
    digit = predict(mlp, X_test[7])    
    print('Prediction: %s' % digit)
    