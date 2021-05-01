############################################################################################
# The following code is the Project 2: Machine Learning Mine Versus Rock; which implements #
# principal component analysis and multi-level perceptron to determine the best batch of   #
# features (components) that provides with the maximum accuracy model in differentiating   #
# between mines (hollow pipe) and surrounding rocks.                                       #
############################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split                # for splitting training and testing data
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier                    # to implement the algorithm
from sklearn.decomposition import PCA                               # to implement Principal Component Analysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix                        # to determine the confusion matrix
import warnings                                                     # to suppress warnings

warnings.filterwarnings('ignore')
# to suppress/ignore warnings
df = pd.read_csv('..\\resources\\sonar_all_data_2.csv',header=None)                # reading given dataset file 'sonar_all_data_2.csv'
X, Y = df.iloc[:,:60].values, df.iloc[:,60].values                  # assigning all features to X; and all labels to Y
xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size=0.3,random_state=0)    # splitting train-test data

sc = StandardScaler()                                               # defining standard scaler
std_xtrain = sc.fit_transform(xtrain)                               # applying transformation to training data
std_xtest = sc.transform(xtest)                                     # applying transformation to test data

num_of_comp, confusion_mat, accuracy_list = list(), list(), list()  # defining lists to store components, confusion matrices and accuracies

# Principal Component Analysis execution to obtain best batch of features,and calculate accuracies and confusion matrix
for index in range(1,61):
    pca = PCA(n_components = index)
    pca_xtrain = pca.fit_transform(std_xtrain)                      # applying transformation to train data
    pca_xtest = pca.transform(std_xtest)                            # applying transformation to test data

# building multi-level perceptron
    mlp = MLPClassifier(hidden_layer_sizes=100, activation='logistic', max_iter=2000, alpha=0.0001, solver='adam', tol=0.0001, learning_rate = 'invscaling')
    mlp.fit(pca_xtrain,ytrain)
    predict_y = mlp.predict(pca_xtest)

# calculate number of components used and test accuracy for using that number of components
    print('\nNumber of components used : ',index)
    accuracy = accuracy_score(ytest, predict_y)
    print('Test accuracy for using ' + str(index) + ' components is %.4f' % accuracy)

    cmat = confusion_matrix(ytest,predict_y)
    num_of_comp.append(index)                                         # appending to number to components
    confusion_mat.append(cmat)
    accuracy_list.append(accuracy)                                    # appending accuracies to accuracy_list

# determine maximum accuracy and the number of components used for the maximum accuracy
max_accuracy = np.max(accuracy_list)                                  # determining maximum accuracy
print('\nThe maximum accuracy is :',max_accuracy)
i_max_acc = accuracy_list.index(max_accuracy)                         # determining number of components used to get maximum accuracy
print('\nNumber of top components considered for maximum accuracy: ',i_max_acc+1)

# plotting the graph of accuracy v/s number of components
plt.plot(np.arange(1,61),accuracy_list)
plt.title('Accuracy v/s Number of Components')                        # title of the plot
plt.xlabel('Number of components')                                    # x-label
plt.ylabel('Accuracy')                                                # y-label
plt.grid()                                                            # plot grid
plt.show()

# obtaining the confusion matrix based on the analysis resulting into maximum accuracy
print('\nThe Confusion Matrix for maximum accuracy is :\n', confusion_mat[i_max_acc])

