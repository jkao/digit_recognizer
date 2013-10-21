import numpy as np

from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


TRAINING_DATA_PATH = "/Users/jeffkao/Projects/School/CZ4032/train.csv"
TEST_DATA_PATH = "/Users/jeffkao/Projects/School/CZ4032/test.csv"

# Load training & test data into memory
DATA = \
    np.genfromtxt(
        TRAINING_DATA_PATH, dtype=float, delimiter=',', skip_header=1)
TEST = \
    np.genfromtxt(
        TEST_DATA_PATH, dtype=float, delimiter=',', skip_header=1)


class KnnDigitRecognizer(object):
    """
    KNN Algorithm we use is:
      1. Use each pixel as a feature
      2. Scale to a standard scaler (zero mean)
      3. Use PCA to reduce dimensions
      4. Use model against test
      5. Obtain confusion matrix
    """
    def __init__(self, numNeighbors=15, weights='distance', pcaComponents=100):
        self.pca = PCA(n_components=pcaComponents)
        self.std = StandardScaler()
        self.numNeighbors = numNeighbors
        self.weights = weights

    def train(self):
        # Separate features from labels
        #   X = features
        #   Y = labels
        self.dataY = np.array([int(y) for y in DATA[:, 0]])
        self.dataX = DATA[:, 1:]

        # Reduce dimensionality
        self.dataX = self.pca.fit_transform(self.dataX)

        # Scale the features
        self.dataX = self.std.fit_transform(self.dataX)

        # Initialize and train the model
        self.clf = \
            neighbors.KNeighborsClassifier(
                self.numNeighbors,
                weights=self.weights)
        self.clf.fit(self.dataX, self.dataY)

    def test(self, dataPath=TEST_DATA_PATH):
        # Separate features from labels
        #   X = features
        #   Y = labels
        self.testY = np.array([int(y) for y in TEST[:, 0]])
        self.testX = TEST[:, 1:]

        # Reduce dimensionality and scale
        self.testX = self.pca.transform(self.testX)
        self.testX = self.std.transform(self.testX)

        yPred = self.clf.predict(self.testX)

        # Return % Accuracy and Confusion Matrix
        return [accuracy_score(self.testY, yPred),
                confusion_matrix(self.testY, yPred)]

if __name__ == "__main__":
    print("START")
    knn = KnnDigitRecognizer(
        numNeighbors=15, weights='distance', pcaComponents=80)
    print("TRAINING")
    knn.train()
    print("TESTING")
    acc, cm = knn.test()
    print(acc)
    print(cm)
    print("DONE")
