from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier



def get_logistic_model(max_iter=200):
    # Makes sure to cover everything
    return LogisticRegression(max_iter=max_iter)

def get_knn_model(n_neighbors=5):
    # Param makes it less sensative to noise
    return KNeighborsClassifier(n_neighbors=n_neighbors)

