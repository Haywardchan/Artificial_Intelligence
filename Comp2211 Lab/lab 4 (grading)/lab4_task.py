import numpy as np
from scipy import stats

class KPrototypes:
    def __init__(self, k, X, n_features, max_iter=100):
        self.k = k
        self.n_features = n_features
        self.max_iter = max_iter

        # TODO: Randomly select k data points from X as the initial prototypes.
        # Hint: Do not hardcode the indices of selected data points.
        # You may use Numpy's 'random.randint' function to randomly choose the initial prototypes.
        selected_idx = np.array(np.random.randint(0, X.shape[0], size=k))                                                 # ndarray of shape (k, )
        self.prototypes = X[selected_idx]                                             # ndarray of shape (k, n_features)

    def euclidean_distance(self, X, is_categ, debug_prototypes=None):
        prototypes = debug_prototypes if debug_prototypes is not None else self.prototypes
        # TODO: only use the numerical feature columns to calculate the Euclidean distance between each data point and
        #       prototypes.
        # Hints:
        #   - Notice that X and prototypes have mismatched shapes.
        #     X: (n_samples, n_features)                    prototypes: (n_prototypes, n_features)
        #   - Only the numerical feature columns are used to calculate the Euclidean distance.
        #     X: (n_samples, n_numerical_features)          prototypes: (n_prototypes, n_numerical_features)
        #   - You may use Numpy's 'count_nonzero', 'reshape', and 'linalg.norm' (or 'sqrt' & 'sum') functions.
        #   - Try broadcasting.
        # n_numer_features = np.sum(~is_categ(X),axis=1)[0]                                             # number of numerical features
        numerical_arrayX=X[:,is_categ==0]
        numerical_arrayP=prototypes[:,is_categ==0]
        dist = np.power(np.sum(np.power(numerical_arrayP[:,np.newaxis] - numerical_arrayX[np.newaxis,:], 2), axis=2), 1/2)
        return dist

    def hamming_distance(self, X, is_categ, debug_prototypes=None):
        prototypes = debug_prototypes if debug_prototypes is not None else self.prototypes
        # TODO: only use the categorical feature columns to calculate the Hamming distance between each data point and
        #       prototypes.
        # Hints:
        #   - Notice that X and prototypes have mismatched shapes.
        #     X: (n_samples, n_features)                    prototypes: (n_prototypes, n_features)
        #   - Only the categorical feature columns are used to calculate the Hamming distance.
        #     X: (n_samples, n_categorical_features)        prototypes: (n_prototypes, n_categorical_features)
        #   - You may use Numpy's 'count_nonzero', 'reshape', 'sum', and 'not_equal' functions.
        #   - Try broadcasting.
        # n_categ_features = np.sum(is_categ(X),axis=1)[0]                                         # number of categorical features
        cat_arrayX=X[:,is_categ==1]
        cat_arrayP=prototypes[:,is_categ==1]
        dist = np.sum(np.not_equal(cat_arrayP[:,np.newaxis,:],cat_arrayX[np.newaxis,:,:]),axis=2)
        return dist

    def fit_predict(self, X, is_categ):
        prev_prototypes = None
        iteration = 0

        # TODO: Set the criterion to leave the loop.
        # Hints:
        #   - The criterion to leave the loop is to satisfy either of the two conditions:
        #     1. Convergence criterion: the prototypes are the same as those in the last iteration.
        #     2. Max number of iterations: the algorithm runs to the max number of iterations, i.e., self.max_iter
        #   - You may use Numpy's 'not_equal' and 'any' function.
        while (iteration < self.max_iter) and ((prev_prototypes is None) or (not np.array_equal(self.prototypes, prev_prototypes))):

            # TODO: Assign the index of the closest prototype to each data point.
            # Hints: You may use numpy.argmin function to find the index of the closest prototype for each data point.
            numer_dist = self.euclidean_distance(X, is_categ)
            categ_dist = self.hamming_distance(X, is_categ)
            dist = numer_dist + categ_dist
            prototype_idx = np.argmin(dist,axis=0)

            prev_prototypes = self.prototypes.copy()            # Push current prototypes to previous.

            # TODO: Reassign prototypes as the mean of the clusters.
            # Hints:
            #  - We mentioned a method to choose specific elements from an array.
            #  - We mentioned that there were lots of functions from NumPy or scipy.stats for statistics.
            #    mean, std, median, mode, etc. On what axis should we find the statistics?
            #  - 'np.mean' and 'stats.mode' has different return shape. See how 'np.squeeze' works.
            for i in range(self.k):
                # A boolean array of shape (n_samples,). e.g., [False, True] means the second data sample is assigned to
                # cluster i but the first data sample is not.
                assigned_idx = (prototype_idx == i)
                if np.count_nonzero(assigned_idx) == 0:
                    continue

                if np.count_nonzero(~is_categ) > 0:
                    # Update the prototypes
                    self.prototypes[i,is_categ==0] = np.mean(X[assigned_idx,:][:, is_categ==0], axis=0)
                    
                if np.count_nonzero(is_categ) > 0:
                    # The returned ndarray of stats.mode does not have the same shape as the 'np.mean' function
                    categ_mode, _ = stats.mode(X[assigned_idx,:][:,is_categ==1], axis=0)

                    # Convert this returned ndarray to the same shape as the 'np.mean' function before updating the prototypes
                    self.prototypes[i, is_categ==1] = np.squeeze(categ_mode)

            iteration += 1
        return prototype_idx

def SSE(X, y, k, centroids):
    sse = 0
    # TODO: For each cluster, calculate the distance (square of difference, i.e. Euclidean/L2-distance) of samples to
    #  the datapoints and accumulate the sum to `sse`. (Hints: use numpy.sum and for loop)
    # Hints:
    #   - X is a Numpy 2D array with shape (num_datapoints, ndim), representing the data points.
    #   - y is a Numpy 1D array with shape (num_datapoints, ), representing which cluster (or which centroid) each data
    #   point correspond to.
    #   - This is very similar to the last TODO of Task 1
    for i in range(k):
        cluster_points = X[y == i]
        dist = np.sum(np.power(cluster_points[:,np.newaxis] - centroids[np.newaxis,i], 2), axis=2)
        sse += np.sum(dist)
    return sse
