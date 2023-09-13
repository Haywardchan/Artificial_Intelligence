import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy import stats


class KNNModel:
  def __init__(self, k, p):
    self.k = k
    self.p = p

  def fit(self, train_dataset, train_labels):
    self.train_dataset = train_dataset # shape: (num_train_samples, dimension)
    self.train_labels = train_labels # shape: (num_train_samples, )

  def compute_Minkowski_distance(self, test_dataset):
    train_exp = np.expand_dims(self.train_dataset, axis=0)
    test_exp = np.expand_dims(test_dataset, axis=1)
    dist = np.power(np.sum(np.power(train_exp - test_exp, self.p), axis=2), 1/self.p)
    return dist # shape: (num_test_samples, num_train_samples)

  def find_k_nearest_neighbor_labels(self, test_dataset):
    dist = self.compute_Minkowski_distance(test_dataset)
    k_nearest_neighbor_labels = np.take(self.train_labels, np.argsort(dist, axis=1)[:, :self.k])
    return k_nearest_neighbor_labels # shape: (num_test_samples, self.k)
    
  def predict(self, test_dataset):
    # TODO
    # return test_predict # shape: (num_test_samples, )
    labels=self.find_k_nearest_neighbor_labels(test_dataset)
    # labels=np.array([[2,0,1,1,1,2,2,2,1]])test for tie data
    predictions=np.squeeze(stats.mode(labels, axis=1, keepdims=False))[0]
    return predictions


def generate_confusion_matrix(test_predict, test_labels):
  # TODO
  # return confusion_matrix # shape: (num_classes, num_classes)
  num_classes=int(np.amax(test_labels)+1)
  # or using len(np.unique(train_labels))
  confusion_matrix=np.zeros((num_classes, num_classes))
  for i in range(num_classes):
    for j in range(num_classes):
      confusion_matrix[j, i] = np.sum(np.logical_and(test_predict == i, test_labels == j))
  return confusion_matrix


def calculate_accuracy(test_predict, test_labels):
  # TODO
  # return accuracy # dtype: float
  accuracy = np.sum(test_predict == test_labels) / test_labels.shape[0]
  return accuracy

def calculate_precision(test_predict, test_labels):
  # TODO
  # return precision # dtype: float
  num_classes=np.amax(test_labels)+1
  confusion_matrix=generate_confusion_matrix(test_predict,test_labels)
  diagonal_matrix=confusion_matrix.diagonal()
  precision=np.zeros((num_classes,))
  for i in np.arange(num_classes):
    precision[i]=np.sum(diagonal_matrix[i]/
                    np.sum(confusion_matrix[:,i]))
  precision= np.average(precision)
  return precision

def calculate_recall(test_predict, test_labels):
  # TODO
  # return recall # dtype: float
  num_classes=np.amax(test_labels)+1
  confusion_matrix=generate_confusion_matrix(test_predict,test_labels)
  diagonal_matrix=confusion_matrix.diagonal()
  recall=np.zeros((num_classes,))
  for i in np.arange(num_classes):
    recall[i]=np.sum(diagonal_matrix[i]/
                    np.sum(confusion_matrix[i,:]))
  recall= np.average(recall)
  return recall

def calculate_macro_f1(test_predict, test_labels):
  # TODO
  # return macro_f1 # dtype: float
  precision=calculate_precision(test_predict,test_labels)
  recall=calculate_recall(test_predict,test_labels)
  macro_f1 = 2*precision*recall/(precision+recall)
  return macro_f1

def calculate_MCC_score(test_predict, test_labels):
  # TODO
  # return MCC_score # dtype: float
  confusion_matrix=generate_confusion_matrix(test_predict,test_labels)
  diagonal_matrix=confusion_matrix.diagonal()
  tp = np.diagonal(confusion_matrix)
  fp = np.sum(confusion_matrix, axis=0)-tp
  fn = np.sum(confusion_matrix, axis=1)-tp
  MCC_score = (np.sum(diagonal_matrix)*test_labels.shape[0]-np.sum((tp + fp)*(tp + fn)))/np.sqrt((test_labels.shape[0]**2-np.sum((tp+fp)**2))*(test_labels.shape[0]**2-np.sum((tp+fn)**2)))
  return MCC_score


class DFoldCV:
  def __init__(self, X, y, k_list, p_list, d, eval_metric):
    self.X = X
    self.y = y
    self.k_list = k_list
    self.p_list = p_list
    self.d = d
    self.eval_metric = eval_metric

  def generate_folds(self):
    # TODO
    # return train_d_folds, test_d_folds # type: tuple
    # concatenate the labels to validation data sets
    datawl=np.concatenate((self.X,self.y[:,np.newaxis]),axis=1)
    # split dataset into D splits
    # folds=np.array_split(datawl,self.d)
    #pack each with diff permuatations
    data=self.X.shape
    spliting_point=[(splits * data[0])//self.d for splits in range(self.d+1)]
    train_d_folds=[np.concatenate([datawl[:spliting_point[i]], datawl[spliting_point[i+1]:]],axis=0)
                    for i in range(self.d)]
    test_d_folds=[datawl[spliting_point[i]:spliting_point[i+1]] for i in range(self.d)]
    return train_d_folds, test_d_folds

  
  def cross_validate(self):
    # TODO
    # return scores # shape: (length of self.k_list, length of self.p_list, self.d)
    scores=np.zeros((len(self.k_list),len(self.p_list),self.d))
    train_d_folds, test_d_folds=self.generate_folds()
    for ki, k in enumerate(self.k_list):
      for pi, p in enumerate(self.p_list):
        knn_model=KNNModel(k,p)
        for d, (traindata, testdata) in enumerate(zip(train_d_folds,test_d_folds)):
            knn_model.fit(traindata[:,:-1],traindata[:,-1])
            predictions=knn_model.predict(testdata[:,:-1])
            scores[ki,pi,d]=self.eval_metric(predictions,testdata[:,-1])
    return scores
  
  def validate_best_parameters(self):
    # TODO
    scores=self.cross_validate()
    avg_scores = np.average(scores, axis=2)
    ki, pi = np.unravel_index(np.argmax(avg_scores, axis=None), avg_scores.shape)
    k_best, p_best = self.k_list[ki], self.p_list[pi]
    return k_best, p_best # type: tuple


### The following part can be deleted or be uncommented. Deleting or commenting out them would be the easiest way.
### If you do not want to comment them out, make sure all other codes are under the indent of if __name__ == '__main__'.
# if __name__ == '__main__':
#   train_dataset = sparse.load_npz("train_dataset.npz")
#   test_dataset = sparse.load_npz("test_dataset.npz")
#   train_dataset = train_dataset.toarray()
#   test_dataset = test_dataset.toarray()
#   train_labels = np.load("train_labels.npy")
#   test_labels = np.load("test_labels.npy")

#   knn_model = KNNModel(10, 2)
#   knn_model.fit(train_dataset, train_labels)
#   dist = knn_model.compute_Minkowski_distance(test_dataset)
#   print(f"The Minkowski distance between the first five test samples and the first five training samples are:\n {dist[ : 5, : 5]}") # should be [[1.40488545 1.41421356 1.40473647 1.41421356 1.40205505]
#                                                                                                                                     #            [1.40172965 1.41421356 1.40153004 1.41421356 1.39793611]
#                                                                                                                                     #            [1.40573171 1.41421356 1.40559629 1.41421356 1.40315911]
#                                                                                                                                     #            [1.40403747 1.41421356 1.40387491 1.41421356 1.40094856]
#                                                                                                                                     #            [1.41421356 1.39611886 1.41421356 1.39841935 1.41421356]]
#   k_nearest_neighbor_labels = knn_model.find_k_nearest_neighbor_labels(test_dataset)
#   print(f"The k nearest neighbor labels for the first five test samples are:\n {k_nearest_neighbor_labels[ : 5, :]}") # should be [[0 1 1 1 1 2 0 0 0 2]
#                                                                                                                       #            [2 1 1 0 0 0 0 0 0 0]
#                                                                                                                       #            [1 0 0 1 1 2 1 1 0 1]
#                                                                                                                       #            [1 1 0 2 2 1 0 1 1 0]
#                                                                                                                       #            [2 2 2 2 2 1 2 0 1 2]]
#   test_predict = knn_model.predict(test_dataset)
#   print(f"The predictions for test data are:\n {test_predict}") # should be [0 0 1 1 2 0 0 0 0 0 0 2 0 0 0 0 0 2 0 1 1 0 0 1 0 0 0 2 2 2 2 0 0 0 0 0 0
#                                                                 # 0 0 0 0 0 2 0 0 2 2 0 1 0 0 1 0 0 0 0 0 0 0 1 2 1 2 0 0 0 0 0 0 0 0 0 0 2
#                                                                 # 0 1 0 0 1 0 0 0 2 0 1 0 0 2 0 1 0 2 1 0 0 2 0 0 1 0]
#   confusion_matrix = generate_confusion_matrix(test_predict, test_labels)
#   print(f"The confusion matrix is:\n {confusion_matrix}") # should be [[48.  3.  1.]
#                                                           #             [16. 11. 10.]
#                                                           #             [ 4.  1.  6.]]
#   accuracy = calculate_accuracy(test_predict, test_labels)
#   print(f"The accuracy is: {accuracy}") # should be 0.65
#   precision = calculate_precision(test_predict, test_labels)
#   print(f"The macro average precision is: {precision}") # should be 0.5973856209150327
#   recall = calculate_recall(test_predict, test_labels) 
#   print(f"The macro average recall is: {recall}") # should be 0.5886095886095887
#   macro_f1 = calculate_macro_f1(test_predict, test_labels)
#   print(f"The macro f1 score is: {macro_f1}") # should be 0.5929651346720406
#   MCC_score = calculate_MCC_score(test_predict, test_labels)
#   print(f"The MCC score is: {MCC_score}") # should be 0.4182135132877802

#   k_list = [5, 10, 15]
#   p_list = [2, 4]
#   d = 10
#   dfoldcv = DFoldCV(train_dataset, train_labels, k_list, p_list, d, calculate_MCC_score)
#   scores = dfoldcv.cross_validate()
#   print(f"The scores for the first k value and the first p value: {scores[0, 0, :]}") # should be [0.35862701 0.44284459 0.32790457 0.39646162 0.21971336 0.3317104
#                                                                                       #            0.27405523 0.3728344  0.391094   0.41420285]
#   best_param = dfoldcv.validate_best_parameters()
#   print(f"The best K value and p value are: {best_param}") # should be (10, 2)
  