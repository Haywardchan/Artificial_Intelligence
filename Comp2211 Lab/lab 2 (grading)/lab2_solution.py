# -*- coding: utf-8 -*-

# **COMP 2211 Exploring Artificial Intelligence** #
## Lab 2 Naive Bayes Classifier ##

# from google.colab import drive
# drive.mount("/content/drive")

# %cd "/content/drive/MyDrive/lab2"
# %ls

# import numpy as np
# train_dataset = np.load("train_dataset.npy")
# test_dataset = np.load("test_dataset.npy")
# train_labels = np.load("train_labels.npy")
# test_labels = np.load("test_labels.npy")

class NaiveBayesClassifier:
  def __init__(self):
    self.train_dataset = None
    self.train_labels = None
    self.train_size = 0
    self.num_features = 0
    self.num_classes = 0

  def fit(self, train_dataset, train_labels):
    self.train_dataset = train_dataset
    self.train_labels = train_labels
    self.train_size = train_dataset.shape[0]
    self.num_features = train_dataset.shape[1]
    self.num_classes = np.amax(train_labels) + 1
  
  def estimate_class_prior(self):
    deltas = (np.arange(self.num_classes) == self.train_labels.reshape(-1, 1))
    class_count = np.sum(deltas, axis=0)
    class_prior = (class_count + 1) / (np.sum(class_count) + self.num_classes)
    return class_prior

  def estimate_likelihoods(self):
    deltas = (np.arange(self.num_classes) == self.train_labels.reshape(-1, 1))
    feature_count = np.dot(self.train_dataset.transpose(), deltas.astype(int))
    likelihoods = (feature_count + 1) / (np.sum(deltas, axis = 0) + 2)
    return likelihoods

  def predict(self, test_dataset):
    class_prior = self.estimate_class_prior()
    yes_likelihoods = self.estimate_likelihoods()
    no_likelihoods = 1 - yes_likelihoods
    log_class_prior = np.log(class_prior)
    log_yes_likelihoods = np.log(yes_likelihoods)
    log_no_likelihoods = np.log(no_likelihoods)
    log_prob = log_class_prior + np.dot(test_dataset, log_yes_likelihoods) + np.dot(test_dataset == False, log_no_likelihoods)
    test_predict = np.argmax(log_prob, axis=1)
    return test_predict


import numpy as np
if __name__ == '__main__':
  train_dataset = np.load("train_dataset.npy")
  test_dataset = np.load("test_dataset.npy")
  train_labels = np.load("train_labels.npy")
  test_labels = np.load("test_labels.npy")

  nb_model = NaiveBayesClassifier()
  nb_model.fit(train_dataset, train_labels)
  print(f"After fitting the training data, the train size is\
  {nb_model.train_size}, the number of features is {nb_model.num_features},\
  the number of class labels is {nb_model.num_classes}.") # should be 900, 2642, 3
  class_prior = nb_model.estimate_class_prior()
  print(f"The class priors are {class_prior}.") # should be [0.51495017 0.26135105 0.22369878]
  likelihoods = nb_model.estimate_likelihoods()
  print(f"The likelihoods of the first 5 features are \n {likelihoods[:5, :]}.") # should be [[0.00214592 0.00843882 0.00492611]
                                                                                 #            [0.00429185 0.00421941 0.00492611]
                                                                                 #            [0.00214592 0.00421941 0.00492611]
                                                                                 #            [0.00214592 0.00843882 0.00492611]
                                                                                 #            [0.00214592 0.00421941 0.00492611]]
  test_predict = nb_model.predict(test_dataset)
  print(f"The predictions for test data are:\n {test_predict}") # should be [0 0 1 0 0 0 1 0 0 0 0 2 0 0 1 0 0 0 0 1 1 0 0 0 0 2 0 0 0 1 2 0 0 0 0 0 0
                                                                #            0 0 1 1 0 0 0 0 1 0 0 1 0 0 0 0 0 1 1 0 1 0 0 2 1 0 0 0 1 0 0 0 0 0 1 0 0
                                                                #            0 1 0 0 0 1 0 0 1 0 1 0 0 0 0 0 1 2 0 0 0 2 0 0 1 0]


  accuracy_score = np.sum(test_predict == test_labels) / test_labels.shape[0]

  print(accuracy_score) # should be 0.62


# drive.flush_and_unmount()