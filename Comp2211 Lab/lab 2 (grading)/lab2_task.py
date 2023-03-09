import numpy as np

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
    # TODO
    self.train_size = np.array(train_dataset).shape[0]
    self.num_features = np.array(train_dataset).shape[1]
    self.num_classes = np.amax(train_labels)+1
  
  def estimate_class_prior(self):
    # TODO
    # return class_prior
    train_0=np.sum(train_labels==0)+1
    train_1=np.sum(train_labels==1)+1
    train_2=np.sum(train_labels==2)+1
    return np.array([train_0,train_1,train_2])/(self.train_size+3)

  def estimate_likelihoods(self):
    # TODO
    # return likelihoods
    label_feature_true_counts = np.array([(np.sum(train_dataset[train_labels == label_type, feature])+1) 
                                for feature in range(self.num_features)
                                for label_type in np.unique(train_labels)])
    num_of_labels=[np.sum(train_labels==0)+2,np.sum(train_labels==1)+2,np.sum(train_labels==2)+2]
    likelihood = label_feature_true_counts.reshape((self.num_features, -1))/num_of_labels
    return likelihood

  def predict(self, test_dataset):
    class_prior = self.estimate_class_prior()
    yes_likelihoods = self.estimate_likelihoods()
    no_likelihoods = 1 - yes_likelihoods
    # TODO
    # return test_predict
    posterior_p=np.array(np.log(class_prior)+
                          np.dot(test_dataset,np.log(yes_likelihoods))+
                          np.dot(1-test_dataset,np.log(no_likelihoods)))
    return np.argmax(posterior_p,axis=1)

### The following part can be deleted or be uncommented. Deleting or commenting out them would be the easiest way.
### If you do not want to comment them out, make sure all other codes are under the indent of if __name__ == '__main__'.
# if __name__ == '__main__':
#   train_dataset = np.load("train_dataset.npy")
#   test_dataset = np.load("test_dataset.npy")
#   train_labels = np.load("train_labels.npy")
#   test_labels = np.load("test_labels.npy")

#   nb_model = NaiveBayesClassifier()
#   nb_model.fit(train_dataset, train_labels)
#   print(f"After fitting the training data, the train size is\
#   {nb_model.train_size}, the number of features is {nb_model.num_features},\
#   the number of class labels is {nb_model.num_classes}.") # should be 900, 2642, 3
#   class_prior = nb_model.estimate_class_prior()
#   print(f"The class priors are {class_prior}.") # should be [0.51495017 0.26135105 0.22369878]
#   likelihoods = nb_model.estimate_likelihoods()
#   print(f"The likelihoods of the first 5 features are \n {likelihoods[:5, :]}.") # should be [[0.00214592 0.00843882 0.00492611]
#                                                                                  #            [0.00429185 0.00421941 0.00492611]
#                                                                                  #            [0.00214592 0.00421941 0.00492611]
#                                                                                  #            [0.00214592 0.00843882 0.00492611]
#                                                                                  #            [0.00214592 0.00421941 0.00492611]]
#   test_predict = nb_model.predict(test_dataset)
#   print(f"The predictions for test data are:\n {test_predict}") # should be [0 0 1 0 0 0 1 0 0 0 0 2 0 0 1 0 0 0 0 1 1 0 0 0 0 2 0 0 0 1 2 0 0 0 0 0 0
#                                                                 #            0 0 1 1 0 0 0 0 1 0 0 1 0 0 0 0 0 1 1 0 1 0 0 2 1 0 0 0 1 0 0 0 0 0 1 0 0
#                                                                 #            0 1 0 0 0 1 0 0 1 0 1 0 0 0 0 0 1 2 0 0 0 2 0 0 1 0]


#   accuracy_score = np.sum(test_predict == test_labels) / test_labels.shape[0]

#   print(accuracy_score) # should be 0.62
