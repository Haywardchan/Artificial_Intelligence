# import sys
# print(sys.version)#3.9.5
# """ a=r'Hi\nhehi'
# print(a.decode('utf-8'))
# print(type(range(5))) """
# a=[1,2,3,4]
# b=[4,5,6]
# print(*zip(a,b))



  # t2=np.zeros((num_classes,))
  # row=np.zeros((num_classes,))
  # col=np.zeros((num_classes,))
  # for i in np.arange(num_classes):
  #   row[i]=np.sum(confusion_matrix[i,:])
  #   col[i]=np.sum(confusion_matrix[:,i])
  #   t2[i]=row[i]*col[i]
  # n=np.sum(diagonal_matrix)*test_labels.shape[0]-np.sum(t2)
  # d=(np.power(test_labels.shape[0],2)-np.sum(np.power(col,2)))*(np.power(test_labels.shape[0],2)-np.sum(np.power(row,2)))
  # # mcc=n/d
  # return n/np.sqrt(d)

  
  # def find_k_nearest_neighbor_labels(self, test_dataset):
  #   # TODO
  #   # return k_nearest_neighbor_labels # shape: (num_test_samples, self.k)
  #   dist=self.compute_Minkowski_distance(test_dataset)
    # sort=np.argsort(dist,axis=1)
    # labels=np.take(train_labels,sort[:,:self.k])
  #   return labels

import sklearn


