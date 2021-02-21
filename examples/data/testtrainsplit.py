
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def datasplit(file,randomstate=3,percentage=0.2):
  data11 = pd.read_csv(file, header=None)
  data11 = data11.drop(data11.columns[[0]], axis=1)
  final_column_name = np.array(data11.columns)[-1]
  data_sorted = data11.loc[data11[final_column_name].argsort()].reset_index(drop=True)
  data_one = data_sorted[data_sorted[final_column_name]==1]
  data_zero = data_sorted[data_sorted[final_column_name]==0].reset_index(drop=True)
  train_index,test_index = train_test_split(np.arange(data_zero.shape[0]),
                                          test_size=percentage,
                                          random_state=np.random.randint(0,100,1)[0])
  training_data = data_zero.loc[train_index]
  data_zero_test = data_zero.loc[test_index]
  testing_data = pd.concat([data_one,data_zero_test])
  return training_data,testing_data

