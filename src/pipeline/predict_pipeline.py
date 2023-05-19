import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, pred_days: int):
        try:
            model_path='artifacts\model.h5'
            test_data=pd.read_csv('./artifacts/test.csv')
            model=load_object(file_path=model_path)

            scaler=MinMaxScaler(feature_range=(0,1))
            test_data=scaler.fit_transform(np.array(test_data).reshape(-1,1))

            test_len = len(test_data)
            print("actual test_len: ",test_len)
            test_len = test_len-10
            print("test_len: ",test_len)

            x_input=test_data[test_len:].reshape(1,-1)
            x_input.shape
                        
            temp_input=list(x_input)
            temp_input=temp_input[0].tolist()

            # days = 7
            # self.pred_days = int(pred_days)
            days = int(pred_days)
            print("type",type(days))
            lst_output=[]
            n_steps=10
            i=0
            while(i<days):
                if(len(temp_input)>10):
                    #print(temp_input)
                    x_input=np.array(temp_input[1:])
                    print("{} day input {}".format(i,x_input))
                    x_input=x_input.reshape(1,-1)
                    x_input = x_input.reshape((1, n_steps, 1))
                    #print(x_input)
                    yhat = model.predict(x_input, verbose=0)
                    print("{} day output {}".format(i,yhat))
                    temp_input.extend(yhat[0].tolist())
                    temp_input=temp_input[1:]
                    #print(temp_input)
                    lst_output.extend(yhat.tolist())
                    i=i+1
                else:
                    x_input = x_input.reshape((1, n_steps,1))
                    yhat = model.predict(x_input, verbose=0)
                    print(yhat[0])
                    temp_input.extend(yhat[0].tolist())
                    print(len(temp_input))
                    lst_output.extend(yhat.tolist())
                    i=i+1
    
            GRU_df3 = scaler.inverse_transform(lst_output)
            print("Future 7 days close price prediction for GRU: ", GRU_df3)
            return GRU_df3
    
        except Exception as e:
            raise CustomException(e,sys)
