import sys
from dataclasses import dataclass

import os
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','proprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):

        try:
            numerical_columns = ['Close']

            num_pipeline = Pipeline(
                steps=[
                ("scaler",MinMaxScaler(feature_range=(0,1)))
                ]
            )
            logging.info("Apply Preprocessing")

            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline,numerical_columns)]
            )
            logging.info("Preprocessing part done")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    # convert an array of values into a dataset matrix
    def create_dataset(self,dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
        
    def initiate_data_transformation(self,train_path,test_path,close_price):
        
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)
            close_price= pd.read_csv(close_price)
            
            train_data_len = len(train_df.values)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            # train_data = train_df[0:train_data_len, :]

            input_feature_train_arr = preprocessing_obj.fit_transform(train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(test_df)
            input_feature_close_price=preprocessing_obj.fit_transform(close_price)

            # reshape into X=t,t+1,t+2,t+3 and Y=t+4
            # logging.info("Start Creating a X_train and y_train data")
            # time_step = 10
            # X_train, y_train = self.create_dataset(train_data, time_step)
            # print(X_train.shape), print(y_train.shape)

            # # reshape input to be [samples, time steps, features] which is required for LSTM
            # X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
            # logging.info("Training data part is done.")

            # logging.info("Start Creating a X_test and y_test data")

            # test_data = preprocessing_obj[train_data_len - 11:, :]
            # test_data.shape

            # X_test, y_test = self.create_dataset(test_data, time_step)
            # print(X_test.shape), print(y_test.shape)

            # X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
            # logging.info("Test data part is done.")

            # input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            # target_feature_train_df=train_df[target_column_name]

            # input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            # target_feature_test_df=test_df[target_column_name]

            # logging.info (
            #     f"Applying preprocessing object on training dataframe and testing dataframe."
            # )


            # input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # train_arr = np.c_[
            #     input_feature_train_arr, np.array(target_feature_train_df)
            # ]

            # test_arr = np.c_[
            #     input_feature_test_arr, np.array(target_feature_test_df)
            # ]

            # logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("save Preprocessing.pkl file is done!")
            logging.info("data_transformation.py part is done!")
            return (
                input_feature_train_arr,
                input_feature_test_arr,input_feature_close_price,self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)