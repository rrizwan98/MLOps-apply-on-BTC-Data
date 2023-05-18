import os
import sys
from dataclasses import dataclass

import math
from sklearn.metrics import mean_squared_error
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Flatten, MaxPool2D, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from keras.utils import to_categorical
from keras.models import Sequential
import tensorflow as tf
import pandas as pd
import numpy as np
import keras

from src.models import BiGRU_model
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
logging.info("necessary libraries are installed in mode_trainer.py")

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def create_dataset(self, dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    
    def initiate_model_trainer(self, train_trns, test_trns,close_price):
        try:
            logging.info("Split training and test input data")
            ##splitting dataset into train and test split
            training_size=int(len(close_price)*0.85)
            test_size=len(close_price)-training_size
            train_data,test_data=close_price[0:training_size,:],close_price[training_size:len(close_price),:1]
            print("train and test data lenght",training_size,test_size)

            # reshape into X=t,t+1,t+2,t+3 and Y=t+4
            time_step = 10
            X_train, y_train = self.create_dataset(train_data, time_step)
            X_test, ytest = self.create_dataset(test_data, time_step)

            logging.info("print train and test data")
            print("X_train shape",X_train.shape), print("y_train shape",y_train.shape)
            print("X_test shape",X_test.shape), print("y_test shape",ytest.shape)

            logging.info("reshape input to be [samples, time steps, features] which is required for LSTM, GRU")
            # reshape input to be [samples, time steps, features] which is required for LSTM
            X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
            X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

            input_shape=(X_train.shape[1],1)
            models = {
                "Random Forest": BiGRU_model(input_shape)
            }

            # define batch size & Epochs
            logging.info("Define Batch size and Epochs")
            batch_size = 16
            epochs = 15   
            logging.info("Start Model Training")
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=ytest,
                                           models=models, batch_size= batch_size, epochs=epochs)

            ## To get best model score from dict
            best_model = model_report
            logging.info("Predict the train and test dataset")
            biGRU_train_predict=best_model.predict(X_train)
            biGRU_test_predict=best_model.predict(X_test)

            ### Calculate RMSE performance metrics
            
            train_mse = math.sqrt(mean_squared_error(y_train,biGRU_train_predict))
            print("train_mse: ",train_mse)

            ### Test Data RMSE
            test_mse = math.sqrt(mean_squared_error(ytest,biGRU_test_predict))
            print("test_mse: ",test_mse)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            logging.info("Model Training part is completed")
            return X_train.shape
        except Exception as e:
            raise CustomException(e,sys)
