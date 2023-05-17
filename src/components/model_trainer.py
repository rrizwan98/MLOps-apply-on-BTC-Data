import os
import sys
from dataclasses import dataclass

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Flatten, MaxPool2D, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from keras.utils import to_categorical
from keras.models import Sequential
import tensorflow as tf
import pandas as pd
import numpy as np
import keras


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
logging.info("necessary libraries are installed in mode_trainer.py")

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train_data_path, y_train_data_path, X_test_data_path, y_test_data_path):
        try:
            logging.info("Split training and test input data")
            print(X_train_data_path)
            X_train = np.load(X_train_data_path,allow_pickle=True)
            y_train = np.load(y_train_data_path,allow_pickle=True)
            X_test = np.load(X_test_data_path,allow_pickle=True)
            y_test = np.load(y_test_data_path,allow_pickle=True)
            # print(X_train)
# Design a custom Convolutional Neural Network Model
            logging.info("creating a custom Convolutional Neural Network Model")           
            model = Sequential()
            model.add(Conv2D(16,kernel_size=(3,3),activation='relu',input_shape=X_train[0].shape))
            model.add(BatchNormalization())
            model.add(MaxPool2D(2,2))
            model.add(Dropout(0.3))

            model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPool2D(2,2))
            model.add(Dropout(0.3))


            model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPool2D(2,2))
            model.add(Dropout(0.4))

            model.add(Flatten())

            model.add(Dense(128,activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            model.add(Dense(1,activation='sigmoid'))
            model.summary()

# model compilation
            logging.info("Model Compilation started")
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# define batch size & Epochs
            logging.info("Define Batch size and Epochs")
            batch_size = 16
            epochs = 1   
            logging.info("Start Model Training")
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                           model=model, batch_size= batch_size, epochs=epochs)

            ## To get best model score from dict
            best_model = model_report
            logging.info("Predict the train and test dataset")
            y_train_pred = best_model.predict(X_train)
            y_test_pred =  best_model.predict(X_test)
            prediction = np.array(y_test_pred)
            prediction = prediction.argmax(axis=1)
## Train predicted score
            # logging.info("confusion_matrix, accuracy_score & f1_score for traing data")
            logging.info("print predicted and actual results")
            # train_cm=confusion_matrix(y_train,y_train_pred.round(), normalize=False)
            # train_model_acc_score = accuracy_score(y_train,y_train_pred.round(), normalize=False)
            # train_model_f1_score = f1_score(y_train,y_train_pred.round(), normalize=False)
            print("Actual Results:", y_test)
            # print("predicted Results:", prediction)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            # best_model.save(self.model_trainer_config.trained_model_file_path)

## Test predicted score
            # logging.info("confusion_matrix, accuracy_score & f1_score for test data")
            # test_cm=confusion_matrix(y_test,y_test_pred.round(), normalize=False)
            # test_model_acc_score = accuracy_score(y_test,y_test_pred.round(), normalize=False)
            # test_model_f1_score = f1_score(y_test,y_test_pred.round(), normalize=False)
            
            # logging.info("print confusion_matrix, accuracy_score & f1_score ")
            # print("********* Training Score ***********")
            # print(train_cm)
            # print(train_model_acc_score)
            # print(train_model_f1_score)

            # print("********* Testing Score ***********")
            # print(test_cm)
            # print(test_model_acc_score)
            # print(test_model_f1_score)
            # ## to get best model name from dict

            # best_model_name = list(model_report.keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]
            # best_model = model_report.model

            # if best_model_score < 0.5:
            #     raise CustomException("Neural Network is not train properly")
            # logging.info(f"Best found model on both training and testing dataset")

            # save_object(
            #     best_model.save(self.model_trainer_config.trained_model_file_path)
            # )
            
            logging.info("Model Training part is completed")
            return y_test_pred
        except Exception as e:
            raise CustomException(e,sys)
