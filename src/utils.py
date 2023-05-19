import os
import sys

import tensorflow as tf
from keras.models import Sequential, load_model, save_model
from keras import layers
from keras.optimizers import RMSprop
import numpy as np 
import pandas as pd
import dill
import joblib
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

from src.exception import CustomException

def save_object(file_path, model):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        # with open(file_path, "wb") as file_obj:
        #     dill.dump(obj, file_obj)
        save_model(model, file_path)
        # model.save(file_path)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, batch_size, epochs):
    try:
        report = {}

        
        print("print train and test data in utils file")
        # print(X_train.shape)
        # print(y_train.shape)
                
        # print(X_test.shape)
        # print(y_test.shape)

        print("model summary in utils file")
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.summary()

            model.compile(optimizer=RMSprop(), loss='mae')
            model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=epochs,batch_size=batch_size,verbose=1)

            return model
    
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        # with open(file_path, "rb") as file_obj:
        #     return dill.load(file_obj)
        return load_model(file_path)

    except Exception as e:
        raise CustomException(e, sys)


