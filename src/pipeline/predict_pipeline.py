import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from keras.models import load_model

def preprocessing(feature):
        X = []   
        img = tf.keras.utils.load_img(feature,target_size=(400,400,3))
        img = tf.keras.utils.img_to_array(img)
        img = img/255
        X.append(img)
        X= np.array(X)
        print(X.shape)
        return X

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.h5")
            model=tf.keras.models.load_model("artifacts\model.h5")
            test_img=preprocessing(features)
            preds=model.predict(test_img)
            preds=np.argmax(preds, axis=1)
            if preds==0:
                preds="Business-Finance-Law"
            elif preds==1:
                preds="Childrens-Books"
            else:
                preds="None"

            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
# class CustomData:
#     def __init__(self,img_path):
        
#         self.age = age

#         self.sex = sex

#         self.cp = cp

#         self.tredtbps = trestbps

#         self.chol = chol

#         self.fbs = fbs

#         self.restecg = restecg

#         self.thalach = thalach

#         self.exang = exang

#         self.oldpeak = oldpeak

#         self.slope = slope

#         self.ca = ca

#         self.thal = thal

#     def get_data_as_data_frame(self):
#         try:
#             custom_data_input_dict = {
#                 "age":[self.age],
#                 "sex":[self.sex],
#                 "cp":[self.cp],
#                 "trestbps":[self.tredtbps],
#                 "chol":[self.chol],
#                 "fbs":[self.fbs],
#                 "restecg":[self.restecg],
#                 "thalach":[self.thalach],
#                 "exang":[self.exang],
#                 "oldpeak":[self.oldpeak],
#                 "slope":[self.slope],
#                 "ca":[self.ca],
#                 "thal":[self.thal]
#             }

#             return pd.DataFrame(custom_data_input_dict)
    
        # except Exception as e:
        #     raise CustomException(e,sys)
